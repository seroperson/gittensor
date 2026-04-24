# The MIT License (MIT)
# Copyright © 2025 Entrius

"""Tests for the pr_api_cache read-through path for MERGED PRs.

Covers:
- Repository.get_cached_pr_api_data: miss, hit (full hydration), empty head_sha, DB error
- Repository.store_pr_api_cache: round-trip serialization
- score_pull_request: cache hit skips all 3 GitHub helpers but still runs calculate_base_score;
  cache miss fetches from GitHub then persists the payload; OPEN/CLOSED never touch the cache.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from gittensor.classes import FileChange, MinerEvaluation, PRState, PullRequest
from gittensor.utils.github_api_tools import FileContentPair
from gittensor.validator.oss_contributions.scoring import score_pull_request
from gittensor.validator.storage.repository import CachedPrApiData, Repository
from gittensor.validator.utils.load_weights import LanguageConfig, RepositoryConfig, TokenConfig


def _make_token_config() -> TokenConfig:
    return TokenConfig(
        structural_bonus={'function_definition': 2.0, 'class_definition': 3.0},
        leaf_tokens={'identifier': 0.5, 'string': 0.2},
        language_configs={'py': LanguageConfig(weight=1.0, language='python')},
    )


def _make_languages() -> dict:
    return {'py': LanguageConfig(weight=1.0, language='python')}


def _make_miner_eval(uid: int = 1) -> MinerEvaluation:
    return MinerEvaluation(uid=uid, hotkey=f'hk_{uid}', github_id=str(uid), github_pat='fake-pat')


def _make_pr(state: PRState, head_ref_oid: str = 'abc123def456') -> PullRequest:
    return PullRequest(
        number=42,
        repository_full_name='owner/repo',
        uid=1,
        hotkey='hk_1',
        github_id='1',
        title='Test PR',
        author_login='alice',
        merged_at=datetime.now(timezone.utc) if state == PRState.MERGED else None,
        created_at=datetime.now(timezone.utc),
        pr_state=state,
        head_ref_oid=head_ref_oid,
        base_ref_oid='base_sha_000',
    )


def _make_master_repos() -> dict:
    return {'owner/repo': RepositoryConfig(weight=1.0)}


def _make_file_change(filename: str = 'src/app.py') -> FileChange:
    return FileChange(
        pr_number=42,
        repository_full_name='owner/repo',
        filename=filename,
        changes=10,
        additions=7,
        deletions=3,
        status='modified',
        patch='@@ diff @@',
    )


class TestGetCachedPrApiData:
    def _make_repo(self, fetchone_result):
        cursor = MagicMock()
        cursor.fetchone.return_value = fetchone_result
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        connection = MagicMock()
        connection.cursor.return_value = cursor
        return Repository(connection), cursor

    def test_returns_none_on_miss(self):
        repo, cursor = self._make_repo(fetchone_result=None)
        assert repo.get_cached_pr_api_data('owner/repo', 42, 'head_sha_abc') is None
        cursor.execute.assert_called_once()

    def test_returns_none_when_head_sha_empty(self):
        repo, cursor = self._make_repo(fetchone_result=None)
        assert repo.get_cached_pr_api_data('owner/repo', 42, '') is None
        cursor.execute.assert_not_called()

    def test_returns_none_on_db_error(self):
        cursor = MagicMock()
        cursor.execute.side_effect = RuntimeError('db exploded')
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        connection = MagicMock()
        connection.cursor.return_value = cursor
        repo = Repository(connection)
        assert repo.get_cached_pr_api_data('owner/repo', 42, 'head_sha') is None

    def test_hydrates_file_changes_and_contents_on_hit(self):
        fc_payload = [
            {
                'filename': 'src/app.py',
                'changes': 10,
                'additions': 7,
                'deletions': 3,
                'status': 'modified',
                'patch': '@@ diff @@',
                'file_extension': 'py',
                'previous_filename': None,
            }
        ]
        contents_payload = {'src/app.py': {'old': 'def f(): pass', 'new': 'def f(): return 1'}}
        repo, _ = self._make_repo(fetchone_result=(fc_payload, contents_payload))

        result = repo.get_cached_pr_api_data('owner/repo', 42, 'head_sha_abc')

        assert isinstance(result, CachedPrApiData)
        assert len(result.file_changes) == 1
        fc = result.file_changes[0]
        assert fc.filename == 'src/app.py'
        assert fc.pr_number == 42
        assert fc.repository_full_name == 'owner/repo'
        assert fc.patch == '@@ diff @@'

        assert 'src/app.py' in result.file_contents
        pair = result.file_contents['src/app.py']
        assert pair.old_content == 'def f(): pass'
        assert pair.new_content == 'def f(): return 1'

    def test_tolerates_jsonb_served_as_string(self):
        import json as _json

        fc_payload = _json.dumps(
            [
                {
                    'filename': 'x.py',
                    'changes': 1,
                    'additions': 1,
                    'deletions': 0,
                    'status': 'added',
                    'patch': None,
                    'file_extension': 'py',
                    'previous_filename': None,
                }
            ]
        )
        contents_payload = _json.dumps({'x.py': {'old': None, 'new': 'print(1)'}})
        repo, _ = self._make_repo(fetchone_result=(fc_payload, contents_payload))
        result = repo.get_cached_pr_api_data('owner/repo', 42, 'head_sha')
        assert result is not None
        assert result.file_changes[0].filename == 'x.py'
        assert result.file_contents['x.py'].new_content == 'print(1)'


class TestStorePrApiCache:
    def test_returns_false_when_head_sha_empty(self):
        repo = Repository(MagicMock())
        assert repo.store_pr_api_cache('owner/repo', 42, '', [], {}) is False

    def test_persists_payload(self):
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        connection = MagicMock()
        connection.cursor.return_value = cursor
        repo = Repository(connection)

        file_changes = [_make_file_change('src/app.py')]
        file_contents = {'src/app.py': FileContentPair(old_content='old', new_content='new')}

        ok = repo.store_pr_api_cache('owner/repo', 42, 'head_sha_abc', file_changes, file_contents)

        assert ok is True
        cursor.execute.assert_called_once()
        args = cursor.execute.call_args[0][1]
        assert args[0] == 'owner/repo'
        assert args[1] == 42
        assert args[2] == 'head_sha_abc'
        # args[3] and args[4] are psycopg2.extras.Json wrappers; their .adapted
        # attribute is the underlying Python value.
        assert args[3].adapted[0]['filename'] == 'src/app.py'
        assert args[4].adapted['src/app.py'] == {'old': 'old', 'new': 'new'}
        connection.commit.assert_called_once()


class TestScoringCachePath:
    @pytest.fixture
    def gh_helpers(self):
        """Patch the three GitHub helpers imported into scoring and yield the mocks."""
        targets = (
            'gittensor.validator.oss_contributions.scoring.get_pull_request_file_changes',
            'gittensor.validator.oss_contributions.scoring.get_merge_base_sha',
            'gittensor.validator.oss_contributions.scoring.fetch_file_contents_with_base',
        )
        with patch(targets[0]) as files, patch(targets[1]) as merge_base, patch(targets[2]) as contents:
            yield files, merge_base, contents

    @pytest.fixture
    def base_score_spy(self):
        """Wrap calculate_base_score to verify it still runs on cache hit."""
        with patch(
            'gittensor.validator.oss_contributions.scoring.calculate_base_score',
            return_value=42.0,
        ) as spy:
            yield spy

    def test_cache_hit_skips_all_github_helpers_and_still_computes_score(self, gh_helpers, base_score_spy):
        files_mock, merge_base_mock, contents_mock = gh_helpers

        cached_fc = [_make_file_change('src/app.py')]
        cached_contents = {'src/app.py': FileContentPair(old_content='old', new_content='new')}

        repo = MagicMock(spec=Repository)
        repo.get_cached_pr_api_data.return_value = CachedPrApiData(
            file_changes=cached_fc, file_contents=cached_contents
        )

        pr = _make_pr(PRState.MERGED)
        miner_eval = _make_miner_eval()

        score_pull_request(pr, miner_eval, _make_master_repos(), _make_languages(), _make_token_config(), repo=repo)

        # Zero GitHub calls on cache hit.
        files_mock.assert_not_called()
        merge_base_mock.assert_not_called()
        contents_mock.assert_not_called()

        # Scoring still ran against the cached file_contents.
        base_score_spy.assert_called_once()
        passed_contents = base_score_spy.call_args[0][3]
        assert passed_contents is cached_contents
        assert pr.base_score == 42.0

        # file_changes were hydrated onto the PR.
        assert pr.file_changes is cached_fc

        # MERGED contribution still counted.
        assert 'owner/repo' in miner_eval.unique_repos_contributed_to

        # No re-write when we already hit.
        repo.store_pr_api_cache.assert_not_called()

    def test_cache_miss_fetches_and_persists_for_merged(self, gh_helpers, base_score_spy):
        files_mock, merge_base_mock, contents_mock = gh_helpers

        fetched_fc = [_make_file_change('src/app.py')]
        files_mock.return_value = fetched_fc
        merge_base_mock.return_value = 'merge_base_sha'
        contents_mock.return_value = {'src/app.py': FileContentPair(old_content='old', new_content='new')}

        repo = MagicMock(spec=Repository)
        repo.get_cached_pr_api_data.return_value = None

        pr = _make_pr(PRState.MERGED)
        miner_eval = _make_miner_eval()

        score_pull_request(pr, miner_eval, _make_master_repos(), _make_languages(), _make_token_config(), repo=repo)

        repo.get_cached_pr_api_data.assert_called_once_with('owner/repo', 42, 'abc123def456')
        files_mock.assert_called_once()
        contents_mock.assert_called_once()

        # Persisted for the next round.
        repo.store_pr_api_cache.assert_called_once()
        args = repo.store_pr_api_cache.call_args[0]
        assert args[0] == 'owner/repo'
        assert args[1] == 42
        assert args[2] == 'abc123def456'
        assert args[3] == fetched_fc

        base_score_spy.assert_called_once()
        assert pr.base_score == 42.0

    @pytest.mark.parametrize('state', [PRState.OPEN, PRState.CLOSED])
    def test_open_and_closed_never_touch_cache(self, gh_helpers, base_score_spy, state):
        files_mock, _, contents_mock = gh_helpers
        files_mock.return_value = [_make_file_change('src/app.py')]
        contents_mock.return_value = {'src/app.py': FileContentPair(old_content='o', new_content='n')}

        repo = MagicMock(spec=Repository)
        pr = _make_pr(state)
        miner_eval = _make_miner_eval()

        score_pull_request(pr, miner_eval, _make_master_repos(), _make_languages(), _make_token_config(), repo=repo)

        repo.get_cached_pr_api_data.assert_not_called()
        repo.store_pr_api_cache.assert_not_called()
        # Still fetches from GitHub on the normal path.
        files_mock.assert_called_once()

    def test_merged_without_head_sha_skips_cache(self, gh_helpers, base_score_spy):
        files_mock, _, contents_mock = gh_helpers
        files_mock.return_value = [_make_file_change('src/app.py')]
        contents_mock.return_value = {'src/app.py': FileContentPair(old_content='o', new_content='n')}

        repo = MagicMock(spec=Repository)
        pr = _make_pr(PRState.MERGED, head_ref_oid='')
        miner_eval = _make_miner_eval()

        score_pull_request(pr, miner_eval, _make_master_repos(), _make_languages(), _make_token_config(), repo=repo)

        repo.get_cached_pr_api_data.assert_not_called()
        repo.store_pr_api_cache.assert_not_called()

    def test_no_repository_means_no_cache_path(self, gh_helpers, base_score_spy):
        files_mock, _, contents_mock = gh_helpers
        files_mock.return_value = [_make_file_change('src/app.py')]
        contents_mock.return_value = {'src/app.py': FileContentPair(old_content='o', new_content='n')}

        pr = _make_pr(PRState.MERGED)
        miner_eval = _make_miner_eval()

        # Passing repo=None (existing callers' default) must fall through cleanly.
        score_pull_request(pr, miner_eval, _make_master_repos(), _make_languages(), _make_token_config())

        files_mock.assert_called_once()
