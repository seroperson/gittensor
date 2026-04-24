"""
Repository class providing database operations for validator storage.

This module consolidates all database operations into a single Repository class,
providing clean methods for storing miners, pull requests, issues, file changes,
and miner evaluations.
"""

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from gittensor.classes import FileChange, Issue, Miner, MinerEvaluation, PullRequest
from gittensor.utils.github_api_tools import FileContentPair

from .queries import (
    BULK_UPSERT_FILE_CHANGES,
    BULK_UPSERT_ISSUES,
    BULK_UPSERT_MINER_EVALUATION,
    BULK_UPSERT_PULL_REQUESTS,
    CLEANUP_STALE_MINER_EVALUATIONS,
    CLEANUP_STALE_MINER_EVALUATIONS_BY_HOTKEY,
    CLEANUP_STALE_MINERS,
    CLEANUP_STALE_MINERS_BY_HOTKEY,
    SELECT_PR_API_CACHE,
    SET_MINER,
    UPSERT_PR_API_CACHE,
)


@dataclass
class CachedPrApiData:
    """Raw GitHub-payload cache for a MERGED PR at a fixed head_sha.

    Holds the immutable outputs of the three GitHub helpers that we skip
    on a cache hit. Scoring still re-runs over these values each round.
    """

    file_changes: List[FileChange]
    file_contents: Dict[str, FileContentPair]


def _serialize_file_change(fc: FileChange) -> dict:
    return {
        'filename': fc.filename,
        'changes': fc.changes,
        'additions': fc.additions,
        'deletions': fc.deletions,
        'status': fc.status,
        'patch': fc.patch,
        'file_extension': fc.file_extension,
        'previous_filename': fc.previous_filename,
    }


def _deserialize_file_change(d: dict, pr_number: int, repository_full_name: str) -> FileChange:
    return FileChange(
        pr_number=pr_number,
        repository_full_name=repository_full_name,
        filename=d['filename'],
        changes=d['changes'],
        additions=d['additions'],
        deletions=d['deletions'],
        status=d['status'],
        patch=d.get('patch'),
        file_extension=d.get('file_extension'),
        previous_filename=d.get('previous_filename'),
    )


class BaseRepository:
    """
    Base repository class that handles database connections and provides
    clean query execution methods.
    """

    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = logging.getLogger(self.__class__.__name__)

    @contextmanager
    def get_cursor(self):
        """
        Context manager for database cursor operations.
        Automatically handles cursor cleanup.
        """
        cursor = self.db.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def execute_command(self, query: str, params: tuple = ()) -> bool:
        """
        Execute an INSERT, UPDATE, or DELETE command.

        Args:
            query: SQL command string
            params: Query parameters tuple

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                self.db.commit()
                return True
        except Exception as e:
            self.db.rollback()
            self.logger.error(f'Error executing command: {e}')
            return False

    def set_entity(self, query: str, params: tuple) -> bool:
        """
        Insert or update an entity using the provided query.

        Args:
            query: SQL INSERT/UPDATE query with ON DUPLICATE KEY UPDATE
            params: Query parameters tuple

        Returns:
            True if successful, False otherwise
        """
        return self.execute_command(query, params)


class Repository(BaseRepository):
    """
    Consolidated repository for all database operations.
    Methods are ordered to match their usage in the storage workflow.
    """

    def __init__(self, db_connection):
        super().__init__(db_connection)

    def set_miner(self, miner: Miner) -> bool:
        """
        Insert a miner (ignore conflicts)

        Args:
            miner: Miner object to store

        Returns:
            True if successful, False otherwise
        """
        params = (miner.uid, miner.hotkey, miner.github_id)
        return self.set_entity(SET_MINER, params)

    def cleanup_stale_miner_data(self, evaluation: MinerEvaluation) -> None:
        """
        Remove stale evaluation data when a miner re-registers on a new uid/hotkey.

        Deletes miner_evaluations and miners rows for the same
        github_id but under a different (uid, hotkey) pair, ensuring only one
        evaluation per real github user exists in the database.

        Args:
            evaluation: The current MinerEvaluation being stored
        """
        if not evaluation.github_id or evaluation.github_id == '0':
            return

        params = (evaluation.github_id, evaluation.uid, evaluation.hotkey)
        eval_params = params + (evaluation.evaluation_timestamp,)

        # Clean up when same github_id re-registers on a new uid/hotkey
        self.execute_command(CLEANUP_STALE_MINER_EVALUATIONS, eval_params)
        self.execute_command(CLEANUP_STALE_MINERS, params)

        # Clean up when same (uid, hotkey) re-links to a new github_id
        reverse_params = (evaluation.uid, evaluation.hotkey, evaluation.github_id)
        reverse_eval_params = reverse_params + (evaluation.evaluation_timestamp,)
        self.execute_command(CLEANUP_STALE_MINER_EVALUATIONS_BY_HOTKEY, reverse_eval_params)
        self.execute_command(CLEANUP_STALE_MINERS_BY_HOTKEY, reverse_params)

    def get_cached_pr_api_data(
        self,
        repository_full_name: str,
        pr_number: int,
        head_sha: str,
    ) -> Optional[CachedPrApiData]:
        """Fetch the cached raw GitHub payload for a MERGED PR at head_sha.

        Returns None on miss, head_sha mismatch, or any DB/parse error.
        Callers must tolerate None and fall through to the full fetch path.
        """
        if not head_sha:
            return None

        try:
            with self.get_cursor() as cursor:
                cursor.execute(SELECT_PR_API_CACHE, (repository_full_name, pr_number, head_sha))
                row = cursor.fetchone()
        except Exception as e:
            self.logger.error(f'Error reading pr_api_cache: {e}')
            return None

        if not row:
            return None

        try:
            fc_payload, contents_payload = row
            # psycopg2 returns JSONB as a parsed dict/list by default; tolerate
            # str too in case the column is plain TEXT or a custom codec is set.
            if isinstance(fc_payload, str):
                fc_payload = json.loads(fc_payload)
            if isinstance(contents_payload, str):
                contents_payload = json.loads(contents_payload)

            file_changes = [_deserialize_file_change(d, pr_number, repository_full_name) for d in fc_payload]
            file_contents = {
                filename: FileContentPair(old_content=pair.get('old'), new_content=pair.get('new'))
                for filename, pair in contents_payload.items()
            }
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f'Corrupt pr_api_cache row for {repository_full_name}#{pr_number}: {e}')
            return None

        return CachedPrApiData(file_changes=file_changes, file_contents=file_contents)

    def store_pr_api_cache(
        self,
        repository_full_name: str,
        pr_number: int,
        head_sha: str,
        file_changes: List[FileChange],
        file_contents: Dict[str, FileContentPair],
    ) -> bool:
        """Persist the raw GitHub payload for a MERGED PR at head_sha."""
        if not head_sha:
            return False

        fc_payload = [_serialize_file_change(fc) for fc in file_changes]
        contents_payload = {
            filename: {'old': pair.old_content, 'new': pair.new_content} for filename, pair in file_contents.items()
        }

        try:
            from psycopg2.extras import Json

            with self.get_cursor() as cursor:
                cursor.execute(
                    UPSERT_PR_API_CACHE,
                    (repository_full_name, pr_number, head_sha, Json(fc_payload), Json(contents_payload)),
                )
                self.db.commit()
                return True
        except Exception as e:
            self.db.rollback()
            self.logger.error(f'Error writing pr_api_cache for {repository_full_name}#{pr_number}: {e}')
            return False

    def store_pull_requests_bulk(self, pull_requests: List[PullRequest]) -> int:
        """
        Bulk insert/update pull requests with efficient SQL conflict resolution

        Args:
            pull_requests: List of PullRequest objects to store

        Returns:
            Count of successfully stored pull requests
        """
        if not pull_requests:
            return 0

        # Prepare data for bulk insert
        values = []
        for pr in pull_requests:
            # uid is causing issues bc it keeps remaining as an np.int64
            if isinstance(pr.uid, np.integer):
                pr.uid = pr.uid.item()

            values.append(
                (
                    pr.number,
                    pr.repository_full_name,
                    pr.uid,
                    pr.hotkey,
                    pr.github_id,
                    pr.title,
                    pr.author_login,
                    pr.merged_at,
                    pr.created_at,
                    pr.pr_state.value,  # Convert PRState enum to string
                    pr.repo_weight_multiplier,
                    pr.base_score,
                    pr.issue_multiplier,
                    pr.open_pr_spam_multiplier,
                    pr.pioneer_dividend,
                    pr.pioneer_rank,
                    pr.time_decay_multiplier,
                    pr.credibility_multiplier,
                    pr.review_quality_multiplier,
                    pr.label_multiplier,
                    pr.label,
                    pr.earned_score,
                    pr.collateral_score,
                    pr.additions,
                    pr.deletions,
                    pr.commits,
                    pr.total_nodes_scored,
                    pr.merged_by_login,
                    pr.description,
                    pr.last_edited_at,
                    pr.code_density,
                    pr.token_score,
                    pr.structural_count,
                    pr.structural_score,
                    pr.leaf_count,
                    pr.leaf_score,
                )
            )

        try:
            with self.get_cursor() as cursor:
                # Use psycopg2's execute_values for efficient bulk insert
                from psycopg2.extras import execute_values

                execute_values(
                    cursor,
                    BULK_UPSERT_PULL_REQUESTS.replace('VALUES %s', 'VALUES %s'),
                    values,
                    template=None,
                    page_size=100,
                )
                self.db.commit()
                return len(values)
        except Exception as e:
            self.db.rollback()
            self.logger.error(f'Error in bulk pull request storage: {e}')
            return 0

    def store_issues_bulk(self, issues: List[Issue]) -> int:
        """
        Bulk insert/update issues with efficient SQL conflict resolution

        Args:
            issues: List of Issue objects to store

        Returns:
            Count of successfully stored issues
        """
        if not issues:
            return 0

        # Prepare data for bulk insert
        values = []
        for issue in issues:
            values.append(
                (
                    issue.number,
                    issue.pr_number,
                    issue.repository_full_name,
                    issue.title,
                    issue.created_at,
                    issue.closed_at,
                    issue.author_login,
                    issue.state,
                    issue.author_association,
                    issue.author_github_id,
                    issue.is_transferred,
                    issue.updated_at,
                    issue.discovery_base_score,
                    issue.discovery_earned_score,
                    issue.discovery_review_quality_multiplier,
                    issue.discovery_repo_weight_multiplier,
                    issue.discovery_time_decay_multiplier,
                    issue.discovery_credibility_multiplier,
                    issue.discovery_open_issue_spam_multiplier,
                )
            )

        try:
            with self.get_cursor() as cursor:
                # Use psycopg2's execute_values for efficient bulk insert
                from psycopg2.extras import execute_values

                execute_values(
                    cursor, BULK_UPSERT_ISSUES.replace('VALUES %s', 'VALUES %s'), values, template=None, page_size=100
                )
                self.db.commit()
                return len(values)
        except Exception as e:
            self.db.rollback()
            self.logger.error(f'Error in bulk issue storage: {e}')
            return 0

    def store_file_changes_bulk(self, file_changes: List[FileChange]) -> int:
        """
        Bulk insert/update file changes with efficient SQL conflict resolution

        Args:
            file_changes: List of FileChange objects to store (must include pr_number and repository_full_name)

        Returns:
            Count of successfully stored file changes
        """
        if not file_changes:
            return 0

        # Prepare data for bulk insert
        values = []
        for file_change in file_changes:
            values.append(
                (
                    file_change.pr_number,
                    file_change.repository_full_name,
                    file_change.filename,
                    file_change.changes,
                    file_change.additions,
                    file_change.deletions,
                    file_change.status,
                    file_change.patch,
                    file_change.file_extension or file_change._calculate_file_extension(),
                )
            )

        try:
            with self.get_cursor() as cursor:
                # Use psycopg2's execute_values for efficient bulk insert
                from psycopg2.extras import execute_values

                execute_values(
                    cursor,
                    BULK_UPSERT_FILE_CHANGES.replace('VALUES %s', 'VALUES %s'),
                    values,
                    template=None,
                    page_size=100,
                )
                self.db.commit()
                return len(values)
        except Exception as e:
            self.db.rollback()
            prs = {(fc.pr_number, fc.repository_full_name) for fc in file_changes}
            self.logger.error(f'Error in bulk file change storage: {e} | PRs: {prs}')
            return 0

    def set_miner_evaluation(self, evaluation: MinerEvaluation) -> bool:
        """
        Insert or update a miner evaluation.

        Args:
            evaluation: MinerEvaluation object to store

        Returns:
            True if successful, False otherwise
        """
        eval_values = [
            (
                evaluation.uid,
                evaluation.hotkey,
                evaluation.github_id,
                evaluation.failed_reason,
                evaluation.base_total_score,
                evaluation.total_score,
                evaluation.total_collateral_score,
                evaluation.total_nodes_scored,
                evaluation.total_open_prs,
                evaluation.total_closed_prs,
                evaluation.total_merged_prs,
                evaluation.total_prs,
                evaluation.unique_repos_count,
                evaluation.is_eligible,
                evaluation.credibility,
                evaluation.total_token_score,
                evaluation.total_structural_count,
                evaluation.total_structural_score,
                evaluation.total_leaf_count,
                evaluation.total_leaf_score,
                evaluation.issue_discovery_score,
                evaluation.issue_token_score,
                evaluation.issue_credibility,
                evaluation.is_issue_eligible,
                evaluation.total_solved_issues,
                evaluation.total_valid_solved_issues,
                evaluation.total_closed_issues,
                evaluation.total_open_issues,
            )
        ]

        try:
            with self.get_cursor() as cursor:
                from psycopg2.extras import execute_values

                execute_values(cursor, BULK_UPSERT_MINER_EVALUATION, eval_values)
                self.db.commit()
                return True
        except Exception as e:
            self.db.rollback()
            self.logger.error(f'Error in miner evaluation storage: {e}')
            return False
