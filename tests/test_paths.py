# Entrius 2025

"""Tests for gittensor.paths resolution (platformdirs + overrides)."""

import platformdirs
import pytest

from gittensor import paths

_ENV_VARS = (
    'GITTENSOR_HOME',
    'GITTENSOR_CONFIG_DIR',
    'GITTENSOR_DATA_DIR',
    'XDG_CONFIG_HOME',
    'XDG_DATA_HOME',
)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Clear all relevant env vars and point HOME at a tmp dir.

    Prevents the developer's real `~/.gittensor` from bleeding into
    precedence tests. platformdirs respects $HOME, so paths resolve under
    `tmp_path` on any platform.
    """
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    return tmp_path


class TestPlatformDefaults:
    def test_config_file(self, isolated_home):
        assert paths.config_file() == platformdirs.user_config_path('gittensor') / 'config.json'

    def test_pats_file(self, isolated_home, monkeypatch):
        monkeypatch.setattr(paths, 'legacy_pats_file', lambda: isolated_home / 'nonexistent.json')
        assert paths.pats_file() == platformdirs.user_data_path('gittensor') / 'miner_pats.json'


class TestGittensorHome:
    def test_unified_root_applies_to_files(self, isolated_home, monkeypatch, tmp_path):
        root = tmp_path / 'gt_root'
        monkeypatch.setenv('GITTENSOR_HOME', str(root))
        assert paths.config_file() == root / 'config' / 'config.json'
        assert paths.pats_file() == root / 'data' / 'miner_pats.json'


class TestPerKindOverrides:
    def test_per_kind_wins_over_gittensor_home(self, isolated_home, monkeypatch, tmp_path):
        monkeypatch.setenv('GITTENSOR_HOME', str(tmp_path / 'gt_root'))
        monkeypatch.setenv('GITTENSOR_CONFIG_DIR', str(tmp_path / 'override_config'))
        assert paths.config_file() == tmp_path / 'override_config' / 'config.json'

    def test_per_kind_wins_over_platform_default(self, isolated_home, monkeypatch, tmp_path):
        override = tmp_path / 'override'
        monkeypatch.setenv('GITTENSOR_CONFIG_DIR', str(override))
        assert paths.config_file() == override / 'config.json'

    def test_gittensor_data_dir_overrides_pats_file(self, isolated_home, monkeypatch, tmp_path):
        override = tmp_path / 'override_data'
        monkeypatch.setenv('GITTENSOR_DATA_DIR', str(override))
        assert paths.pats_file() == override / 'miner_pats.json'


class TestLegacyFallback:
    def test_existing_legacy_config_wins_over_platform_default(self, isolated_home):
        legacy_dir = isolated_home / '.gittensor'
        legacy_dir.mkdir()
        (legacy_dir / 'config.json').write_text('{}')

        assert paths.config_file() == legacy_dir / 'config.json'

    def test_missing_legacy_config_falls_through_to_platform_default(self, isolated_home):
        assert paths.config_file() == platformdirs.user_config_path('gittensor') / 'config.json'

    def test_explicit_override_beats_legacy(self, isolated_home, monkeypatch, tmp_path):
        legacy_dir = isolated_home / '.gittensor'
        legacy_dir.mkdir()
        (legacy_dir / 'config.json').write_text('{}')

        override = tmp_path / 'override'
        monkeypatch.setenv('GITTENSOR_CONFIG_DIR', str(override))
        assert paths.config_file() == override / 'config.json'

    def test_existing_legacy_pats_wins_over_platform_default(self, isolated_home, monkeypatch, tmp_path):
        legacy_file = tmp_path / 'fake_repo_data_miner_pats.json'
        legacy_file.write_text('[]')
        monkeypatch.setattr(paths, 'legacy_pats_file', lambda: legacy_file)

        assert paths.pats_file() == legacy_file

    def test_gittensor_home_beats_legacy_pats(self, isolated_home, monkeypatch, tmp_path):
        legacy_file = tmp_path / 'fake_legacy_pats.json'
        legacy_file.write_text('[]')
        monkeypatch.setattr(paths, 'legacy_pats_file', lambda: legacy_file)

        root = tmp_path / 'gt_root'
        monkeypatch.setenv('GITTENSOR_HOME', str(root))
        assert paths.pats_file() == root / 'data' / 'miner_pats.json'


class TestPrecedence:
    def test_full_precedence_order(self, isolated_home, monkeypatch, tmp_path):
        """Verify: per-kind > GITTENSOR_HOME > legacy > platform default."""
        (isolated_home / '.gittensor').mkdir()
        (isolated_home / '.gittensor' / 'config.json').write_text('{}')
        monkeypatch.setenv('GITTENSOR_HOME', str(tmp_path / 'gt_home'))
        monkeypatch.setenv('GITTENSOR_CONFIG_DIR', str(tmp_path / 'per_kind'))

        # Per-kind wins
        assert paths.config_file() == tmp_path / 'per_kind' / 'config.json'

        # Drop per-kind: GITTENSOR_HOME wins
        monkeypatch.delenv('GITTENSOR_CONFIG_DIR')
        assert paths.config_file() == tmp_path / 'gt_home' / 'config' / 'config.json'

        # Drop GITTENSOR_HOME: legacy wins
        monkeypatch.delenv('GITTENSOR_HOME')
        assert paths.config_file() == isolated_home / '.gittensor' / 'config.json'

        # Remove legacy file: platform default wins
        (isolated_home / '.gittensor' / 'config.json').unlink()
        (isolated_home / '.gittensor').rmdir()
        assert paths.config_file() == platformdirs.user_config_path('gittensor') / 'config.json'


class TestLegacyHelpers:
    def test_legacy_config_file(self, isolated_home):
        assert paths.legacy_config_file() == isolated_home / '.gittensor' / 'config.json'

    def test_legacy_pats_file_is_repo_data(self):
        assert paths.legacy_pats_file().name == 'miner_pats.json'
        assert paths.legacy_pats_file().parent.name == 'data'
