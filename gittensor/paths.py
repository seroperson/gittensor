# Entrius 2025

"""Platform-aware path resolution for gittensor config and data files.

Resolution precedence (per kind):
    1. GITTENSOR_<KIND>_DIR env var        (per-kind override)
    2. GITTENSOR_HOME env var              (unified root)
    3. Legacy path if it exists on disk    (backward compat)
    4. platformdirs.user_<kind>_path       (XDG on Linux, native elsewhere)

Resolvers are pure functions (no module-level caching) so tests can toggle
env vars freely without worrying about import order.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import platformdirs

_APP = 'gittensor'

Kind = Literal['config', 'data']


def _platform_dir(kind: Kind) -> Path:
    if kind == 'config':
        return platformdirs.user_config_path(_APP)
    if kind == 'data':
        return platformdirs.user_data_path(_APP)
    raise ValueError(f'unknown path kind: {kind}')


def _resolve_dir(kind: Kind) -> Path:
    per_kind = os.environ.get(f'GITTENSOR_{kind.upper()}_DIR')
    if per_kind:
        return Path(per_kind)

    unified = os.environ.get('GITTENSOR_HOME')
    if unified:
        return Path(unified) / kind

    return _platform_dir(kind)


def _resolve_file(kind: Kind, filename: str, legacy_file: Path) -> Path:
    """Resolve a path to a specific file, preferring legacy_file if it exists on disk."""
    if _explicit_override(kind):
        return _resolve_dir(kind) / filename
    if legacy_file.exists():
        return legacy_file
    return _platform_dir(kind) / filename


def _explicit_override(kind: Kind) -> bool:
    """True iff the user set an env var that must win over legacy-exists."""
    return bool(os.environ.get(f'GITTENSOR_{kind.upper()}_DIR') or os.environ.get('GITTENSOR_HOME'))


def legacy_config_file() -> Path:
    """Pre-XDG CLI config location: `~/.gittensor/config.json`."""
    return Path.home() / '.gittensor' / 'config.json'


def legacy_pats_file() -> Path:
    """Pre-XDG validator PAT storage: `<repo-root>/data/miner_pats.json`."""
    return Path(__file__).resolve().parents[1] / 'data' / 'miner_pats.json'


def config_file() -> Path:
    return _resolve_file('config', 'config.json', legacy_config_file())


def pats_file() -> Path:
    return _resolve_file('data', 'miner_pats.json', legacy_pats_file())
