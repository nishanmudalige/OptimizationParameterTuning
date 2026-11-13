"""Machine-independent locations of the data, models and results.

Nothing in this package hard-codes a path. Every location is resolved in this order:

1. an explicit function argument / command-line flag,
2. an environment variable (``MEMTUNE_ROOT``, ``MEMTUNE_DATA_DIR``, ``MEMTUNE_MODELS_DIR``,
   ``MEMTUNE_RESULTS_DIR``),
3. the repository layout, found from the location of this file (not from the working directory).

All paths are :class:`pathlib.Path` objects, so they work the same on Linux, macOS and Windows.
"""
from __future__ import annotations

import os
from pathlib import Path

#: File name of the merged dataset (one row per problem instance x ``mem`` value).
DATASET_NAME = "lbfgs_runs.csv"


def _find_root() -> Path:
    env = os.environ.get("MEMTUNE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "memtune").is_dir():
            return parent
    return Path.cwd().resolve()  # installed copy without the repository: fall back to cwd


def repo_root() -> Path:
    """Repository root (``MEMTUNE_ROOT`` overrides)."""
    return _find_root()


def _dir(env_name: str, default_name: str, explicit) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(env_name)
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / default_name


def data_dir(explicit=None) -> Path:
    return _dir("MEMTUNE_DATA_DIR", "data", explicit)


def models_dir(explicit=None) -> Path:
    return _dir("MEMTUNE_MODELS_DIR", "models", explicit)


def results_dir(explicit=None, create: bool = True) -> Path:
    path = _dir("MEMTUNE_RESULTS_DIR", "results", explicit)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(explicit=None) -> Path:
    """Path of the merged dataset. ``explicit`` may be the CSV file itself or a directory."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        return p / DATASET_NAME if p.is_dir() else p
    return data_dir() / DATASET_NAME
