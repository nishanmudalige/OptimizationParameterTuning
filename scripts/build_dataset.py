#!/usr/bin/env python
"""Build the merged dataset ``lbfgs_runs.csv`` (one row per problem instance x ``mem`` value).

Two steps, either of which can be skipped:

1. ``--problem-logs`` / ``--solver-logs``: collect the JSON-lines log files written by the data
   collection step into two intermediate tables, ``problem_metadata.csv`` and ``solver_runtimes.csv``.
2. Merge the two tables, mark failed runs (error -> 2x time limit, time-out -> time limit) and write
   ``lbfgs_runs.csv`` plus ``lbfgs_runs_success.csv`` (runs without any error).

Examples
--------
Merge existing intermediate tables found in ``data/`` (default)::

    python scripts/build_dataset.py

Start from raw logs, anywhere on disk::

    python scripts/build_dataset.py --problem-logs /path/to/problem_logs --solver-logs /path/to/solver_logs

Every location can also be set with ``MEMTUNE_DATA_DIR`` (see ``memtune.paths``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from memtune import paths

JOIN_KEY = ["problem", "name", "nvar"]
ERROR_COLS = ["error_problem", "error_type", "error_solver"]
EXPR_COLS = ["is_constant (ExprTree)", "is_linear (ExprTree)"]
PROBLEM_COLS_REG = ["nvar", "tree_length", "tree_depth", "time_obj", "time_grad"]
SOLVER_COLS_REG = ["mem", "neval_obj", "neval_grad", "timed_bytes"]
SOLVER_COLS_MODEL = ["stats_elapsed_time", "timed_time"]
BLANKS = ["NaN", "None", "nan"]


def _read_logs(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        df = pd.read_json(f, lines=True)
        frames.append(df.replace(r"^\s*$", np.nan, regex=True).replace(BLANKS, np.nan))
    if not frames:
        raise FileNotFoundError("no log files found (looked for output.json_*.log)")
    return pd.concat(frames, ignore_index=True)


def collect_problem_logs(log_dir: Path) -> pd.DataFrame:
    df = _read_logs(sorted(Path(log_dir).rglob("output.json_*.log")))
    for col in PROBLEM_COLS_REG:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=PROBLEM_COLS_REG + JOIN_KEY)


def collect_solver_logs(log_dir: Path) -> pd.DataFrame:
    df = _read_logs(sorted(Path(log_dir).rglob("output.json_*.log")))
    return df.dropna(subset=SOLVER_COLS_REG + SOLVER_COLS_MODEL + JOIN_KEY)


def merge_tables(solver: pd.DataFrame, problem: pd.DataFrame, time_limit: float = 3600.0) -> pd.DataFrame:
    """Join solver runs with problem features and assign times to failed runs."""
    df = pd.merge(solver, problem, on=JOIN_KEY, how="left", suffixes=("_solver", "_problem"))
    df[EXPR_COLS] = df[EXPR_COLS].eq(True)
    df[ERROR_COLS] = df[ERROR_COLS].replace(r"^\s*$", np.nan, regex=True).replace(BLANKS, np.nan)
    df["timed_memory"] = df["timed_bytes"] / 1e6
    df.loc[df["error_solver"].notna(), "stats_elapsed_time"] = time_limit * 2
    df.loc[df["error_solver"] == f"Timeout after {float(time_limit)} s", "stats_elapsed_time"] = time_limit
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="directory for input/output tables (default: data/ or MEMTUNE_DATA_DIR)")
    ap.add_argument("--problem-logs", help="directory with problem log files (output.json_*.log), searched recursively")
    ap.add_argument("--solver-logs", help="directory with solver log files (output.json_*.log), searched recursively")
    ap.add_argument("--time-limit", type=float, default=3600.0, help="per-run time limit in seconds (default 3600)")
    a = ap.parse_args(argv)

    data = paths.data_dir(a.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    problem_csv, solver_csv = data / "problem_metadata.csv", data / "solver_runtimes.csv"

    if a.problem_logs:
        collect_problem_logs(Path(a.problem_logs)).to_csv(problem_csv, index=False)
    if a.solver_logs:
        collect_solver_logs(Path(a.solver_logs)).to_csv(solver_csv, index=False)
    for f in (problem_csv, solver_csv):
        if not f.is_file():
            ap.error(f"{f} not found: pass --problem-logs / --solver-logs, or place the file there")

    df = merge_tables(pd.read_csv(solver_csv), pd.read_csv(problem_csv), a.time_limit)
    out = data / paths.DATASET_NAME
    df.to_csv(out, index=False)
    ok = df[df[ERROR_COLS].isna().all(axis=1)]
    ok.to_csv(data / "lbfgs_runs_success.csv", index=False)
    print(f"{len(df)} runs -> {out}\n{len(ok)} error-free runs -> {data / 'lbfgs_runs_success.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
