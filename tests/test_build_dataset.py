"""Unit tests for the table-merging step of ``scripts/build_dataset.py`` (synthetic data)."""
import importlib.util
import sys

import pandas as pd

from conftest import ROOT

spec = importlib.util.spec_from_file_location("build_dataset", ROOT / "scripts" / "build_dataset.py")
build_dataset = importlib.util.module_from_spec(spec)
sys.modules["build_dataset"] = build_dataset
spec.loader.exec_module(build_dataset)


def _tables(errors):
    solver = pd.DataFrame({
        "problem": ["p"] * 4, "name": ["P"] * 4, "nvar": [10] * 4, "mem": [1, 2, 3, 4],
        "neval_obj": [5, 5, 5, 5], "neval_grad": [5, 5, 5, 5], "timed_bytes": [2e6] * 4,
        "stats_elapsed_time": [1.0, 2.0, 3600.0, 3.0], "timed_time": [1.0, 2.0, 3600.0, 3.0],
        "error_solver": errors,
    })
    problem = pd.DataFrame({
        "problem": ["p"], "name": ["P"], "nvar": [10], "tree_length": [7], "tree_depth": [3],
        "time_obj": [1e-6], "time_grad": [2e-6],
        "is_constant (ExprTree)": [False], "is_linear (ExprTree)": [True],
        "error_problem": [""], "error_type": [""],
    })
    return solver, problem


def test_failed_runs_get_penalty_times():
    errors = ["", "Boom", "Timeout after 3600.0 s", None]
    df = build_dataset.merge_tables(*_tables(errors), time_limit=3600.0)
    t = df.sort_values("mem")["stats_elapsed_time"].tolist()
    assert t == [1.0, 7200.0, 3600.0, 3.0]  # ok, error -> 2x limit, time-out -> limit, ok


def test_time_limit_is_configurable():
    errors = ["", "Boom", "Timeout after 100.0 s", None]
    df = build_dataset.merge_tables(*_tables(errors), time_limit=100.0)
    assert df.sort_values("mem")["stats_elapsed_time"].tolist() == [1.0, 200.0, 100.0, 3.0]


def test_join_keeps_all_solver_rows_and_adds_problem_features():
    df = build_dataset.merge_tables(*_tables([None] * 4))
    assert len(df) == 4
    assert (df["tree_length"] == 7).all()
    assert df["timed_memory"].tolist() == [2.0] * 4
