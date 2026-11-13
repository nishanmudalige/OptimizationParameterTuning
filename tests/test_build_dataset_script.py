"""Run scripts/build_dataset.py end to end on the tiny example logs in tests/data/example_logs."""
import subprocess
import sys

import pandas as pd

from conftest import ROOT

LOGS = ROOT / "tests" / "data" / "example_logs"


def _run(args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dataset.py"), *args],
                          cwd=cwd, capture_output=True, text=True, timeout=120)


def test_build_from_example_logs(tmp_path):
    r = _run(["--data-dir", str(tmp_path), "--problem-logs", str(LOGS / "problem_logs"),
              "--solver-logs", str(LOGS / "solver_logs")], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(tmp_path / "lbfgs_runs.csv").sort_values(["name", "mem"]).reset_index(drop=True)
    assert len(df) == 6
    assert (tmp_path / "problem_metadata.csv").is_file() and (tmp_path / "solver_runtimes.csv").is_file()
    # problem features were joined onto every run
    assert (df.loc[df["name"] == "ARGLINA", "tree_length"] == 12).all()
    # penalty rules: time-out -> the limit, other error -> twice the limit, normal runs untouched
    arglina = df[df["name"] == "ARGLINA"].set_index("mem")["stats_elapsed_time"]
    assert arglina[3] == 3600.0
    assert df[(df["name"] == "WOODS") & (df["mem"] == 2)]["stats_elapsed_time"].iloc[0] == 7200.0
    assert arglina[1] == 0.05
    # runs without error go to the "success" table
    ok = pd.read_csv(tmp_path / "lbfgs_runs_success.csv")
    assert len(ok) == 4


def test_gzip_option_writes_a_file_memtune_can_read(tmp_path):
    r = _run(["--data-dir", str(tmp_path), "--gzip", "--problem-logs", str(LOGS / "problem_logs"),
              "--solver-logs", str(LOGS / "solver_logs")], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "lbfgs_runs.csv.gz").is_file() and not (tmp_path / "lbfgs_runs.csv").exists()
    from memtune.data import load_raw

    raw = load_raw(tmp_path)  # finds the .gz and has every column the pipeline needs
    assert len(raw) == 6


def test_missing_inputs_give_a_clear_message(tmp_path):
    r = _run(["--data-dir", str(tmp_path)], cwd=tmp_path)
    assert r.returncode == 2
    assert "not found" in r.stderr and "--problem-logs" in r.stderr
