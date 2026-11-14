"""Every script in scripts/ starts, and the long-running ones run in their quick modes."""
import subprocess
import sys

import pytest

from conftest import ROOT

SCRIPTS = ["build_dataset", "export_demo_data", "plot_global_mem", "plot_per_problem", "robustness_check",
           "train_model_comparison", "train_one_step_model"]


def run(script, args, cwd, timeout=600):
    env_extra = {"MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    import os

    return subprocess.run([sys.executable, str(ROOT / "scripts" / f"{script}.py"), *args], cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env={**os.environ, **env_extra})


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_works_from_any_directory(script, tmp_path):
    r = run(script, ["--help"], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()


@pytest.mark.full
def test_plot_global_mem(tmp_path, full_csv):
    r = run("plot_global_mem", ["--data", str(full_csv), "--out", str(tmp_path)], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "global_mem_plots").glob("*.pdf"))) == 6


@pytest.mark.full
def test_plot_per_problem_limited(tmp_path, full_csv):
    r = run("plot_per_problem", ["--data", str(full_csv), "--out", str(tmp_path), "--limit", "2"], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "plots").rglob("*.pdf"))) >= 20


@pytest.mark.full
def test_train_model_comparison_quick(tmp_path, full_csv):
    r = run("train_model_comparison", ["--data", str(full_csv), "--out", str(tmp_path), "--quick",
                                       "--families", "Ridge,Lasso"], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "model_comparison"
    assert (out / "leaderboard.csv").is_file() and (out / "summary.json").is_file()
    assert list(out.glob("best_model_*.pkl"))


def test_unknown_family_is_reported(tmp_path, full_csv_or_sample):
    r = run("train_model_comparison", ["--data", str(full_csv_or_sample), "--out", str(tmp_path), "--families", "Nope"], cwd=tmp_path)
    assert r.returncode != 0
    assert "unknown model families" in (r.stderr + r.stdout)
