"""Nothing depends on the working directory or on a fixed location.

Each test runs the program from an unrelated directory and/or redirects data and output with
environment variables or flags.
"""
import json
import os
import shutil
import subprocess
import sys

from conftest import ROOT, SAMPLE_CSV


def run(args, cwd, env_extra=None):
    env = {**os.environ, **(env_extra or {})}
    env.pop("MEMTUNE_ROOT", None)
    return subprocess.run([sys.executable, "-m", "memtune", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


def test_info_from_another_working_directory(tmp_path):
    r = run(["info"], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert str(ROOT) in r.stdout  # repo root found from the package location, not from cwd


def test_data_dir_environment_variable(tmp_path):
    data = tmp_path / "somewhere else" / "my data"        # spaces on purpose (Windows/macOS paths)
    data.mkdir(parents=True)
    shutil.copy(SAMPLE_CSV, data / "lbfgs_runs.csv")
    out = tmp_path / "out"
    r = run(["evaluate", "--out", str(out)], cwd=tmp_path, env_extra={"MEMTUNE_DATA_DIR": str(data)})
    assert r.returncode == 0, r.stderr
    result = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    assert result["n_instances"] == 25
    assert (out / "table4.csv").is_file()


def test_data_flag_accepts_file_or_directory(tmp_path):
    folder = tmp_path / "data folder"
    folder.mkdir()
    shutil.copy(SAMPLE_CSV, folder / "lbfgs_runs.csv")
    for target in (SAMPLE_CSV, folder):  # a CSV file, and a directory that contains lbfgs_runs.csv
        r = run(["evaluate", "--data", str(target), "--out", str(tmp_path / "o")], cwd=tmp_path)
        assert r.returncode == 0, r.stderr


def test_missing_data_gives_a_helpful_error(tmp_path):
    r = run(["evaluate", "--data", str(tmp_path / "nope.csv"), "--out", str(tmp_path)], cwd=tmp_path)
    assert r.returncode != 0
    assert "Dataset not found" in r.stderr


def test_no_hard_coded_absolute_paths_in_sources():
    """Guards against the old problems: '/home/<user>/...' and 'C:\\Users\\...' inside the code."""
    bad = ("/home/", "C:\\\\Users", "C:\\Users", "MacbookPro", "/Users/")
    offenders = []
    for folder in ("src", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            offenders += [f"{path.relative_to(ROOT)}: {b}" for b in bad if b in text]
    assert not offenders, offenders
