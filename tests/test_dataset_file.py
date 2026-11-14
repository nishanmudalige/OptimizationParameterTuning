"""The compressed dataset shipped in ``data/`` is intact, and the code finds it without unzipping."""
import gzip
import hashlib
import shutil

import pandas as pd

from conftest import ROOT, SAMPLE_CSV
from memtune.paths import dataset_path

GZ = ROOT / "data" / "lbfgs_runs.csv.gz"
SUMS = dict(reversed(line.split(maxsplit=1)) for line in (ROOT / "data" / "SHA256SUMS").read_text().splitlines())
SUMS = {k.strip(): v for k, v in SUMS.items()}  # name -> digest


def _sha(fileobj):
    h = hashlib.sha256()
    for chunk in iter(lambda: fileobj.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def test_compressed_file_checksum():
    with open(GZ, "rb") as f:
        assert _sha(f) == SUMS["lbfgs_runs.csv.gz"]


def test_unzipped_content_checksum():
    with gzip.open(GZ, "rb") as f:
        assert _sha(f) == SUMS["lbfgs_runs.csv"]


def test_unzipping_with_the_documented_command_restores_the_csv(tmp_path):
    import subprocess
    import sys

    shutil.copy(GZ, tmp_path / GZ.name)
    subprocess.run([sys.executable, "-m", "gzip", "-d", GZ.name], cwd=tmp_path, check=True)
    with open(tmp_path / "lbfgs_runs.csv", "rb") as f:
        assert _sha(f) == SUMS["lbfgs_runs.csv"]


def test_shape_matches_documentation():
    df = pd.read_csv(GZ, low_memory=False)
    assert df.shape == (21802, 97)
    assert df["mem"].min() == 1 and df["mem"].max() == 100


def test_gz_is_found_when_only_gz_exists(tmp_path):
    shutil.copy(GZ, tmp_path / GZ.name)
    assert dataset_path(tmp_path) == (tmp_path / "lbfgs_runs.csv.gz").resolve()
    assert dataset_path(tmp_path / "lbfgs_runs.csv") == (tmp_path / "lbfgs_runs.csv.gz").resolve()


def test_plain_csv_wins_over_gz(tmp_path):
    shutil.copy(GZ, tmp_path / GZ.name)
    shutil.copy(SAMPLE_CSV, tmp_path / "lbfgs_runs.csv")
    assert dataset_path(tmp_path) == (tmp_path / "lbfgs_runs.csv").resolve()


def test_default_location_is_the_shipped_file(monkeypatch):
    monkeypatch.delenv("MEMTUNE_DATA_DIR", raising=False)
    assert dataset_path() in (GZ, ROOT / "data" / "lbfgs_runs.csv")
