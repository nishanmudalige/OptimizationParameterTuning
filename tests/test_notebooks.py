"""Smoke test: the modelling notebooks run top to bottom against the packaged paths.

Needs the full dataset and ``pip install nbclient ipykernel``; otherwise skipped.
Notebook 01 also reads extra intermediate tables, so it is not included here.
"""
import os
import shutil

import pytest

from conftest import ROOT

pytestmark = pytest.mark.full
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

NOTEBOOKS = ["02_random_forest", "03_gradient_boosting", "04_linear_regression",
             "05_hybrid_framework", "06_end_to_end_pipeline"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_runs(name, full_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMTUNE_DATA_DIR", str(full_csv.parent))
    monkeypatch.setenv("MEMTUNE_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    nbclient.NotebookClient(nb, timeout=1200, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()
    # published models are never modified by a notebook
    sums = (ROOT / "models" / "SHA256SUMS").read_text().splitlines()
    import hashlib
    for line in sums:
        digest, rel = line.split(maxsplit=1)
        assert hashlib.sha256((ROOT / "models" / rel.strip()).read_bytes()).hexdigest() == digest
