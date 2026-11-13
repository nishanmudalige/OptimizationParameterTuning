"""Shared fixtures. Fast tests use the small committed sample; ``@pytest.mark.full`` tests need the real data."""
import json
import os
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
SAMPLE_CSV = TESTS / "data" / "sample_runs.csv"


def _full_dataset():
    from memtune.paths import dataset_path

    path = dataset_path()
    return path if path.is_file() else None


@pytest.fixture(scope="session")
def sample_csv():
    return SAMPLE_CSV


@pytest.fixture(scope="session")
def full_csv():
    path = _full_dataset()
    if path is None:
        pytest.skip("full dataset not available (set MEMTUNE_DATA_DIR or put lbfgs_runs.csv[.gz] in data/)")
    return path


@pytest.fixture(scope="session")
def golden_sample():
    return json.loads((TESTS / "golden" / "golden_sample.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def golden_full():
    return json.loads((TESTS / "golden" / "golden_full.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sample_result(sample_csv):
    from memtune import evaluate_framework

    return evaluate_framework(sample_csv)


def assert_matches_golden(result, golden, rtol=1e-9):
    """Exact for counts / chosen mem, tight relative tolerance for floating-point metrics."""
    for key in ("n_rows", "n_instances", "n_test_instances", "n_test_rows"):
        assert result[key] == golden[key], key
    for stage in ("stage1", "stage2"):
        for model, metrics in golden[stage].items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                assert result[stage][model][metric] == pytest.approx(value, rel=rtol), f"{stage}.{model}.{metric}"
    for policy, row in golden["table4"].items():
        for metric, value in row.items():
            assert result["table4"][policy][metric] == pytest.approx(value, rel=rtol), f"table4.{policy}.{metric}"
    for policy, chosen in golden["chosen_mem"].items():
        assert result["chosen_mem"][policy] == chosen, f"chosen mem differs for {policy}"
