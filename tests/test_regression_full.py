"""Regression test (full data): reproduces the reference run AND the numbers reported in the papers.

Skipped automatically when the full dataset is not available.
"""
import pytest
from conftest import assert_matches_golden

pytestmark = pytest.mark.full


@pytest.fixture(scope="module")
def full_result(full_csv):
    from memtune import evaluate_framework

    return evaluate_framework(full_csv)


def test_full_matches_reference(full_result, golden_full):
    assert_matches_golden(full_result, golden_full)


def test_dataset_size(full_result):
    assert full_result["n_instances"] == 215
    assert full_result["n_test_instances"] == 43
    assert full_result["n_rows"] == 21500


@pytest.mark.parametrize(
    "policy, q90, mean",
    [
        ("baseline1_mem5", 2.6839, 2.2616),    # default mem = 5
        ("baseline2_mem87", 1.2799, 1.0933),   # best fixed mem chosen with hindsight
        ("framework_rf", 2.1954, 1.6367),      # framework, random-forest Stage 2
        ("framework_gb", 2.0613, 1.4591),      # framework, gradient-boosting Stage 2
    ],
)
def test_paper_table4(full_result, policy, q90, mean):
    row = full_result["table4"][policy]
    assert row["Q90"] == pytest.approx(q90, abs=5e-5)
    assert row["mean"] == pytest.approx(mean, abs=5e-5)
