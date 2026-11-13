"""The published models are intact, load with the pinned libraries and keep their input shapes."""
import hashlib
from pathlib import Path

import pytest

from memtune.models import MODEL_FILES, check_versions, load_model
from memtune.paths import models_dir

SUMS = {}
for line in (Path(__file__).resolve().parents[1] / "models" / "SHA256SUMS").read_text().splitlines():
    digest, rel = line.split(maxsplit=1)
    SUMS[rel.strip()] = digest


def test_versions_are_the_pinned_ones():
    assert check_versions(strict=False) == []


@pytest.mark.parametrize("key, rel", sorted(MODEL_FILES.items()))
def test_checksum_unchanged(key, rel):
    assert hashlib.sha256((models_dir() / rel).read_bytes()).hexdigest() == SUMS[rel]


@pytest.mark.parametrize(
    "key, n_features",
    [("stage1_rf", 15), ("stage1_xgb", 15), ("stage1_lr", 15),
     ("stage2_hybrid_rf", 3), ("stage2_hybrid_xgb", 3),
     ("stage2_truth_rf", 3), ("stage2_truth_xgb", 3), ("stage2_truth_lr", 3)],
)
def test_model_loads_with_expected_input_width(key, n_features):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = load_model(key)
    assert model.n_features_in_ == n_features
