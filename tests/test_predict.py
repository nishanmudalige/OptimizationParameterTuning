"""The public ``recommend_mem`` API agrees with the evaluation pipeline."""
import numpy as np
import pytest

from memtune import load_framework, recommend_mem
from memtune.data import load_dataset
from memtune.predict import PROBLEM_FEATURES


@pytest.fixture(scope="module")
def test_problems(sample_csv):
    _, _, test = load_dataset(sample_csv)
    first = test[test["mem"] == 1].reset_index(drop=True)
    return first


def test_problem_features_are_constant_within_an_instance(sample_csv):
    _, _, test = load_dataset(sample_csv)
    varying = test.groupby(["name", "nvar"])[PROBLEM_FEATURES].nunique(dropna=False).max()
    assert (varying <= 1).all(), varying[varying > 1]


def test_recommendation_equals_evaluated_choice(test_problems, golden_sample):
    rec = recommend_mem(test_problems, stage2="rf")
    expected = golden_sample["chosen_mem"]["framework_rf"]
    got = {f"{n}|{v}": int(m) for n, v, m in zip(test_problems["name"], test_problems["nvar"], rec)}
    assert got == expected


def test_recommendation_is_in_range_and_deterministic(test_problems):
    fw = load_framework()
    a = recommend_mem(test_problems, framework=fw, stage2="gb")
    b = recommend_mem(test_problems, framework=fw, stage2="gb")
    assert np.array_equal(a, b)
    assert a.min() >= 1 and a.max() <= 100


def test_missing_feature_columns_are_reported(test_problems):
    with pytest.raises(ValueError, match="missing feature columns"):
        recommend_mem(test_problems.drop(columns=["tree_depth"]))
