"""Regression test (fast): the packaged pipeline reproduces the reference results on the sample data.

The reference (``tests/golden/golden_sample.json``) was produced by ``tests/golden/generate_golden.py``,
which re-implements the ORIGINAL notebook logic independently of the ``memtune`` package.
"""
from conftest import assert_matches_golden


def test_sample_matches_reference(sample_result, golden_sample):
    assert_matches_golden(sample_result, golden_sample)


def test_split_is_the_papers_split(sample_csv, golden_sample):
    from memtune.data import load_dataset

    _, _, test = load_dataset(sample_csv)
    names = sorted(f"{n}|{v}" for n, v in test[["name", "nvar"]].drop_duplicates().itertuples(index=False))
    assert names == golden_sample["test_instances"]
