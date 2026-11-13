"""Data preparation behaves as in the notebooks."""
import pandas as pd
import pytest

from memtune.data import GROUP_KEY, N_MEM, load_raw, prepare, split_instances


def test_prepare_keeps_only_complete_instances(sample_csv):
    raw = load_raw(sample_csv)
    # remove one mem value from the first instance -> it must disappear
    first = raw[GROUP_KEY].drop_duplicates().iloc[0]
    drop = (raw["name"] == first["name"]) & (raw["nvar"] == first["nvar"]) & (raw["mem"] == 50)
    df = prepare(raw[~drop])
    assert (df.groupby(GROUP_KEY).size() == N_MEM).all()
    assert len(df[GROUP_KEY].drop_duplicates()) == len(raw[GROUP_KEY].drop_duplicates()) - 1


def test_split_keeps_instances_together_and_is_reproducible(sample_csv):
    df = prepare(load_raw(sample_csv))
    tr1, te1 = split_instances(df)
    tr2, te2 = split_instances(df)
    pd.testing.assert_frame_equal(te1, te2)
    assert not set(map(tuple, tr1[GROUP_KEY].drop_duplicates().to_numpy())) & set(
        map(tuple, te1[GROUP_KEY].drop_duplicates().to_numpy()))
    assert len(tr1) + len(te1) == len(df)
    assert len(te1[GROUP_KEY].drop_duplicates()) == len(df[GROUP_KEY].drop_duplicates()) - int(
        0.8 * len(df[GROUP_KEY].drop_duplicates()))


def test_missing_columns_are_reported(tmp_path, sample_csv):
    bad = tmp_path / "bad.csv"
    pd.read_csv(sample_csv).drop(columns=["tree_depth"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="tree_depth"):
        load_raw(bad)
