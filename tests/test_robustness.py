"""The robustness evaluation (NOT in the paper): other splits, and whole problems held out."""
import numpy as np
import pandas as pd
import pytest


def runs(times_by_instance):
    """A small table of runs: ``{(name, nvar): [time at mem 1, time at mem 2, ...]}``."""
    rows = [{"name": name, "nvar": nvar, "mem": mem, "stats_elapsed_time": t}
            for (name, nvar), times in times_by_instance.items() for mem, t in enumerate(times, start=1)]
    return pd.DataFrame(rows)


def test_best_fixed_mem_minimises_the_mean_ratio():
    from memtune.robustness import best_fixed_mem

    # mem 2 is best for a and close to best for b; mem 3 is best for b but terrible for a
    train = runs({("a", 10): [9.0, 1.0, 30.0], ("b", 10): [9.0, 2.5, 2.0]})
    assert best_fixed_mem(train) == 2


def test_best_fixed_mem_breaks_ties_towards_the_largest_mem():
    from memtune.robustness import best_fixed_mem

    # differences below the 0.1 s tolerance count as ties, so mem 2 and mem 3 are equally good
    train = runs({("a", 10): [5.0, 1.00, 1.05], ("b", 10): [5.0, 2.03, 2.00]})
    assert best_fixed_mem(train) == 3


@pytest.fixture(scope="module")
def sample(sample_csv):
    from memtune import load_framework
    from memtune.data import load_dataset

    df, _, _ = load_dataset(sample_csv)
    return df, load_framework()


@pytest.fixture(scope="module")
def lopo(sample):
    from memtune.robustness import leave_problems_out

    df, base = sample
    return leave_problems_out(df, base, folds=3, seed=0, n_jobs=1)


def test_leave_problems_out_tests_every_instance_once_on_unseen_problems(sample, lopo):
    from memtune.robustness import POLICIES

    df, _ = sample
    instances = df[["name", "nvar"]].drop_duplicates()
    one = lopo[lopo["policy"] == "one_step_rf"]
    assert len(one) == len(instances) and not one.duplicated(["name", "nvar"]).any()
    assert set(lopo["policy"]) == set(POLICIES)
    # a problem (all its sizes) is in exactly one fold
    assert (lopo.groupby("name")["fold"].nunique() == 1).all()
    assert (lopo["ratio"] >= 1 - 1e-12).all() and (lopo["time"] >= lopo["best_time"] - 1e-12).all()
    assert lopo["mem"].between(1, 100).all()


def test_fixed_policies_choose_what_they_say(lopo):
    assert (lopo.loc[lopo["policy"] == "default_mem5", "mem"] == 5).all()
    fixed = lopo[lopo["policy"] == "fixed_train"]
    assert (fixed["mem"] == fixed["fixed_train_mem"]).all()
    assert fixed.groupby("fold")["mem"].nunique().eq(1).all()


def test_summaries(lopo):
    from memtune.robustness import SUMMARY_COLUMNS, across_splits, paired_comparison, per_split_table, pooled_table

    pooled = pooled_table(lopo, n_boot=200)
    assert list(pooled.index) == ["default_mem5", "fixed_train", "framework_rf", "framework_gb", "one_step_rf"]
    assert (pooled["mean_lo"] <= pooled["mean"]).all() and (pooled["mean"] <= pooled["mean_hi"]).all()
    assert (pooled["Q50"] <= pooled["Q90"]).all() and pooled["small_mem"].between(0, 1).all()
    assert pooled.loc["default_mem5", "small_mem"] == 0

    per_fold = per_split_table(lopo, "fold")
    assert len(per_fold) == 3 * 5
    across = across_splits(per_fold)
    assert [c for _, c in across.columns][:len(SUMMARY_COLUMNS)] == SUMMARY_COLUMNS

    cmp = paired_comparison(lopo, "one_step_rf")
    assert cmp["faster"] + cmp["slower"] + cmp["tie"] == (lopo["policy"] == "one_step_rf").sum()
    assert paired_comparison(lopo, "default_mem5") == {"faster": 0, "slower": 0, "tie": cmp["faster"] + cmp["slower"] + cmp["tie"]}


def test_repeated_splits_follow_the_papers_protocol(sample):
    from memtune.data import split_instances
    from memtune.robustness import repeated_splits

    df, base = sample
    rows = repeated_splits(df, base, seeds=[3], n_jobs=1)
    _, test = split_instances(df, seed=3)
    expected = set(map(tuple, test[["name", "nvar"]].drop_duplicates().to_numpy()))
    got = set(map(tuple, rows.loc[rows["policy"] == "framework_rf", ["name", "nvar"]].to_numpy()))
    assert got == expected and (rows["seed"] == 3).all()


def test_refit_is_reproducible(sample):
    from memtune.data import FEATURES, split_instances
    from memtune.robustness import refit_framework

    df, base = sample
    train, test = split_instances(df, seed=1)
    X = test[FEATURES].to_numpy(dtype=float)
    a, b = refit_framework(train, base, n_jobs=1), refit_framework(train, base, n_jobs=1)
    for which in ("rf", "gb", "one_step"):
        # the same trees every time; a forest's average can differ in the last digit between runs
        assert np.allclose(a.log_time(X, which), b.log_time(X, which), rtol=1e-9, atol=1e-12), which
