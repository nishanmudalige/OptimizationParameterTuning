"""The one-step model (NOT in the paper): reproducible, free of the framework's small-mem bias, and its
test-set numbers stay as documented. The paper's own numbers are checked in test_regression_full.py."""
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT


def test_recommend_one_step_on_sample(sample_csv):
    from memtune import load_framework, recommend_mem
    from memtune.data import load_dataset

    _, _, test = load_dataset(sample_csv)
    first = test[test["mem"] == 1].reset_index(drop=True)
    rec = recommend_mem(first, load_framework(), stage2="one_step")
    assert len(rec) == len(first) and rec.min() >= 1 and rec.max() <= 100


def test_paper_evaluation_can_leave_it_out(sample_csv):
    from memtune import evaluate_framework

    assert "one_step_rf" in evaluate_framework(sample_csv)["table4"]
    res = evaluate_framework(sample_csv, one_step=False)
    assert "one_step_rf" not in res["table4"] and "one_step_rf" not in res["chosen_mem"]


@pytest.fixture(scope="module")
def full(full_csv):
    from memtune import evaluate_framework, load_framework
    from memtune.data import load_dataset

    _, train, test = load_dataset(full_csv)
    return {"train": train, "test": test, "fw": load_framework(), "res": evaluate_framework(full_csv)}


@pytest.mark.full
def test_documented_test_set_numbers(full):
    row = full["res"]["table4"]["one_step_rf"]
    assert row["Q90"] == pytest.approx(1.3942, abs=5e-4)
    assert row["mean"] == pytest.approx(1.1404, abs=5e-4)
    assert row["total_time"] == pytest.approx(2583.6, abs=0.5)
    # better than the paper's framework (both Stage 2 models) and the default mem = 5 on every summary
    for other in ("framework_rf", "framework_gb", "baseline1_mem5"):
        for metric in ("Q90", "mean", "total_time"):
            assert row[metric] < full["res"]["table4"][other][metric], (other, metric)


@pytest.mark.full
def test_bias_of_the_framework_and_not_of_the_one_step_model(full):
    """Within an instance, measured time falls with mem; the framework's prediction rises, the one-step one falls."""
    from memtune.data import FEATURES
    from memtune.one_step import within_instance_trend

    test, fw = full["test"], full["fw"]
    X = test[FEATURES].to_numpy(dtype=float)
    assert within_instance_trend(test, np.log1p(test["stats_elapsed_time"])) < 0
    assert within_instance_trend(test, fw.log_time(X, "gb")) > 0.3
    assert within_instance_trend(test, fw.log_time(X, "rf")) > 0
    assert within_instance_trend(test, fw.log_time(X, "one_step")) < 0
    chosen = full["res"]["chosen_mem"]
    small = {k: sum(m <= 3 for m in chosen[k].values()) for k in ("framework_gb", "framework_rf", "one_step_rf")}
    assert small["framework_gb"] >= 20 and small["framework_rf"] >= 15 and small["one_step_rf"] <= 2


@pytest.mark.full
def test_retraining_reproduces_the_committed_model(full, tmp_path, full_csv):
    """The training script rebuilds a model that makes the same choices as the committed file."""
    import joblib

    from memtune.data import FEATURES
    from memtune.evaluate import performance_ratios
    from memtune.models import MODEL_FILES

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_one_step_model.py"), "--data", str(full_csv),
                        "--out", str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    model = joblib.load(tmp_path / MODEL_FILES["one_step_rf"])
    test = full["test"]
    m = performance_ratios(test, model.predict(test[FEATURES].to_numpy(dtype=float)))
    got = {f"{n}|{v}": int(k) for n, v, k in m[["name", "nvar", "mem"]].itertuples(index=False)}
    assert got == full["res"]["chosen_mem"]["one_step_rf"]
