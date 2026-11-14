"""How much do the end-to-end results depend on the train/test split? (NOT in the paper.)

The paper reports one split of the 215 complete instances by instance (80 % / 20 %, seed 66). Two things are worth
knowing about that split:

* 43 test instances are few, so the summary statistics move when the split changes;
* an *instance* is a problem at one size, and every test instance is a problem that also occurs in the training
  data at another size. The reported numbers therefore describe a new size of a known problem, not a new problem.

This module refits everything on other splits and evaluates it the same way as the paper:

``repeated_splits``
    the paper's protocol (split by instance) with other seeds;
``leave_problems_out``
    grouped cross-validation by problem name: all sizes of a problem are held out together, so every instance is
    tested by models that never saw its problem.

The models are refitted with the hyper-parameters of the committed ones and the recipe of the notebooks: Stage 1
(random forest and XGBoost) on the training rows, Stage 2 on the 5-fold out-of-fold Stage 1 predictions. No
hyper-parameter is tuned again. Policies compared on every test set:

==================  ====================================================================================
``default_mem5``    the default of JSOSolvers.jl
``fixed_train``     one fixed ``mem`` for all problems, chosen on the *training* instances of that split
                    (smallest mean performance ratio); unlike the paper's ``mem = 87`` it uses no test data
``framework_rf``    the paper's framework: hybrid Stage 1, random-forest Stage 2
``framework_gb``    the alternative XGBoost Stage 2 on the same Stage 1 (not part of the framework)
``one_step_rf``     the one-step model (see :mod:`memtune.one_step`)
==================  ====================================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import one_step
from .data import FEATURES, GROUP_KEY, TIME_COLUMN, log_targets, log_time, split_instances
from .evaluate import TOLERANCE_SECONDS, fixed_mem_score, performance_ratios
from .models import Framework

POLICIES = ("default_mem5", "fixed_train", "framework_rf", "framework_gb", "one_step_rf")
POLICY_LABELS = {
    "default_mem5": "Default mem = 5",
    "fixed_train": "Fixed mem chosen on the training instances",
    "framework_rf": "Framework (paper)",
    "framework_gb": "Alternative, XGBoost Stage 2",
    "one_step_rf": "One-step model",
}
SUMMARY_COLUMNS = ["Q50", "Q75", "Q90", "mean", "time_ratio", "small_mem"]


def _fresh(model, n_jobs):
    new = clone(model)
    if "n_jobs" in new.get_params() and type(new).__name__.startswith("RandomForest"):
        new.set_params(n_jobs=n_jobs)  # same trees for any number of jobs; averages can differ in the last digit
    return new


def refit_framework(train: pd.DataFrame, base: Framework, n_jobs: int = -1) -> Framework:
    """Refit Stage 1, Stage 2 and the one-step model on ``train`` with the settings of ``base``."""
    X = train[FEATURES].to_numpy(dtype=float)
    y1, y2 = log_targets(train), log_time(train)
    s1_rf, s1_xgb = _fresh(base.stage1_rf, n_jobs), _fresh(base.stage1_xgb, n_jobs)
    # Stage 2 learns from out-of-fold Stage 1 predictions, as in notebooks/05_hybrid_framework.ipynb
    oof_rf = cross_val_predict(_fresh(base.stage1_rf, n_jobs), X, y1, cv=5)
    oof_xgb = cross_val_predict(_fresh(base.stage1_xgb, n_jobs), X, y1, cv=5)
    X2 = np.column_stack([oof_rf[:, 0], oof_rf[:, 1], oof_xgb[:, 2]])
    s1_rf.fit(X, y1)
    s1_xgb.fit(X, y1)
    return Framework(
        stage1_rf=s1_rf, stage1_xgb=s1_xgb,
        stage2_rf=_fresh(base.stage2_rf, n_jobs).fit(X2, y2),
        stage2_xgb=_fresh(base.stage2_xgb, n_jobs).fit(X2, y2),
        one_step=one_step.make_model(n_jobs=n_jobs).fit(X, y2),
    )


def best_fixed_mem(train: pd.DataFrame, tol: float = TOLERANCE_SECONDS) -> int:
    """The single ``mem`` with the smallest mean performance ratio over the instances in ``train``."""
    t = train.pivot_table(index=GROUP_KEY, columns="mem", values=TIME_COLUMN).to_numpy(dtype=float)
    best = t.min(axis=1, keepdims=True)
    ratio = np.where(np.abs(t - best) <= tol, 1.0, t / best).mean(axis=0)
    mems = np.sort(train["mem"].unique())
    return int(mems[np.flatnonzero(ratio == ratio.min())[-1]])  # ties -> largest mem, as everywhere else


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame, base: Framework, n_jobs: int = -1) -> pd.DataFrame:
    """One row per (test instance, policy): chosen ``mem``, its measured time, the best time and the ratio."""
    fw = refit_framework(train, base, n_jobs)
    X = test[FEATURES].to_numpy(dtype=float)
    fixed = best_fixed_mem(train)
    scores = {
        "default_mem5": fixed_mem_score(test, 5),
        "fixed_train": fixed_mem_score(test, fixed),
        "framework_rf": fw.log_time(X, "rf"),
        "framework_gb": fw.log_time(X, "gb"),
        "one_step_rf": fw.log_time(X, "one_step"),
    }
    parts = []
    for policy, score in scores.items():
        m = performance_ratios(test, score)
        parts.append(pd.DataFrame({
            "name": m["name"], "nvar": m["nvar"], "policy": policy, "mem": m["mem"],
            "time": m[TIME_COLUMN], "best_time": m[TIME_COLUMN + "_best"], "ratio": m["ratio"],
        }))
    out = pd.concat(parts, ignore_index=True)
    out["fixed_train_mem"] = fixed
    return out


def summarize(rows: pd.DataFrame) -> dict:
    """Summary of one policy on one test set (``rows``: one row per instance)."""
    r = rows["ratio"].to_numpy(dtype=float)
    return {
        "Q50": float(np.quantile(r, 0.50)), "Q75": float(np.quantile(r, 0.75)), "Q90": float(np.quantile(r, 0.90)),
        "mean": float(r.mean()),
        "time_ratio": float(rows["time"].sum() / rows["best_time"].sum()),  # total chosen time / total best time
        "small_mem": float((rows["mem"] <= 3).mean()),                      # share of instances with mem <= 3
    }


def repeated_splits(df: pd.DataFrame, base: Framework, seeds, n_jobs: int = -1, progress=None) -> pd.DataFrame:
    """The paper's protocol (80 % / 20 % by instance) for every seed in ``seeds``. Long format, column ``seed``."""
    parts = []
    for seed in seeds:
        train, test = split_instances(df, seed=seed)
        part = evaluate_split(train, test, base, n_jobs)
        part["seed"] = seed
        parts.append(part)
        if progress:
            progress(f"split with seed {seed}: done")
    return pd.concat(parts, ignore_index=True)


def leave_problems_out(df: pd.DataFrame, base: Framework, folds: int = 5, seed: int = 0, n_jobs: int = -1,
                       progress=None) -> pd.DataFrame:
    """Grouped ``folds``-fold cross-validation by problem name. Long format, column ``fold``."""
    names = np.sort(df["name"].unique())
    order = np.random.RandomState(seed).permutation(len(names))
    group_of = {name: i for i, name in zip(order, names)}  # shuffled group labels, the same on every system
    groups = df["name"].map(group_of).to_numpy()
    parts = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(df, groups=groups)):
        train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        assert not set(train["name"]) & set(test["name"])
        part = evaluate_split(train, test, base, n_jobs)
        part["fold"] = fold
        parts.append(part)
        if progress:
            progress(f"fold {fold + 1} of {folds}: {test['name'].nunique()} problems held out")
    return pd.concat(parts, ignore_index=True)


def per_split_table(rows: pd.DataFrame, by: str) -> pd.DataFrame:
    """One summary row per (split, policy)."""
    records = [{by: key, "policy": policy, **summarize(g)}
               for (key, policy), g in rows.groupby([by, "policy"], sort=True)]
    return pd.DataFrame(records)


def across_splits(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over the splits, per policy (rows in the order of :data:`POLICIES`)."""
    g = table.groupby("policy")[SUMMARY_COLUMNS]
    out = pd.concat({"mean": g.mean(), "sd": g.std(ddof=1)}, axis=1)
    return out.reindex([p for p in POLICIES if p in out.index])


def pooled_table(rows: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """All test instances together, per policy, with a 95 % bootstrap interval (over instances) for the mean."""
    rng = np.random.RandomState(seed)
    records = []
    for policy in [p for p in POLICIES if p in set(rows["policy"])]:
        g = rows[rows["policy"] == policy]
        r = g["ratio"].to_numpy(dtype=float)
        boot = r[rng.randint(0, len(r), size=(n_boot, len(r)))].mean(axis=1)
        records.append({"policy": policy, "n": len(r), **summarize(g),
                        "mean_lo": float(np.quantile(boot, 0.025)), "mean_hi": float(np.quantile(boot, 0.975))})
    return pd.DataFrame(records).set_index("policy")


def paired_comparison(rows: pd.DataFrame, policy: str, reference: str = "default_mem5",
                      tol: float = TOLERANCE_SECONDS) -> dict:
    """On how many test instances is ``policy`` faster / slower than ``reference`` (differences above ``tol``)?"""
    key = [c for c in ("seed", "fold") if c in rows.columns] + GROUP_KEY
    a = rows[rows["policy"] == policy].set_index(key)["time"]
    b = rows[rows["policy"] == reference].set_index(key)["time"]
    d = (a - b.reindex(a.index)).to_numpy()
    return {"faster": int((d < -tol).sum()), "slower": int((d > tol).sum()), "tie": int((np.abs(d) <= tol).sum())}
