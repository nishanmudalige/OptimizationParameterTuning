"""A one-step run-time model: an alternative to the paper's two-stage framework (NOT in the paper).

Why it exists
-------------
The paper's Stage 2 sees only three numbers, the predicted ``neval_obj``, ``neval_grad`` and
``timed_bytes``. Allocated bytes grow with ``mem`` *and*, across problems, with problem size, so Stage 2
learned "more bytes -> slower". Inside one problem this pushes the predicted run time up as ``mem`` grows
(median within-instance correlation +0.56 for the XGBoost Stage 2), whereas the measured run time goes down
(-0.25). The framework therefore favours very small ``mem`` (``mem`` <= 3 for 176 of the 280 instances).

The one-step model predicts ``log1p(run time)`` directly from the 15 features (``mem`` included), so it
can see how ``mem`` interacts with the problem. Its settings were chosen by 5-fold cross-validation over
the *training* instances only (grouped by instance, scored by the mean performance ratio); the test
instances were not used. See ``scripts/train_one_step_model.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES, GROUP_KEY, log_time

#: Settings picked by ``scripts/train_one_step_model.py --select`` (training instances only).
PARAMS = {"n_estimators": 200, "min_samples_leaf": 20, "max_features": 0.5, "random_state": 0}

#: Candidate settings compared by the cross-validation.
GRID = [{"min_samples_leaf": leaf, "max_features": mf} for leaf in (1, 5, 20) for mf in (1.0, 0.5)]


def make_model(**overrides):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**{**PARAMS, **overrides, "n_jobs": -1})


def fit(train: pd.DataFrame, **overrides):
    """Fit the one-step model on a prepared training frame (all 100 ``mem`` rows per instance)."""
    return make_model(**overrides).fit(train[FEATURES].to_numpy(dtype=float), log_time(train))


def cross_validate(train: pd.DataFrame, grid=GRID, folds: int = 5) -> pd.DataFrame:
    """Grouped K-fold over training instances; returns Q90 / mean performance ratio per setting."""
    from sklearn.model_selection import GroupKFold

    from .evaluate import performance_ratios, summarize_ratios

    X, y = train[FEATURES].to_numpy(dtype=float), log_time(train)
    groups = (train["name"] + "|" + train["nvar"].astype(str)).to_numpy()
    rows = []
    for setting in grid:
        ratios, total = [], 0.0
        for tr, va in GroupKFold(folds).split(X, y, groups):
            model = make_model(**setting).fit(X[tr], y[tr])
            m = performance_ratios(train.iloc[va], model.predict(X[va]))
            ratios += list(m["ratio"])
            total += float(m["stats_elapsed_time"].sum())
        rows.append({**setting, **summarize_ratios(ratios), "total_time": total})
    return pd.DataFrame(rows).sort_values(["mean", "Q90"]).reset_index(drop=True)


def within_instance_trend(frame: pd.DataFrame, values) -> float:
    """Median over instances of corr(mem, values): > 0 means 'larger mem looks slower'."""
    w = frame[GROUP_KEY + ["mem"]].copy()
    w["v"] = np.asarray(values, dtype=float)
    corr = w.groupby(GROUP_KEY).apply(lambda g: np.corrcoef(g["mem"], g["v"])[0, 1], include_groups=False)
    return float(corr.median())
