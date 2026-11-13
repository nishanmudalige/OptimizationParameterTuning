"""End-to-end evaluation: choose ``mem`` per test instance and compare with the best observed run."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from .data import FEATURES, GROUP_KEY, TIME_COLUMN, load_dataset, log_targets, log_time
from .models import Framework, load_framework

#: Differences below this many seconds count as a tie with the best run (ratio 1).
TOLERANCE_SECONDS = 0.10


def performance_ratios(test: pd.DataFrame, score, tol: float = TOLERANCE_SECONDS) -> pd.DataFrame:
    """Performance ratio per instance for a policy.

    ``score`` has one value per row of ``test``; the policy picks the ``mem`` with the *lowest* score
    (ties -> largest ``mem``). The reference is the fastest observed run (ties -> largest ``mem``).
    A chosen run within ``tol`` seconds of the reference counts as ratio 1.
    """
    w = test[GROUP_KEY + ["mem", TIME_COLUMN]].copy()
    w["p"] = np.asarray(score)
    asc = [True, True, True, False]
    chosen = w.sort_values(GROUP_KEY + ["p", "mem"], ascending=asc).groupby(GROUP_KEY).first().reset_index()
    best = w.sort_values(GROUP_KEY + [TIME_COLUMN, "mem"], ascending=asc).groupby(GROUP_KEY).first().reset_index()
    m = chosen.merge(best, on=GROUP_KEY, suffixes=("", "_best"))
    raw = (m[TIME_COLUMN] / m[TIME_COLUMN + "_best"]).to_numpy()
    close = np.abs(m[TIME_COLUMN] - m[TIME_COLUMN + "_best"]) <= tol
    m["ratio"] = np.where(close, 1.0, raw)
    return m


def summarize_ratios(ratios) -> dict:
    r = np.asarray(ratios, dtype=float)
    return {
        "Q50": float(np.quantile(r, 0.50)),
        "Q75": float(np.quantile(r, 0.75)),
        "Q90": float(np.quantile(r, 0.90)),
        "mean": float(r.mean()),
    }


def fixed_mem_score(test: pd.DataFrame, mem: int) -> np.ndarray:
    """Score that makes a policy always choose ``mem`` (used for the baselines)."""
    return -(test["mem"] == mem).astype(float).to_numpy()


def evaluate_framework(data_path=None, models_dir=None, default_mem: int = 5, fixed_mem: int = 87,
                       strict: bool = True, one_step: bool = True) -> dict:
    """Recompute the paper's Stage 1, Stage 2 and end-to-end (Table 4) numbers.

    Returns a nested dict: ``stage1``, ``stage2``, ``table4`` (Q50/Q75/Q90/mean and total times) and
    ``chosen_mem`` (the ``mem`` picked per test instance, key ``"name|nvar"``). With ``one_step`` (and the
    model file present) ``table4`` and ``chosen_mem`` also get a row ``one_step_rf``: the one-step model,
    which is NOT in the paper (see :mod:`memtune.one_step`).
    """
    fw: Framework = load_framework(models_dir, strict=strict)
    df, _, test = load_dataset(data_path)

    X = test[FEATURES].to_numpy(dtype=float)
    y1 = log_targets(test)
    y2 = log_time(test)
    p_rf, p_xgb = fw.stage1_rf.predict(X), fw.stage1_xgb.predict(X)
    X2 = fw.stage1_hybrid(X)

    out = {
        "n_rows": int(len(df)),
        "n_instances": int(len(df[GROUP_KEY].drop_duplicates())),
        "n_test_instances": int(len(test[GROUP_KEY].drop_duplicates())),
        "n_test_rows": int(len(test)),
        "stage1": {
            "rf": {"mse": float(mean_squared_error(y1, p_rf)), "r2": float(r2_score(y1, p_rf))},
            "gb": {"mse": float(mean_squared_error(y1, p_xgb)), "r2": float(r2_score(y1, p_xgb))},
        },
        "stage2": {},
        "table4": {},
        "chosen_mem": {},
    }
    scores = {
        f"baseline1_mem{default_mem}": fixed_mem_score(test, default_mem),
        f"baseline2_mem{fixed_mem}": fixed_mem_score(test, fixed_mem),
    }
    for key, model in (("rf", fw.stage2_rf), ("gb", fw.stage2_xgb)):
        pred = model.predict(X2)
        out["stage2"][key] = {"mse": float(mean_squared_error(y2, pred)), "r2": float(r2_score(y2, pred))}
        scores[f"framework_{key}"] = pred
    if one_step and fw.one_step is not None:
        scores["one_step_rf"] = fw.log_time(X, "one_step")
    for name, score in scores.items():
        m = performance_ratios(test, score)
        out["table4"][name] = {
            **summarize_ratios(m["ratio"]),
            "total_time": float(m[TIME_COLUMN].sum()),
            "oracle_time": float(m[TIME_COLUMN + "_best"].sum()),
        }
        out["chosen_mem"][name] = {f"{n}|{v}": int(k) for n, v, k in m[GROUP_KEY + ["mem"]].itertuples(index=False)}
    return out


def table4_frame(result: dict) -> pd.DataFrame:
    """Table 4 of the paper as a DataFrame."""
    return pd.DataFrame(result["table4"]).T[["Q50", "Q75", "Q90", "mean", "total_time", "oracle_time"]]
