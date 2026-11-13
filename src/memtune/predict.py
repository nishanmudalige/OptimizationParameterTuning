"""Recommend ``mem`` for new problems with the trained framework."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES
from .models import Framework, load_framework

MEM_GRID = np.arange(1, 101)
PROBLEM_FEATURES = [c for c in FEATURES if c != "mem"]


def predicted_times(problems: pd.DataFrame, fw: Framework, stage2: str = "rf", mem_grid=MEM_GRID) -> pd.DataFrame:
    """Predicted log run time for every problem and every ``mem`` in ``mem_grid`` (long format)."""
    missing = [c for c in PROBLEM_FEATURES if c not in problems.columns]
    if missing:
        raise ValueError(f"problem table is missing feature columns: {missing}")
    base = problems[PROBLEM_FEATURES].reset_index(drop=True)
    rows = base.loc[base.index.repeat(len(mem_grid))].reset_index().rename(columns={"index": "problem_index"})
    rows["mem"] = np.tile(np.asarray(mem_grid), len(base))
    X = rows[FEATURES].to_numpy(dtype=float)
    rows["predicted_log_time"] = fw.log_time(X, stage2)
    return rows[["problem_index", "mem", "predicted_log_time"]]


def explain(problems: pd.DataFrame, framework=None, mem_grid=MEM_GRID, models_dir=None) -> pd.DataFrame:
    """Every intermediate quantity of the framework, for every problem and every ``mem`` (long format).

    Columns: ``problem_index``, ``mem``, the Stage 1 predictions ``stage1_neval_obj``, ``stage1_neval_grad`` and
    ``stage1_bytes`` (natural scale), and the Stage 2 predicted run times ``stage2_time_rf`` / ``stage2_time_gb``
    in seconds. Also ``stage2_log_time_rf`` / ``stage2_log_time_gb``, the values ``recommend_mem`` minimizes,
    and ``one_step_time`` / ``one_step_log_time`` for the one-step model (not in the paper; see
    :mod:`memtune.one_step`) when its file is present.
    Used by the interactive demo; :func:`recommend_mem` gives the same choice.
    """
    fw = framework or load_framework(models_dir)
    missing = [c for c in PROBLEM_FEATURES if c not in problems.columns]
    if missing:
        raise ValueError(f"problem table is missing feature columns: {missing}")
    base = problems[PROBLEM_FEATURES].reset_index(drop=True)
    rows = base.loc[base.index.repeat(len(mem_grid))].reset_index().rename(columns={"index": "problem_index"})
    rows["mem"] = np.tile(np.asarray(mem_grid), len(base))
    stage1 = fw.stage1_hybrid(rows[FEATURES].to_numpy(dtype=float))  # log1p scale
    out = rows[["problem_index", "mem"]].copy()
    out["stage1_neval_obj"] = np.expm1(stage1[:, 0])
    out["stage1_neval_grad"] = np.expm1(stage1[:, 1])
    out["stage1_bytes"] = np.expm1(stage1[:, 2])
    for key in ("rf", "gb"):
        log_time = fw.stage2(key).predict(stage1)
        out[f"stage2_log_time_{key}"] = log_time
        out[f"stage2_time_{key}"] = np.expm1(log_time)
    if fw.one_step is not None:
        log_time = fw.log_time(rows[FEATURES].to_numpy(dtype=float), "one_step")
        out["one_step_log_time"] = log_time
        out["one_step_time"] = np.expm1(log_time)
    return out


def recommend_mem(problems: pd.DataFrame, framework=None, stage2: str = "rf", models_dir=None) -> np.ndarray:
    """Recommended ``mem`` per row of ``problems``.

    ``problems`` needs the columns in :data:`PROBLEM_FEATURES` (everything in ``data.FEATURES`` except
    ``mem``). Ties go to the largest ``mem``, as in the paper's evaluation.

    ``stage2`` is ``"rf"`` or ``"gb"`` for the paper's framework, or ``"one_step"`` for the one-step model
    (not in the paper; it avoids the framework's bias towards very small ``mem``, see :mod:`memtune.one_step`).
    """
    fw = framework or load_framework(models_dir)
    long = predicted_times(problems, fw, stage2)
    best = (
        long.sort_values(["problem_index", "predicted_log_time", "mem"], ascending=[True, True, False])
        .groupby("problem_index")
        .first()
    )
    return best["mem"].reindex(range(len(problems))).to_numpy(dtype=int)
