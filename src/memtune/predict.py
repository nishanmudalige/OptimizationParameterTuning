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
    rows["predicted_log_time"] = fw.stage2(stage2).predict(fw.stage1_hybrid(X))
    return rows[["problem_index", "mem", "predicted_log_time"]]


def recommend_mem(problems: pd.DataFrame, framework=None, stage2: str = "rf", models_dir=None) -> np.ndarray:
    """Recommended ``mem`` per row of ``problems``.

    ``problems`` needs the columns in :data:`PROBLEM_FEATURES` (everything in ``data.FEATURES`` except
    ``mem``). Ties go to the largest ``mem``, as in the paper's evaluation.
    """
    fw = framework or load_framework(models_dir)
    long = predicted_times(problems, fw, stage2)
    best = (
        long.sort_values(["problem_index", "predicted_log_time", "mem"], ascending=[True, True, False])
        .groupby("problem_index")
        .first()
    )
    return best["mem"].reindex(range(len(problems))).to_numpy(dtype=int)
