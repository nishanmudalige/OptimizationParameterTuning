"""Data preparation and the train/test split used in the paper."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import dataset_path

#: Instance identifier: a problem name and its number of variables.
GROUP_KEY = ["name", "nvar"]

#: Stage 1 input features (``mem`` is the parameter we tune).
FEATURES = [
    "nvar", "mem", "tree_length", "tree_depth", "time_obj", "time_grad", "extract_duration_problem",
    "highest_degree (ExprTree)", "count_plus_minus", "count_trigonometric_function",
    "count_exponential_function", "count_op_function", "alloc_obj", "alloc_grad", "total_alloc",
]

#: Stage 1 targets: evaluation counts and allocated bytes.
STAGE1_TARGETS = ["neval_obj", "neval_grad", "timed_bytes"]

#: Stage 2 target: wall-clock solve time (seconds).
TIME_COLUMN = "stats_elapsed_time"

#: Number of ``mem`` values per instance (1..100); instances with fewer are dropped.
N_MEM = 100

#: Seed and fraction of the instance-level split used in the paper.
SPLIT_SEED = 66
TRAIN_FRACTION = 0.8

REQUIRED_COLUMNS = sorted(set(FEATURES + STAGE1_TARGETS + [TIME_COLUMN] + GROUP_KEY))


def load_raw(path=None) -> pd.DataFrame:
    """Read the merged dataset (only the columns the pipeline needs)."""
    csv = dataset_path(path)
    if not csv.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv}\n"
            "Place it there, point MEMTUNE_DATA_DIR / --data to it, or see data/README.md."
        )
    header = pd.read_csv(csv, nrows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{csv} is missing required columns: {missing}")
    return pd.read_csv(csv, usecols=REQUIRED_COLUMNS, low_memory=False)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the instances that were run for all 100 ``mem`` values.

    Mirrors the notebooks: sort by (name, nvar, mem), replace missing solver counters by 0,
    and keep only complete instances.
    """
    df = raw.copy()
    df = df[GROUP_KEY + [c for c in df.columns if c not in GROUP_KEY]]
    df = df.sort_values(GROUP_KEY + ["mem"]).reset_index(drop=True)
    df[STAGE1_TARGETS] = df[STAGE1_TARGETS].fillna(0)
    df["mem_count"] = df.groupby(GROUP_KEY).transform("size")
    return df[df["mem_count"] == N_MEM].reset_index(drop=True)


def split_instances(df: pd.DataFrame, seed: int = SPLIT_SEED, train_fraction: float = TRAIN_FRACTION):
    """Split by instance (all 100 ``mem`` rows of an instance stay together)."""
    instances = df[GROUP_KEY].drop_duplicates().sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(train_fraction * len(instances))
    train = df.merge(instances.iloc[:n_train], on=["nvar", "name"], how="inner").reset_index(drop=True)
    test = df.merge(instances.iloc[n_train:], on=["nvar", "name"], how="inner").reset_index(drop=True)
    return train, test


def load_dataset(path=None):
    """Convenience: ``(prepared_df, train_df, test_df)``."""
    df = prepare(load_raw(path))
    train, test = split_instances(df)
    return df, train, test


def log_targets(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df[STAGE1_TARGETS].to_numpy(dtype=float))


def log_time(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df[[TIME_COLUMN]].to_numpy(dtype=float)).ravel()
