"""Loading the published Stage 1 / Stage 2 models."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np

from .paths import models_dir as _models_dir

#: Library versions the pickles were written with. XGBoost is strict: other versions can load the
#: files but return wrong predictions (verified: 3.0.5 gives a memory-usage R^2 of -52 instead of 0.86).
REQUIRED_VERSIONS = {"xgboost": "3.2.0", "scikit-learn": "1.5.1"}

#: Relative locations inside ``models/``.
MODEL_FILES = {
    "stage1_rf": "stage1/random_forest.pkl",
    "stage1_xgb": "stage1/xgboost.pkl",
    "stage1_lr": "stage1/linear_regression.pkl",
    "stage2_hybrid_rf": "stage2_hybrid/random_forest.pkl",
    "stage2_hybrid_xgb": "stage2_hybrid/xgboost.pkl",
    "stage2_truth_rf": "stage2_ground_truth/random_forest.pkl",
    "stage2_truth_xgb": "stage2_ground_truth/xgboost.pkl",
    "stage2_truth_lr": "stage2_ground_truth/linear_regression.pkl",
    # not in the paper: see memtune.one_step
    "one_step_rf": "one_step/random_forest.pkl",
}

#: Policies that choose ``mem``: the paper's framework with either Stage 2, and the one-step model.
POLICIES = ("rf", "gb", "one_step")


def check_versions(strict: bool = True) -> list:
    """Compare installed library versions with :data:`REQUIRED_VERSIONS`.

    Returns the list of problems. With ``strict`` an XGBoost mismatch raises ``RuntimeError``,
    because it silently corrupts predictions; other mismatches only warn.
    """
    problems = []
    for pkg, wanted in REQUIRED_VERSIONS.items():
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            have = None
        if have != wanted:
            problems.append(f"{pkg}=={wanted} required, found {have}")
    if problems:
        msg = "; ".join(problems) + ". Install the pinned versions: pip install -r requirements-lock.txt"
        if strict and any(p.startswith("xgboost") for p in problems):
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)
    return problems


def load_model(key: str, directory=None):
    """Load one model by key (see :data:`MODEL_FILES`)."""
    return joblib.load(Path(_models_dir(directory)) / MODEL_FILES[key])


@dataclass
class Framework:
    """The two-stage framework of the paper.

    Stage 1 maps problem features (and ``mem``) to predicted log evaluation counts and log allocated
    bytes; Stage 2 maps those three quantities to log run time. The *hybrid* Stage 1 uses the random
    forest for ``neval_obj`` / ``neval_grad`` and XGBoost for ``timed_bytes``.
    """

    stage1_rf: object
    stage1_xgb: object
    stage2_rf: object
    stage2_xgb: object
    one_step: object = None  # not in the paper: see memtune.one_step

    def stage1_hybrid(self, X: np.ndarray) -> np.ndarray:
        a = self.stage1_rf.predict(X)
        b = self.stage1_xgb.predict(X)
        return np.column_stack([a[:, 0], a[:, 1], b[:, 2]])

    def stage2(self, which: str):
        return {"rf": self.stage2_rf, "xgb": self.stage2_xgb, "gb": self.stage2_xgb}[which]

    def log_time(self, X: np.ndarray, which: str = "rf") -> np.ndarray:
        """Predicted log1p run time for rows of ``data.FEATURES``.

        ``which`` is ``"rf"`` / ``"gb"`` (the paper's framework: Stage 1 hybrid, then that Stage 2) or
        ``"one_step"`` (features -> run time directly; not in the paper).
        """
        if which == "one_step":
            if self.one_step is None:
                raise FileNotFoundError(f"one-step model not found ({MODEL_FILES['one_step_rf']}); "
                                        "run scripts/train_one_step_model.py")
            return self.one_step.predict(X)
        return self.stage2(which).predict(self.stage1_hybrid(X))


def load_framework(directory=None, strict: bool = True) -> Framework:
    check_versions(strict=strict)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        return Framework(
            stage1_rf=load_model("stage1_rf", directory),
            stage1_xgb=load_model("stage1_xgb", directory),
            stage2_rf=load_model("stage2_hybrid_rf", directory),
            stage2_xgb=load_model("stage2_hybrid_xgb", directory),
            one_step=load_model("one_step_rf", directory)
            if (Path(_models_dir(directory)) / MODEL_FILES["one_step_rf"]).is_file() else None,
        )
