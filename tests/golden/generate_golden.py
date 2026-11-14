"""Reference ("golden") results, computed with the ORIGINAL notebook logic.

This script is intentionally independent of the ``memtune`` package: it is a
verbatim re-implementation of ``final_pipeline.ipynb`` / ``multiple_regression_hybrid.ipynb``
(data preparation, split, Stage 1 hybrid, Stage 2 models, mem selection, performance ratios).
The regression tests compare the packaged code against the JSON it writes.

Usage:
    python tests/golden/generate_golden.py --data PATH/new_raw.csv --models PATH/models --out OUT.json
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

F = ["nvar", "mem", "tree_length", "tree_depth", "time_obj", "time_grad", "extract_duration_problem",
     "highest_degree (ExprTree)", "count_plus_minus", "count_trigonometric_function",
     "count_exponential_function", "count_op_function", "alloc_obj", "alloc_grad", "total_alloc"]
T = ["neval_obj", "neval_grad", "timed_bytes"]
G = ["name", "nvar"]


def load_model(folder, original_name, new_name):
    """Load from the original course layout (model_weights/<original_name>) or the new models/ layout."""
    base = Path(folder)
    path = base / original_name
    return joblib.load(path if path.is_file() else base / new_name)


def e2e(te, pred, tol=0.10):
    w = te[G + ["mem", "stats_elapsed_time"]].copy()
    w["p"] = pred
    ch = w.sort_values(G + ["p", "mem"], ascending=[True, True, True, False]).groupby(G).first().reset_index()
    o = w.sort_values(G + ["stats_elapsed_time", "mem"], ascending=[True, True, True, False]).groupby(G).first().reset_index()
    m = ch.merge(o, on=G, suffixes=("", "_b"))
    raw = (m.stats_elapsed_time / m.stats_elapsed_time_b).to_numpy()
    r = np.where(np.abs(m.stats_elapsed_time - m.stats_elapsed_time_b) <= tol, 1.0, raw)
    return r, m


def summary(r):
    return {"Q50": float(np.quantile(r, .5)), "Q75": float(np.quantile(r, .75)),
            "Q90": float(np.quantile(r, .9)), "mean": float(r.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.data, low_memory=False)
    df = df[G + [c for c in df.columns if c not in G]].sort_values(G + ["mem"]).reset_index()
    solver_metrics = ["neval_grad", "neval_obj", "timed_bytes"]
    df[solver_metrics] = df[solver_metrics].fillna(0)
    df = df[list(set(F + T + ["stats_elapsed_time"] + G))]
    df["mem_count"] = df.groupby(G).transform("size")
    df = df[df["mem_count"] == 100].reset_index(drop=True)
    inst = df[G].drop_duplicates().sample(frac=1, random_state=66).reset_index(drop=True)
    ntr = int(0.8 * len(inst))
    tr = df.merge(inst.iloc[:ntr], on=["nvar", "name"]).reset_index(drop=True)  # noqa: F841 (kept: mirrors the notebook)
    te = df.merge(inst.iloc[ntr:], on=["nvar", "name"]).reset_index(drop=True)

    rf1 = load_model(a.models, "best_rf_model_reg.pkl", "stage1/random_forest.pkl")
    gb1 = load_model(a.models, "best_xgb_model_reg.pkl", "stage1/xgboost.pkl")
    rf2 = load_model(a.models, "best_hybrid_rf_model_time_new_cv.pkl", "stage2_hybrid/random_forest.pkl")
    gb2 = load_model(a.models, "best_hybrid_xgb_model_time_new_cv.pkl", "stage2_hybrid/xgboost.pkl")

    Xt = te[F].to_numpy(float)
    yt = np.log1p(te[T].to_numpy(float))
    p_rf, p_gb = rf1.predict(Xt), gb1.predict(Xt)
    X2 = np.column_stack([p_rf[:, 0], p_rf[:, 1], p_gb[:, 2]])
    y2 = np.log1p(te[["stats_elapsed_time"]].to_numpy(float)).ravel()

    out = {
        "n_rows": int(len(df)),
        "n_instances": int(len(inst)),
        "n_train_instances": int(ntr),
        "n_test_instances": int(len(inst) - ntr),
        "n_test_rows": int(len(te)),
        "test_instances": sorted(f"{n}|{v}" for n, v in inst.iloc[ntr:][G].itertuples(index=False)),
        "test_instances_sha256": hashlib.sha256(
            "\n".join(sorted(f"{n}|{v}" for n, v in inst.iloc[ntr:][G].itertuples(index=False))).encode()).hexdigest(),
        "stage1": {
            "rf": {"mse": float(mean_squared_error(yt, p_rf)), "r2": float(r2_score(yt, p_rf))},
            "gb": {"mse": float(mean_squared_error(yt, p_gb)), "r2": float(r2_score(yt, p_gb))},
            "hybrid_X2_sum": float(X2.sum()),
        },
        "stage2": {},
        "table4": {},
        "chosen_mem": {},
    }
    for name, m in (("rf", rf2), ("gb", gb2)):
        p = m.predict(X2)
        out["stage2"][name] = {"mse": float(mean_squared_error(y2, p)), "r2": float(r2_score(y2, p))}
    policies = {
        "baseline1_mem5": -(te.mem == 5).astype(float).to_numpy(),
        "baseline2_mem87": -(te.mem == 87).astype(float).to_numpy(),
        "framework_rf": rf2.predict(X2),
        "framework_gb": gb2.predict(X2),
    }
    for name, p in policies.items():
        r, m = e2e(te, p)
        out["table4"][name] = {**summary(r), "total_time": float(m.stats_elapsed_time.sum()),
                               "oracle_time": float(m.stats_elapsed_time_b.sum())}
        out["chosen_mem"][name] = {f"{n}|{v}": int(k) for n, v, k in m[G + ["mem"]].itertuples(index=False)}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("n_rows", "n_instances", "n_test_instances", "stage1", "stage2", "table4")}, indent=1))


if __name__ == "__main__":
    main()
