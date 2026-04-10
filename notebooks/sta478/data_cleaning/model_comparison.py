"""
=============================================================================
MULTIVARIATE MULTIPLE REGRESSION — FULL MODEL COMPARISON
=============================================================================
Run with: python model_comparison.py
Or in tmux: tmux new -s train && python model_comparison.py | tee run.log

All output is printed with timestamps so you can check progress anytime.
Results are saved to CSV after each model family completes.
=============================================================================
"""

import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Linear Models ---
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor

# --- Tree-Based ---
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor

# --- Neighbors ---
from sklearn.neighbors import KNeighborsRegressor

# --- SVM ---
from sklearn.svm import SVR

# --- Optional ---
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

import pickle


# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = "./datasets/final.csv"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

feature_cols_reg = ["nvar", "tree_length", "tree_depth", "time_obj", "time_grad", "mem"]
target_cols_reg  = ["neval_obj", "neval_grad", "timed_bytes"]
target_col_model = ["stats_elapsed_time"]
group_key = ["problem", "name", "nvar"]


# =============================================================================
# HELPERS
# =============================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def threshold_report(y_true_orig, y_pred_orig, target_names, thresholds=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50]):
    relative_error = np.abs(y_pred_orig - y_true_orig) / np.abs(y_true_orig + 1e-8)
    report = {}
    for i, col in enumerate(target_names):
        log(f"  {col}:")
        report[col] = {}
        for t in thresholds:
            rate = np.mean(relative_error[:, i] < t)
            log(f"    <{t:5.0%}:  {rate:.2%}")
            report[col][f"<{t:.0%}"] = round(rate, 4)
    return report


def format_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# =============================================================================
# LOAD DATA
# =============================================================================
log("=" * 70)
log("LOADING DATA")
log("=" * 70)

df = pd.read_csv(DATA_PATH)
df = df.sort_values(["name", "nvar", "mem"]).reset_index(drop=True)
log(f"Loaded {len(df)} rows from {DATA_PATH}")

first = list(set(group_key + feature_cols_reg + target_cols_reg + target_col_model))
df = df[first + [c for c in df.columns if c not in first]]


# =============================================================================
# SPLIT
# =============================================================================
log("=" * 70)
log("SPLITTING DATA")
log("=" * 70)

instances = df[group_key].drop_duplicates()
instances = instances.sample(frac=1, random_state=66).reset_index(drop=True)

n = len(instances)
n_train = int(0.8 * n)
n_valid = int(0.10 * n)

train_inst = instances.iloc[:n_train]
valid_inst = instances.iloc[n_train:n_train + n_valid]
test_inst  = instances.iloc[n_train + n_valid:]

train_df = df.merge(train_inst, on=group_key, how="inner").reset_index(drop=True)
valid_df = df.merge(valid_inst, on=group_key, how="inner").reset_index(drop=True)
test_df  = df.merge(test_inst,  on=group_key, how="inner").reset_index(drop=True)

log(f"Instances: {n} total | {len(train_inst)} train | {len(valid_inst)} valid | {len(test_inst)} test")
log(f"Rows:      {len(df)} total | {len(train_df)} train | {len(valid_df)} valid | {len(test_df)} test")


# =============================================================================
# PREPARE ARRAYS
# =============================================================================
X_train = train_df[feature_cols_reg].to_numpy(dtype=float)
X_valid = valid_df[feature_cols_reg].to_numpy(dtype=float)
X_test  = test_df[feature_cols_reg].to_numpy(dtype=float)

y_train = np.log1p(train_df[target_cols_reg].to_numpy(dtype=float))
y_valid = np.log1p(valid_df[target_cols_reg].to_numpy(dtype=float))
y_test  = np.log1p(test_df[target_cols_reg].to_numpy(dtype=float))

log(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
log(f"Features: {feature_cols_reg}")
log(f"Targets:  {target_cols_reg}")


# =============================================================================
# DEFINE ALL CANDIDATES
# =============================================================================
candidates = []

candidates.append(("Ridge", Ridge, [
    {"alpha": a} for a in [0.01, 0.1, 1, 10, 100, 1000]
], False))

candidates.append(("Lasso", Lasso, [
    {"alpha": a, "max_iter": 5000} for a in [0.001, 0.01, 0.1, 1.0]
], True))

candidates.append(("ElasticNet", ElasticNet, [
    {"alpha": a, "l1_ratio": r, "max_iter": 5000}
    for a in [0.01, 0.1, 1.0]
    for r in [0.2, 0.5, 0.8]
], True))

candidates.append(("BayesianRidge", BayesianRidge, [{}], True))

candidates.append(("SGD", SGDRegressor, [
    {"loss": loss, "penalty": pen, "alpha": a, "max_iter": 5000, "random_state": 123}
    for loss in ["squared_error", "huber"]
    for pen in ["l2", "elasticnet"]
    for a in [0.0001, 0.001, 0.01]
], True))

candidates.append(("KNN", KNeighborsRegressor, [
    {"n_neighbors": k, "weights": w}
    for k in [3, 5, 10, 20]
    for w in ["uniform", "distance"]
], False))

# candidates.append(("SVR", SVR, [
#     {"kernel": k, "C": c, "epsilon": e}
#     for k in ["rbf", "linear"]
#     for c in [0.1, 1.0, 10.0]
#     for e in [0.01, 0.1]
# ], True))

candidates.append(("DecisionTree", DecisionTreeRegressor, [
    {"max_depth": d, "min_samples_leaf": m, "random_state": 123}
    for d in [5, 10, 20, None]
    for m in [5, 10, 20]
], False))

candidates.append(("RandomForest", RandomForestRegressor, [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": m, "random_state": 123, "n_jobs": -1}
    for n in [100, 300]
    for d in [10, 20, None]
    for m in [5, 10]
], False))

candidates.append(("ExtraTrees", ExtraTreesRegressor, [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": m, "random_state": 123, "n_jobs": -1}
    for n in [100, 300]
    for d in [10, 20, None]
    for m in [5, 10]
], False))

candidates.append(("GradientBoosting", GradientBoostingRegressor, [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr, "min_samples_leaf": m, "random_state": 123}
    for n in [200, 500]
    for d in [3, 5, 7]
    for lr in [0.05, 0.1]
    for m in [5, 10]
], True))

candidates.append(("HistGradientBoosting", HistGradientBoostingRegressor, [
    {"max_iter": n, "max_depth": d, "learning_rate": lr, "min_samples_leaf": m, "random_state": 123}
    for n in [200, 500]
    for d in [5, 10, None]
    for lr in [0.05, 0.1]
    for m in [5, 20]
], True))

candidates.append(("AdaBoost", AdaBoostRegressor, [
    {"n_estimators": n, "learning_rate": lr, "random_state": 123}
    for n in [50, 100, 200]
    for lr in [0.01, 0.1, 1.0]
], True))

if HAS_XGB:
    candidates.append(("XGBoost", XGBRegressor, [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr,
         "min_child_weight": m, "random_state": 123, "verbosity": 0, "n_jobs": -1}
        for n in [200, 500]
        for d in [3, 5, 7]
        for lr in [0.05, 0.1]
        for m in [5, 10]
    ], True))

if HAS_LGBM:
    candidates.append(("LightGBM", LGBMRegressor, [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr,
         "min_child_samples": m, "random_state": 123, "verbosity": -1, "n_jobs": -1}
        for n in [200, 500]
        for d in [5, 10, -1]
        for lr in [0.05, 0.1]
        for m in [5, 20]
    ], True))


total_combos = sum(len(pg) for _, _, pg, _ in candidates)
log(f"Model families: {len(candidates)}")
log(f"Total combinations: {total_combos}")
log(f"Optional libs: XGBoost={'yes' if HAS_XGB else 'NO'} | LightGBM={'yes' if HAS_LGBM else 'NO'}")


# =============================================================================
# RUN ALL SWEEPS
# =============================================================================
results = {}
global_best_name = None
global_best_mse = np.inf
global_best_model = None
run_start = time.time()
combos_done = 0

for fam_idx, (name, model_class, param_grid, needs_wrapper) in enumerate(candidates):
    log("")
    log("=" * 70)
    log(f"[{fam_idx+1}/{len(candidates)}] {name} — {len(param_grid)} combinations")
    log("=" * 70)

    fam_start = time.time()
    best_model = None
    best_mse = np.inf
    best_params = None

    for i, params in enumerate(param_grid):
        iter_start = time.time()
        try:
            base = model_class(**params)
            model = MultiOutputRegressor(base) if needs_wrapper else base
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            mse = mean_squared_error(y_valid, pred)
            r2 = r2_score(y_valid, pred)
            iter_time = time.time() - iter_start

            combos_done += 1
            elapsed_total = time.time() - run_start
            avg_per_combo = elapsed_total / combos_done
            remaining = avg_per_combo * (total_combos - combos_done)

            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_params = params
                log(f"  NEW BEST ({i+1}/{len(param_grid)}) MSE={mse:.4f} R²={r2:.4f} "
                    f"({iter_time:.1f}s) ETA={format_eta(remaining)} | {params}")
            else:
                if (i + 1) % 10 == 0 or (i + 1) == len(param_grid):
                    log(f"  ({i+1}/{len(param_grid)}) best so far MSE={best_mse:.4f} "
                        f"({iter_time:.1f}s) ETA={format_eta(remaining)}")

        except Exception as e:
            combos_done += 1
            log(f"  FAILED ({i+1}/{len(param_grid)}): {str(e)[:80]}")
            continue

    fam_time = time.time() - fam_start

    if best_model is not None:
        pred = best_model.predict(X_valid)
        r2 = r2_score(y_valid, pred)
        results[name] = {
            "model": best_model,
            "params": best_params,
            "mse": best_mse,
            "r2": r2,
            "time": fam_time,
        }

        log(f"  >>> {name} DONE: MSE={best_mse:.4f}  R²={r2:.4f}  Time={format_eta(fam_time)}")
        log(f"      Best params: {best_params}")

        if best_mse < global_best_mse:
            global_best_mse = best_mse
            global_best_name = name
            global_best_model = best_model
            log(f"  *** NEW GLOBAL BEST: {name} MSE={best_mse:.4f} ***")

        # --- Save incremental leaderboard after each family ---
        leaderboard = []
        for rname, rinfo in sorted(results.items(), key=lambda x: x[1]["mse"]):
            leaderboard.append({
                "model": rname,
                "mse": round(rinfo["mse"], 6),
                "r2": round(rinfo["r2"], 6),
                "time_sec": round(rinfo["time"], 1),
                "params": str(rinfo["params"]),
            })
        pd.DataFrame(leaderboard).to_csv(f"{OUTPUT_DIR}/leaderboard.csv", index=False)
        log(f"  Leaderboard saved to {OUTPUT_DIR}/leaderboard.csv")
    else:
        log(f"  >>> {name} FAILED: no valid model found")


# =============================================================================
# FINAL LEADERBOARD
# =============================================================================
total_time = time.time() - run_start
log("")
log("=" * 70)
log(f"FINAL LEADERBOARD (total time: {format_eta(total_time)})")
log("=" * 70)
log(f"{'Rank':>4}  {'Model':25s}  {'MSE':>10}  {'R²':>10}  {'Time':>8}")
log("-" * 70)

sorted_results = sorted(results.items(), key=lambda x: x[1]["mse"])
for rank, (rname, rinfo) in enumerate(sorted_results, 1):
    marker = " <<<< BEST" if rname == global_best_name else ""
    log(f"{rank:4d}  {rname:25s}  {rinfo['mse']:10.4f}  {rinfo['r2']:10.4f}  "
        f"{format_eta(rinfo['time']):>8s}{marker}")


# =============================================================================
# BEST MODEL — DETAILED VALIDATION REPORT
# =============================================================================
log("")
log("=" * 70)
log(f"BEST MODEL: {global_best_name}")
log(f"Params: {results[global_best_name]['params']}")
log(f"Validation MSE={global_best_mse:.4f}  R²={results[global_best_name]['r2']:.4f}")
log("=" * 70)

y_pred_valid = global_best_model.predict(X_valid)
valid_report = threshold_report(np.expm1(y_valid), np.expm1(y_pred_valid), target_cols_reg)

log("")
log("Per-target validation metrics:")
for i, col in enumerate(target_cols_reg):
    mse  = mean_squared_error(y_valid[:, i], y_pred_valid[:, i])
    rmse = np.sqrt(mse)
    r2   = r2_score(y_valid[:, i], y_pred_valid[:, i])
    log(f"  {col:20s}  MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")


# =============================================================================
# TEST SET — FINAL EVALUATION
# =============================================================================
log("")
log("=" * 70)
log("TEST SET EVALUATION (final)")
log("=" * 70)

y_test_pred = global_best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2  = r2_score(y_test, y_test_pred)
log(f"Test MSE={test_mse:.4f}  R²={test_r2:.4f}")

test_report = threshold_report(np.expm1(y_test), np.expm1(y_test_pred), target_cols_reg)

log("")
log("Per-target test metrics:")
for i, col in enumerate(target_cols_reg):
    mse  = mean_squared_error(y_test[:, i], y_test_pred[:, i])
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test[:, i], y_test_pred[:, i])
    log(f"  {col:20s}  MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")


# =============================================================================
# SAVE BEST MODEL + SUMMARY
# =============================================================================
model_path = f"{OUTPUT_DIR}/best_model_{global_best_name}.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "model": global_best_model,
        "name": global_best_name,
        "params": results[global_best_name]["params"],
        "mse": global_best_mse,
        "r2": results[global_best_name]["r2"],
        "feature_cols": feature_cols_reg,
        "target_cols": target_cols_reg,
    }, f)
log(f"Best model saved to {model_path}")

summary = {
    "global_best": global_best_name,
    "global_best_mse": round(global_best_mse, 6),
    "total_time_sec": round(total_time, 1),
    "total_combinations": total_combos,
    "n_train": len(train_df),
    "n_valid": len(valid_df),
    "n_test": len(test_df),
    "test_mse": round(test_mse, 6),
    "test_r2": round(test_r2, 6),
    "valid_report": valid_report,
    "test_report": test_report,
    "leaderboard": {rname: {"mse": round(rinfo["mse"], 6), "r2": round(rinfo["r2"], 6),
                            "params": str(rinfo["params"]), "time": round(rinfo["time"], 1)}
                    for rname, rinfo in sorted_results},
}
with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
log(f"Summary saved to {OUTPUT_DIR}/summary.json")

log("")
log("=" * 70)
log("DONE")
log("=" * 70)
