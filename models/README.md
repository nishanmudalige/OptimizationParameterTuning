# Trained models

All models are scikit-learn or XGBoost regressors saved with `joblib`. They were saved with
**scikit-learn 1.5.1** and **XGBoost 3.2.0**; load them with those versions only
(`requirements-lock.txt`). `SHA256SUMS` lists the checksum of every file and is verified by the tests.

> Only unpickle files from sources you trust: loading a pickle can run arbitrary code.

| File | Stage | Input → output | Used for |
|---|---|---|---|
| `stage1/random_forest.pkl` | 1 | 15 problem features (incl. `mem`) → log1p(`neval_obj`, `neval_grad`, `timed_bytes`) | hybrid Stage 1: `neval_obj`, `neval_grad` |
| `stage1/xgboost.pkl` | 1 | same | hybrid Stage 1: `timed_bytes` |
| `stage1/linear_regression.pkl` | 1 | same | linear baseline |
| `stage2_hybrid/random_forest.pkl` | 2 | hybrid Stage 1 output (3 values) → log1p(`stats_elapsed_time`) | **framework, random-forest Stage 2** |
| `stage2_hybrid/xgboost.pkl` | 2 | same | **framework, XGBoost Stage 2** |
| `stage2_ground_truth/random_forest.pkl` | 2 | *observed* `neval_obj`, `neval_grad`, `timed_bytes` → log1p time | Stage 2 on ground-truth inputs |
| `stage2_ground_truth/xgboost.pkl` | 2 | same | same |
| `stage2_ground_truth/linear_regression.pkl` | 2 | same | same |
| `one_step/random_forest.pkl` | – | 15 problem features (incl. `mem`) → log1p(`stats_elapsed_time`) | **one-step model, not in the paper** (see below) |
| `legacy/stage2_rf_on_predicted_stage1.pkl` | 2 | predictions of the random-forest Stage 1 | used only by notebooks 02, 03 and 06 |
| `legacy/stage2_xgb_on_predicted_stage1.pkl` | 2 | same | same |

The `stage2_hybrid` models were trained on out-of-fold Stage 1 predictions of the training instances; the
`stage2_ground_truth` models on observed quantities. Do not feed one kind with the other's scale: the
hybrid models expect log-scale inputs.

Feature order for Stage 1 is `memtune.data.FEATURES`; targets are in `log1p` scale.

## The one-step model (not in the paper)

`one_step/random_forest.pkl` predicts the run time directly from the features, without Stage 1. It exists because
the framework's Stage 2 favours very small `mem` (see [`docs/ONE_STEP_MODEL.md`](../docs/ONE_STEP_MODEL.md)). It is a scikit-learn random
forest (`n_estimators=200, min_samples_leaf=20, max_features=0.5, random_state=0`, settings chosen by grouped
5-fold cross-validation on the training instances), trained on the paper's 172 training instances by
`scripts/train_one_step_model.py`. On Windows the script rebuilds this exact file. On Linux and macOS
scikit-learn draws different random numbers for the same seed, so the forest comes out slightly different with
the same quality (on Linux: 90th percentile 1.41, mean 1.11, total 2,687 s on the test problems, against 1.39,
1.14 and 2,584 s for this file). Unlike the other files it was made in this repository, not for the paper.

## Use

```python
from memtune import load_framework, recommend_mem
fw = load_framework()                      # checks the library versions, loads the framework and one-step models
```

or `python -m memtune recommend problems.csv` ([`../docs/USAGE.md`](../docs/USAGE.md)).

## Names in the earlier repository layout

| Now | Before (`Felix_STA478/…`) |
|---|---|
| `stage1/random_forest.pkl` | `model_weights/best_rf_model_reg.pkl` |
| `stage1/xgboost.pkl` | `model_weights/best_xgb_model_reg.pkl` |
| `stage1/linear_regression.pkl` | `lr_stage1_model.joblib` |
| `stage2_hybrid/random_forest.pkl` | `model_weights/best_hybrid_rf_model_time_new_cv.pkl` |
| `stage2_hybrid/xgboost.pkl` | `model_weights/best_hybrid_xgb_model_time_new_cv.pkl` |
| `stage2_ground_truth/random_forest.pkl` | `model_weights/best_rf_model_time_raw.pkl` |
| `stage2_ground_truth/xgboost.pkl` | `model_weights/best_xgb_model_time_raw.pkl` |
| `stage2_ground_truth/linear_regression.pkl` | `model_weights/best_lr_model_time.pkl` (identical to `lr_stage2_ground_truth.pkl`) |
| `legacy/stage2_rf_on_predicted_stage1.pkl` | `model_weights/best_rf_model_time.pkl` |
| `legacy/stage2_xgb_on_predicted_stage1.pkl` | `model_weights/best_xgb_model_time.pkl` |

The files are byte-identical to the earlier ones (checked with SHA-256). Everything else from the earlier
layout (superseded models, course material) remains available under the git tag `archive/coursework`, e.g.
`https://github.com/nishanmudalige/OptimizationParameterTuning/tree/archive/coursework/Felix_STA478/model_weights`.
