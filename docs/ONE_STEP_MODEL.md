# The one-step model

> **This model is not in the paper.** It was added to the repository afterwards. The paper's framework and its
> numbers are unchanged; `python -m memtune reproduce` checks them.

## Why the framework often recommends a very small `mem`

Of the 280 instances in the data, the framework recommends `mem` ≤ 3 for 157 with the random-forest Stage 2 and for
176 with the XGBoost Stage 2.

The reason is what Stage 2 gets to see: only the three Stage 1 predictions (objective evaluations, gradient
evaluations, allocated memory). Allocated memory grows with `mem`. Across problems it also grows with the problem
size, and large problems take longer. Stage 2 therefore learned that more allocated memory means a longer run, and
within one problem its predicted run time *rises* with `mem`. The measured run time usually *falls*.

| Within one instance, correlation of `mem` with … (median over the 43 test instances) | |
|---|---:|
| the measured run time | −0.25 |
| the run time predicted by the framework, XGBoost Stage 2 | +0.56 |

On the test instances this costs time. The ratios of the paper's table improve on the default, but the *total* run
time of the framework's choices is larger than with the default, because some of the small-`mem` choices fall on
long-running problems:

| Choice of `mem` (43 test instances) | Median | 75th pct. | 90th pct. | Mean | Total time | `mem` ≤ 3 |
|---|---:|---:|---:|---:|---:|---:|
| Default `mem = 5` | 1.059 | 1.628 | 2.684 | 2.262 | 3,309 s | 0 |
| Fixed `mem = 87`, the best single value on the test instances | 1.000 | 1.056 | 1.280 | 1.093 | 2,474 s | 0 |
| Framework, random-forest Stage 2 (paper) | 1.000 | 1.577 | 2.195 | 1.637 | 4,977 s | 20 |
| Framework, XGBoost Stage 2 (paper) | 1.000 | 1.297 | 2.061 | 1.459 | 4,884 s | 25 |
| **One-step model** | 1.009 | 1.131 | 1.394 | 1.140 | 2,584 s | 0 |

## The model

A random forest that predicts the logarithm of the run time directly from the 15 inputs of Stage 1 (the 14 problem
features and `mem`). Because `mem` is an input, the model can learn its effect separately from the problem size.

| | |
|---|---|
| Model | `sklearn.ensemble.RandomForestRegressor` |
| Settings | 200 trees, `min_samples_leaf = 20`, `max_features = 0.5`, `random_state = 0` |
| Target | `log1p` of the run time in seconds |
| Training data | the same 172 training instances as the framework (17,200 runs) |
| File | `models/one_step/random_forest.pkl` (4.6 MB), checksum in `models/SHA256SUMS` |
| Code | [`src/memtune/one_step.py`](../src/memtune/one_step.py), [`scripts/train_one_step_model.py`](../scripts/train_one_step_model.py) |

**How the settings were chosen.** Six combinations (`min_samples_leaf` 1, 5, 20 × `max_features` 0.5, 1.0) were
compared by 5-fold cross-validation on the training instances, with all runs of an instance in the same fold, and
scored by the mean performance ratio. The 43 test instances were not used for any choice. `--select` reruns this:

```bash
python scripts/train_one_step_model.py --select
```

## Using it

```bash
python -m memtune recommend problems.csv --stage2 one_step
```

```python
from memtune import recommend_mem
problems["recommended_mem"] = recommend_mem(problems, stage2="one_step")
```

`python -m memtune evaluate` prints its row (`one_step_rf`) under the paper's rows, and the demo offers it as a
third model.

## What it does not show

- It does **not** beat the fixed `mem = 87` on the test instances. That value was picked by looking at the test
  results, so the comparison favours it, but it means that on this benchmark a single large `mem` is hard to improve
  on. [`ROBUSTNESS.md`](ROBUSTNESS.md) describes a fairer version of that comparison, with the fixed value chosen on
  the training data.
- The numbers above come from the same single split as the paper's, with the same limits.
- Retraining gives a byte-identical file on Windows. On Linux and macOS scikit-learn draws different random numbers
  for the same seed, so the retrained forest differs slightly (on Linux: 90th percentile 1.41, mean 1.11, total
  2,687 s, against 1.39, 1.14 and 2,584 s for the committed file). The conclusions are the same, and the tests check
  them on every system.
