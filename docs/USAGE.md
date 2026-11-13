# User guide

All commands work the same on Linux, macOS and Windows once the environment is activated
([`INSTALL.md`](INSTALL.md)). They can be started from any directory.

```
raw measurements ──► build_dataset ──► lbfgs_runs.csv ──► evaluate ──► results/table4.csv
 (data collection)                          │
                                            ├──► plots (scripts/plot_*.py)
                                            └──► model comparison (scripts/train_model_comparison.py)

problem features (CSV) ──► recommend ──► recommended mem
```

## 1. Recommend `mem` for your own problems

Prepare a CSV with one row per problem and these columns (they are the problem features of the paper;
`mem` is *not* needed because the model scans 1..100):

| Column | Meaning |
|---|---|
| `nvar` | number of variables |
| `tree_length`, `tree_depth` | size and depth of the objective's expression tree |
| `highest_degree (ExprTree)` | highest polynomial degree found in the expression tree |
| `count_plus_minus`, `count_trigonometric_function`, `count_exponential_function`, `count_op_function` | operator counts in the expression tree |
| `time_obj`, `time_grad` | time of one objective / gradient evaluation (s) |
| `extract_duration_problem` | time spent extracting the features (s) |
| `alloc_obj`, `alloc_grad`, `total_alloc` | bytes allocated by one objective / gradient evaluation, and in total |

```bash
python -m memtune recommend my_problems.csv --output recommendations.csv
python -m memtune recommend my_problems.csv --stage2 gb      # XGBoost Stage 2 instead of the random forest
```

The output is your CSV with an added `recommended_mem` column. From Python:

```python
import pandas as pd
from memtune import recommend_mem

problems = pd.read_csv("my_problems.csv")
problems["recommended_mem"] = recommend_mem(problems, stage2="rf")   # or "gb"
```

Ties between equally good `mem` values go to the largest one, as in the paper's evaluation.
The paper's results are for the unconstrained problems of `OptimizationProblems.jl` and L-BFGS from
`JSOSolvers.jl`; treat recommendations for very different problems with care.

## 2. Reproduce the paper's results

```bash
python -m memtune evaluate                      # needs the dataset (data/README.md)
python -m memtune evaluate --data /path/to/lbfgs_runs.csv --out /path/to/output
```

It prints the Stage 1 / Stage 2 accuracy and the end-to-end table, and writes
`evaluation.json` and `table4.csv` to the results folder. The expected output is in the table of the
[README](../README.md). The split is by problem instance (80 % / 20 %, seed 66), so all 100 `mem`
values of an instance are on the same side.

## 3. Build the dataset from raw measurements

```bash
python scripts/build_dataset.py                                   # merge problem_metadata.csv + solver_runtimes.csv in data/
python scripts/build_dataset.py --problem-logs DIR --solver-logs DIR   # start from the raw log files
python scripts/build_dataset.py --time-limit 3600
```

Failed runs get a penalty time: an error counts as twice the time limit, a time-out as the time limit.
Output: `lbfgs_runs.csv` (all runs) and `lbfgs_runs_success.csv` (runs without error). The columns are
documented in [`../data/README.md`](../data/README.md).

## 4. Plots

```bash
python scripts/plot_global_mem.py   [--data CSV] [--out DIR]   # average metric vs mem over all instances
python scripts/plot_per_problem.py  [--data CSV] [--out DIR]   # one folder of plots per problem instance
```

The scripts write PDFs under `results/` and need no display (they use a non-interactive backend).

## 5. Compare model families (long)

```bash
python scripts/train_model_comparison.py [--data CSV] [--out DIR]
```

Fits many regression families on the Stage 1 targets and writes a leaderboard and the best model to
`results/model_comparison/`. It can take hours; run it in `tmux`/`screen` on a server. It never writes into
`models/`.

## 6. Notebooks

| Notebook | Content |
|---|---|
| `01_data_exploration` | the dataset, missing values, distributions |
| `02_random_forest` | Stage 1 and Stage 2 random forests, hyper-parameter search (searches are commented out; the saved models are loaded) |
| `03_gradient_boosting` | the same with XGBoost |
| `04_linear_regression` | linear-regression baselines |
| `05_hybrid_framework` | the hybrid Stage 1 and the end-to-end evaluation behind the paper's tables |
| `06_end_to_end_pipeline` | an earlier end-to-end pipeline (random-forest Stage 1 only, older Stage 2 model in `models/legacy/`); kept for reference |

Run them top to bottom. They load the committed models and write anything they train to
`results/retrained_models/`, never into `models/`.

## 7. Tests

```bash
python -m pytest                 # fast tests: sample dataset, no extra files
python -m pytest -m full         # only the tests that need the full dataset
```

See [`../tests/README.md`](../tests/README.md).
