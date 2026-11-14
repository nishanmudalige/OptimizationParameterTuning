# User guide

All commands work the same on Linux, macOS and Windows once the environment is activated
([`INSTALL.md`](INSTALL.md)). They can be started from any directory.

## What runs, and how long

All of the following were run from a fresh clone on Windows 11 (Python 3.12), and the tests also on Ubuntu 24.04.

| What | Command | Time |
|---|---|---|
| Recompute the paper's results and compare them with the reference | `python -m memtune reproduce` | 5 s |
| Write the result tables to `results/` | `python -m memtune evaluate` | 5 s |
| Check the installation | `python -m memtune doctor` | 5 s |
| Interactive demo | `python -m memtune demo` | 10 s the first time |
| Recommend `mem` | `python -m memtune recommend problems.csv` | 5 s |
| Test suite | `python -m pytest` | 3 min (1 min without the notebook tests) |
| Merge logs into the dataset | `python scripts/build_dataset.py ...` | 10 s (example logs) |
| Global plots | `python scripts/plot_global_mem.py` | 5 s |
| Per-problem plots | `python scripts/plot_per_problem.py [--limit N]` | 13 min for all, seconds with `--limit` |
| Model comparison | `python scripts/train_model_comparison.py --quick` | seconds; the full sweep takes hours |
| Notebooks 01 – 06 | `jupyter nbconvert --execute ...` | 10 – 20 s each |
| Other train/test splits (not in the paper) | `python scripts/robustness_check.py` | 3 min |

Julia: the data-collection scripts (parallel L-BFGS runs over problems and `mem` values) are written in Julia and
are **not part of this repository yet**; everything here starts from their output, the dataset in `data/`.

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
python -m memtune recommend my_problems.csv --stage2 one_step  # the one-step model (not in the paper)
```

The output is your CSV with an added `recommended_mem` column. From Python:

```python
import pandas as pd
from memtune import recommend_mem

problems = pd.read_csv("my_problems.csv")
problems["recommended_mem"] = recommend_mem(problems, stage2="rf")   # or "gb", or "one_step"
```

Ties between equally good `mem` values go to the largest one, as in the paper's evaluation.
The paper's results are for the unconstrained problems of `OptimizationProblems.jl` and L-BFGS from
`JSOSolvers.jl`; treat recommendations for very different problems with care.

`"rf"` and `"gb"` are the paper's framework. It tends to pick a very small `mem` (Stage 2 links more allocated
memory to a longer run). `"one_step"` is a random forest that predicts the run time directly from the features; it
is not in the paper, avoids that bias and did better on the test instances
([`ONE_STEP_MODEL.md`](ONE_STEP_MODEL.md)). Retrain it with `python scripts/train_one_step_model.py` (`--select`
first reruns the cross-validation that chose its settings, on the training instances only).

## 2. Reproduce the paper's results

```bash
python -m memtune reproduce                     # every value next to its reference, marked ok or DIFFERENT
python -m memtune evaluate                      # uses data/lbfgs_runs.csv.gz, no unzipping needed
python -m memtune evaluate --data /path/to/lbfgs_runs.csv.gz --out /path/to/output
```

`reproduce` is the check: it exits with status 0 only if all 32 reference values agree, and writes
`results/reproduction.json` ([`REPRODUCING.md`](REPRODUCING.md) lists every result and its command). `evaluate`
writes the tables. It prints the Stage 1 / Stage 2 accuracy and the end-to-end table (the paper's rows, plus a row `one_step_rf`
for the one-step model, which is not in the paper), and writes
`evaluation.json` and `table4.csv` to the results folder. The expected output is in the table of the
[README](../README.md). The split is by problem instance (80 % / 20 %, seed 66), so all 100 `mem`
values of an instance are on the same side.

## 3. Build the dataset from raw measurements

The script merges the logs of the data-collection step into `lbfgs_runs.csv`. The logs themselves (and the
Julia scripts that write them) are not in this repository; a tiny synthetic example of the log format is in
`tests/data/example_logs/`, so you can try the script immediately:

```bash
python scripts/build_dataset.py --data-dir results/demo \
    --problem-logs tests/data/example_logs/problem_logs --solver-logs tests/data/example_logs/solver_logs
```

```powershell
python scripts\build_dataset.py --data-dir results\demo `
    --problem-logs tests\data\example_logs\problem_logs --solver-logs tests\data\example_logs\solver_logs
```

With your own logs, point `--problem-logs` and `--solver-logs` at the folders that contain the
`output.json_*.log` files (searched recursively). If you already have the two intermediate tables
(`problem_metadata.csv`, `solver_runtimes.csv`) in `data/`, run it without arguments. Options:
`--data-dir`, `--time-limit 3600`, `--gzip` (write `lbfgs_runs.csv.gz`).

Failed runs get a penalty time: an error counts as twice the time limit, a time-out message as the time limit.
Output: `lbfgs_runs.csv` (all runs) and `lbfgs_runs_success.csv` (runs without error). The columns are
documented in [`../data/README.md`](../data/README.md).

## 4. Plots

```bash
python scripts/plot_global_mem.py   [--data CSV] [--out DIR]               # 6 PDFs, 5 seconds
python scripts/plot_per_problem.py  [--data CSV] [--out DIR] [--limit N]   # PDFs for each problem instance
```

`plot_per_problem.py` writes about 3,650 PDFs (47 MB) for all 280 instances and takes about 13 minutes; use
`--limit 5` to plot just the first five instances in a few seconds. Output goes under `results/` and no display
is needed (a non-interactive backend is used).

## 5. Compare model families

```bash
python scripts/train_model_comparison.py --quick                       # 1 setting per family, a few seconds
python scripts/train_model_comparison.py --families Ridge,RandomForest # only these families (full grid)
python scripts/train_model_comparison.py                               # full sweep: hundreds of fits, hours
```

Fits many regression families on the Stage 1 targets using the complete (error-free) rows, and writes a
leaderboard, a summary and the best model to `results/model_comparison/`. The full sweep can take hours; run it
in `tmux`/`screen` on a server. It never writes into `models/`. LightGBM is used only if installed.

## 6. Notebooks

| Notebook | Content |
|---|---|
| `01_data_exploration` | the dataset, missing values, distributions |
| `02_random_forest` | Stage 1 and Stage 2 random forests (the hyper-parameter searches are commented out; the saved models are loaded) |
| `03_gradient_boosting` | the same with XGBoost |
| `04_linear_regression` | linear-regression baselines |
| `05_hybrid_framework` | the hybrid Stage 1 and the end-to-end evaluation behind the paper's tables |
| `06_end_to_end_pipeline` | an earlier end-to-end pipeline (random-forest Stage 1 only, older Stage 2 model in `models/legacy/`); kept for reference |

Install the notebook tools once (`python -m pip install -r requirements-notebooks.txt`, see
[`INSTALL.md`](INSTALL.md)), then run them top to bottom in JupyterLab, or headlessly with
`jupyter nbconvert --to notebook --execute --output-dir results/notebooks notebooks/<name>.ipynb`.
They load the committed models and write anything they train to `results/retrained_models/`, never into
`models/`. The outputs are cleared in the repository; run a notebook to see its results.

## 7. Interactive demo

```bash
python -m memtune demo                 # computes the data if needed, starts a local server, opens the browser
python -m memtune demo --port 8080     # another port (if the port is taken, the next free one is used)
python -m memtune demo --build-only    # only compute site/data/ (280 instances, 2.8 MB, about 10 s)
python -m memtune demo --rebuild       # recompute the data even if it is up to date
```

The data is recomputed automatically when a model, the dataset or the export script is newer than it. Opening
`site/index.html` by double-click does not work, because a browser does not let a local file read its data; the page
then says so. The page loads its chart library (Plotly) from a public CDN, so it needs an internet connection.

The page (`site/index.html`, `app.js`, `style.css`) is static: it reads the precomputed files in `site/data/`, made
from the committed models with `memtune.predict.explain`, so it always shows what the package computes (a test checks
that the recommendations for the test instances equal the paper's evaluation, and another test runs the page in a
real browser). The URL keeps the selection, e.g. `#id=sinquad_1000&model=gb`, so you can link to an example. The
model shown first is the paper's framework with the random-forest Stage 2.

Hosting:
- **GitHub Pages:** `.github/workflows/pages.yml` builds, tests and publishes the page on every push to `main` once a
  repository admin has set *Settings → Pages → Source* to **GitHub Actions**. Address: <https://nishanmudalige.github.io/OptimizationParameterTuning/>.
- **Hugging Face:** create a Space with SDK *Static*, run `python -m memtune demo --build-only`, and upload the contents of `site/` (its
  `README.md` already has the Space settings).

Plan and next steps: [`DEMO_ROADMAP.md`](DEMO_ROADMAP.md).

## 8. Other train/test splits (not in the paper)

```bash
python scripts/robustness_check.py                 # 20 other splits and whole problems held out, about 3 min
```

Refits all models on other splits and compares the default, a fixed `mem` chosen on the training data, the framework
and the one-step model. Method and output files: [`ROBUSTNESS.md`](ROBUSTNESS.md).

## 9. Check the installation

```bash
python -m memtune doctor            # versions, checksums, a recomputation of the sample results
python -m memtune doctor --quick    # versions and checksums only
python -m memtune info              # the resolved paths
```

## 10. Tests

```bash
python -m pytest                 # all tests, about 3 min (uses the shipped dataset)
python -m pytest -m "not full"   # fast tests only (sample dataset), under a minute
python -m ruff check .           # the linter (python -m pip install -r requirements-dev.txt)
```

See [`../tests/README.md`](../tests/README.md).
