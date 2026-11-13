# User guide

All commands work the same on Linux, macOS and Windows once the environment is activated
([`INSTALL.md`](INSTALL.md)). They can be started from any directory.

## What runs, and how long

All of the following were run from a fresh clone on Windows 11 (Python 3.12), and the tests also on Ubuntu 24.04.

| What | Command | Time |
|---|---|---|
| Recompute the paper's results | `python -m memtune evaluate` | 5 s |
| Recommend `mem` | `python -m memtune recommend problems.csv` | 5 s |
| Test suite | `python -m pytest` | 25 s (2.5 min with the notebook tests) |
| Merge logs into the dataset | `python scripts/build_dataset.py ...` | 10 s (example logs) |
| Global plots | `python scripts/plot_global_mem.py` | 5 s |
| Per-problem plots | `python scripts/plot_per_problem.py [--limit N]` | 13 min for all, seconds with `--limit` |
| Model comparison | `python scripts/train_model_comparison.py --quick` | seconds; the full sweep takes hours |
| Notebooks 01 – 06 | `jupyter nbconvert --execute ...` | 10 – 20 s each |
| Interactive demo data | `python scripts/export_demo_data.py` | 10 s |

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
python -m memtune evaluate                      # uses data/lbfgs_runs.csv.gz, no unzipping needed
python -m memtune evaluate --data /path/to/lbfgs_runs.csv.gz --out /path/to/output
```

It prints the Stage 1 / Stage 2 accuracy and the end-to-end table, and writes
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
python scripts/export_demo_data.py              # writes site/data/ (280 instances, 2.5 MB, about 10 s)
python -m http.server --directory site 8000     # open http://localhost:8000
```

The page (`site/index.html`, `app.js`, `style.css`) is static: it reads the precomputed files in `site/data/`, made
from the committed models with `memtune.predict.explain`, so it always shows what the package computes (a test checks
that the test-problem recommendations equal the paper's evaluation). The URL keeps the selection, e.g.
`#id=sinquad_1000&model=gb`, so you can link to an example.

Hosting:
- **GitHub Pages:** `.github/workflows/pages.yml` builds, tests and publishes the page on every push to `main` once a
  repository admin has set *Settings → Pages → Source* to **GitHub Actions**. Address: <https://nishanmudalige.github.io/OptimizationParameterTuning/>.
- **Hugging Face:** create a Space with SDK *Static*, run the export, and upload the contents of `site/` (its
  `README.md` already has the Space settings).

Plan and next steps: [`DEMO_ROADMAP.md`](DEMO_ROADMAP.md).

## 8. Tests

```bash
python -m pytest                 # all tests, about 1.5 min (uses the shipped dataset)
python -m pytest -m "not full"   # fast tests only (sample dataset), about 20 s
```

See [`../tests/README.md`](../tests/README.md).
