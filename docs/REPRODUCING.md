# Reproducing the results

Everything on this page starts from a fresh clone and the installation in [`INSTALL.md`](INSTALL.md). All commands
work the same on Linux, macOS and Windows, from any directory, once the environment is activated.

## In one command

```bash
python -m memtune reproduce
```

It recomputes the results from the committed models and the dataset (5 seconds) and prints every value next to its
reference, marked `ok` or `DIFFERENT`:

```text
End-to-end performance ratios (43 test instances)
  quantity                                               reference  recomputed
  Default mem = 5 (Baseline 1), median ratio                1.0592      1.0592   ok
  ...
  Alternative, XGBoost Stage 2, 90th pct. ratio             2.0613      2.0613   ok
  mem chosen for every test instance                     reference   identical   ok

32 of 32 reference values reproduced (details: results/reproduction.json).
```

The exit status is 0 only if all 32 values agree (relative tolerance 1e-6; counts and the chosen `mem` must be
identical). Add `--figures` to also write the aggregate figures.

## What is reproduced, and by which command

| Result | Command | Output | Time |
|---|---|---|---|
| Size of the benchmark: 215 complete instances, 43 test instances, 4,300 test runs | `python -m memtune reproduce` | terminal, `results/reproduction.json` | 5 s |
| Stage 1 accuracy on the test set (MSE and R² of the log targets; random forest and XGBoost) | `python -m memtune reproduce` | same | |
| Stage 2 accuracy on the test set (MSE and R² of the log run time; random forest and XGBoost) | `python -m memtune reproduce` | same | |
| End-to-end performance ratios (the paper's Table 4): default `mem = 5`, fixed `mem = 87`, the framework, and the alternative XGBoost Stage 2 of the journal article; median, 75th and 90th percentile, mean, total time | `python -m memtune reproduce` | same | |
| The same table as files | `python -m memtune evaluate` | `results/table4.csv`, `results/evaluation.json` | 5 s |
| Aggregate figures: `mem` against run time, evaluations and memory, over all instances | `python scripts/plot_global_mem.py` | `results/global_mem_plots/*.pdf` (6 files) | 5 s |
| Per-instance figures | `python scripts/plot_per_problem.py [--limit N]` | `results/` (about 3,650 PDF files for all instances) | 13 min for all |
| Comparison of model families for Stage 1 | `python scripts/train_model_comparison.py` (`--quick` for a short run) | `results/model_comparison/` | hours (seconds with `--quick`) |
| The modelling steps as notebooks | `jupyter lab notebooks/` | in the notebook | 10 – 20 s each |

The values of the end-to-end table are in the [README](../README.md#results).

## Why the numbers can be trusted

- **Fixed inputs.** The dataset and every model file have SHA-256 checksums (`data/SHA256SUMS`,
  `models/SHA256SUMS`); the tests and `python -m memtune doctor` verify them.
- **Fixed software.** `requirements-lock.txt` pins every package. The models are pickle files of scikit-learn 1.5.1
  and XGBoost 3.2.0; with another XGBoost the files load but predict wrong values, so the package refuses to run.
- **An independent reference.** `tests/golden/generate_golden.py` re-implements the original notebook logic without
  importing `memtune` and wrote `tests/golden/golden_full.json`. `reproduce` and the regression tests compare the
  package with that file, so an error in the package cannot hide by changing both sides.
- **Every system, every push.** Continuous integration runs the tests on Linux, macOS and Windows with Python 3.10
  and 3.12, and inside the Docker image. The reference values are reproduced to 1e-9 on all of them.

## What is deterministic and what is not

| Step | Same result everywhere? |
|---|---|
| Predictions of the committed models, and everything computed from them (`reproduce`, `evaluate`, `recommend`, the demo) | Yes, on all tested systems |
| The train/test split (seed 66) | Yes |
| Retraining a random forest (`scripts/train_one_step_model.py`, `scripts/robustness_check.py`) | Identical on repeated runs on one system. Windows on one side and Linux / macOS on the other draw different random numbers for the same seed, so the forests differ slightly between them |
| The hyper-parameter searches in the notebooks | They are commented out (1,000 random settings × 5 folds each, hours of computing); the notebooks load the saved models instead |

## What cannot be reproduced from this repository

- **The measurements.** The 21,802 solver runs were collected with Julia scripts (`JSOSolvers.jl`,
  `OptimizationProblems.jl`) that are not part of this repository yet. `scripts/build_dataset.py` turns their logs
  into the dataset and can be tried on the example logs in `tests/data/example_logs/`
  ([`USAGE.md`](USAGE.md#3-build-the-dataset-from-raw-measurements)).
- **Run times on your machine.** Run times, and the timing features of a problem, depend on the hardware that
  measured them.
- **The saved models from scratch.** The models in `models/` are the ones used for the paper. Their hyper-parameters
  are stored in the files; the searches that selected them are documented in the notebooks but were not rerun for
  this repository.

## Beyond the paper

Two analyses in this repository are **not in the paper**; they are kept apart so that the published numbers stay
untouched.

| Analysis | Command | Documentation |
|---|---|---|
| A one-step model without the framework's preference for very small `mem` | `python scripts/train_one_step_model.py --select` | [`ONE_STEP_MODEL.md`](ONE_STEP_MODEL.md) |
| The evaluation on other train/test splits, and with whole problems held out | `python scripts/robustness_check.py` | [`ROBUSTNESS.md`](ROBUSTNESS.md) |
