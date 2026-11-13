# memtune: choosing the L-BFGS memory parameter with machine learning

[![tests](https://github.com/nishanmudalige/OptimizationParameterTuning/actions/workflows/tests.yml/badge.svg)](https://github.com/nishanmudalige/OptimizationParameterTuning/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

L-BFGS has one setting that strongly affects its speed: the **memory** `mem`, the number of past
updates it keeps. [`JSOSolvers.jl`](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) defaults to
`mem = 5`. This repository contains a two-stage statistical-learning framework that recommends `mem` for
an unconstrained problem from cheap problem features, the benchmark data it was built on, the trained
models, and everything needed to reproduce the reported numbers on Linux, macOS or Windows.

1. **Stage 1** predicts, for a problem and a candidate `mem`, the number of objective and gradient
   evaluations and the allocated memory (random forest and XGBoost).
2. **Stage 2** turns those three quantities into a predicted run time (random forest or XGBoost).
3. The framework recommends the `mem` in 1..100 with the smallest predicted run time.

## Interactive demo

[![Screenshot of the interactive demo: Stage 1 and Stage 2 curves over mem, and the recommendations of the framework and the one-step model](docs/images/demo.png)](https://nishanmudalige.github.io/OptimizationParameterTuning/)

**[Open the demo](https://nishanmudalige.github.io/OptimizationParameterTuning/)**. Pick one of the 280 benchmark problems (or a random test problem the models never
saw), press *Run*, and watch it go through the stages: problem features, predicted solver work (Stage 1), predicted
run time for every `mem` from 1 to 100 (Stage 2), and the recommended `mem`, compared with the default 5, the fixed 87,
the one-step model (see [below](#a-one-step-model-not-in-the-paper)) and the best measured run. To run it on your own
machine:

```bash
python scripts/export_demo_data.py              # precompute the data (about 10 s)
python -m http.server --directory site 8000     # then open http://localhost:8000
```

The page is static, so it can be hosted on GitHub Pages (built by `.github/workflows/pages.yml`) or as a Hugging Face
static Space; see the [demo roadmap](docs/DEMO_ROADMAP.md), which also describes the next phase: live features for any
problem and size.

## Cite

If you use this code or data, please cite the paper (see [`CITATION.cff`](CITATION.cff); GitHub shows a
"Cite this repository" button):

> Y. Gao, *Leveraging High-Performance Scientific Computing for the Limited-Memory Hyperparameter Tuning of
> Large-Scale Unconstrained Continuous Optimization Solvers*, Journal of Computational and Data Exploration
> 2(1), 2026. DOI: [10.33137/codex.v2i1.47829](https://doi.org/10.33137/codex.v2i1.47829)

## Quick start

One command creates a private environment (`.venv`) with the exact package versions, installs `memtune` and
runs a smoke test. It needs **Python 3.12** (3.10 and 3.11 also work; 3.13 and newer do **not**, because the
pinned numpy has no build for them). The script finds a suitable Python by itself.

<table>
<tr><th>Linux / macOS</th><th>Windows (PowerShell)</th></tr>
<tr><td>

```bash
git clone https://github.com/nishanmudalige/OptimizationParameterTuning.git
cd OptimizationParameterTuning
./scripts/setup.sh
source .venv/bin/activate
```

</td><td>

```powershell
git clone https://github.com/nishanmudalige/OptimizationParameterTuning.git
cd OptimizationParameterTuning
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.venv\Scripts\Activate.ps1
```

</td></tr>
</table>

Add `--notebooks` (Linux/macOS) or `-Notebooks` (Windows) to also install JupyterLab. Prefer to do it by hand, use
Conda, WSL or Docker, or something went wrong? See [`docs/INSTALL.md`](docs/INSTALL.md). Then check everything with
`python -m pytest`.

No Python setup wanted? `docker build -t memtune https://github.com/nishanmudalige/OptimizationParameterTuning.git#main && docker run --rm memtune`
recomputes the results below in a container (details in the [install guide](docs/INSTALL.md#docker-nothing-to-install-except-docker)).

Recommend `mem` for your own problems (a CSV with one row per problem; columns listed in
[`docs/USAGE.md`](docs/USAGE.md)):

```bash
python -m memtune recommend my_problems.csv --output recommendations.csv
```

## Results

Recomputed from the committed models and the benchmark data with `python -m memtune evaluate`
(215 complete problem instances × 100 values of `mem` = 21,500 L-BFGS runs; 172 instances for training, **43 for
testing**; one fit, one split, seed 66). The *performance ratio* of a policy on an instance is its run time
divided by the run time of the best `mem` in hindsight (1 is perfect).

| Policy | Median | 75th pct. | 90th pct. | Mean |
|---|---:|---:|---:|---:|
| Default `mem = 5` (Baseline 1) | 1.059 | 1.628 | 2.684 | 2.262 |
| Best fixed `mem = 87`, chosen with hindsight (Baseline 2) | 1.000 | 1.056 | 1.280 | 1.093 |
| Framework, random-forest Stage 2 | 1.000 | 1.577 | 2.195 | 1.637 |
| Framework, XGBoost Stage 2 | 1.000 | 1.297 | 2.061 | 1.459 |

Instance-specific selection improves on the default, but a single large fixed value chosen with
hindsight is a strong reference. The numbers come from one train/test split; see the paper for the
discussion and its limitations.

### A one-step model (not in the paper)

The framework tends to recommend a very small `mem`: of the 280 instances in the data, the XGBoost Stage 2 picks
`mem` ≤ 3 for 176, the random forest for 157. The reason is that Stage 2 sees only the three Stage 1 numbers.
Allocated memory grows with `mem`, and across problems it also grows with problem size, so Stage 2 learned that more
memory means a longer run. Within one problem, its predicted run time therefore *rises* with `mem` (median correlation
+0.56 for XGBoost), while the measured run time *falls* (−0.25). On the test problems this costs time: the framework's
choices take 4,884 s (XGBoost) and 4,977 s (random forest) in total, against 3,309 s for the default `mem = 5`.

`memtune` therefore also ships a **one-step model**: a random forest that predicts the run time directly from the 15
features, `mem` included (`models/one_step/`, [`src/memtune/one_step.py`](src/memtune/one_step.py)). It was trained on
the same 172 training instances, with its settings chosen by cross-validation on those instances only; the test
instances were not used.

| Policy (43 test instances) | Median | 75th pct. | 90th pct. | Mean | Total time |
|---|---:|---:|---:|---:|---:|
| Default `mem = 5` | 1.059 | 1.628 | 2.684 | 2.262 | 3,309 s |
| Best fixed `mem = 87`, chosen with hindsight | 1.000 | 1.056 | 1.280 | 1.093 | 2,474 s |
| Framework, XGBoost Stage 2 (paper) | 1.000 | 1.297 | 2.061 | 1.459 | 4,884 s |
| One-step model (not in the paper) | 1.009 | 1.131 | 1.394 | 1.140 | 2,584 s |

It never picks `mem` ≤ 3 on the test problems and beats the framework and the default on every summary, but not the
fixed `mem = 87`, which was chosen by looking at the test results. Use it with
`python -m memtune recommend problems.csv --stage2 one_step`; retrain it with
`python scripts/train_one_step_model.py --select`.

## What you can run

| | Command | Time |
|---|---|---|
| Recompute the results above | `python -m memtune evaluate` | 5 s |
| Recommend `mem` for your problems | `python -m memtune recommend problems.csv` | 5 s |
| Tests (79, incl. all notebooks) | `python -m pytest` | 2.5 min |
| Plots | `python scripts/plot_global_mem.py`, `python scripts/plot_per_problem.py --limit 5` | seconds |
| Compare model families | `python scripts/train_model_comparison.py --quick` | seconds (full sweep: hours) |
| Retrain the one-step model | `python scripts/train_one_step_model.py --select` | 20 s |
| Build the dataset from logs | `python scripts/build_dataset.py` (example logs in `tests/data/example_logs`) | seconds |
| Notebooks | `python -m pip install -r requirements-notebooks.txt`, then `jupyter lab notebooks/` | 10 – 20 s each |
| Interactive demo | `python scripts/export_demo_data.py`, then `python -m http.server --directory site 8000` | 10 s |

Details and exact commands for every system: [`docs/USAGE.md`](docs/USAGE.md). The Julia scripts that collect the
raw measurements are not part of this repository yet; the dataset they produced is in `data/`.

## Repository layout

```
├── src/memtune/        Python package: data preparation, models, evaluation, recommendation, CLI
├── scripts/            command-line scripts: dataset building, plots, model comparison, one-step model, demo data
├── notebooks/          the exploration and modelling notebooks behind the paper (outputs cleared)
├── models/             trained Stage 1 / Stage 2 models, the one-step model, checksums (see models/README.md)
├── data/               the dataset, compressed (lbfgs_runs.csv.gz), and its documentation
├── results/            generated tables and figures (not in git)
├── tests/              regression tests, a small sample dataset and reference ("golden") results
├── site/               the interactive demo (static page; data generated by scripts/export_demo_data.py)
└── docs/               INSTALL.md (setup on every system), USAGE.md (user guide), DEMO_ROADMAP.md
```

## Reproducing the results

1. Install (above). The dataset is included as `data/lbfgs_runs.csv.gz` and is read directly, no unzipping
   needed ([`data/README.md`](data/README.md) explains how to get the plain CSV if you want it).
2. `python -m memtune evaluate` recomputes Stage 1 and Stage 2 accuracy and the table above.
3. `python -m pytest` runs the regression tests: the packaged code must reproduce results computed
   independently with the original notebook logic ([`tests/README.md`](tests/README.md)).

## Pre-trained models

Eight scikit-learn / XGBoost models of the paper, plus the one-step model, in [`models/`](models/README.md). They are pickles: load them only with
the versions in `requirements-lock.txt` (scikit-learn 1.5.1, **XGBoost 3.2.0**). Other XGBoost versions load
the files but silently return wrong predictions, so `memtune` refuses to run with a different one.
Only load pickles from sources you trust.

## License and authors

MIT License, see [`LICENSE`](LICENSE). Authors: Yuchen (Felix) Gao, Tangi Migot and Nishan Mudalige (Gao and Mudalige:
University of Toronto Mississauga). Data collection uses the Julia [JuliaSmoothOptimizers](https://juliasmoothoptimizers.github.io/)
ecosystem (`JSOSolvers.jl`, `OptimizationProblems.jl`).
