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

## Cite

If you use this code or data, please cite the paper (see [`CITATION.cff`](CITATION.cff); GitHub shows a
"Cite this repository" button):

> Y. Gao, *Leveraging High-Performance Scientific Computing for the Limited-Memory Hyperparameter Tuning of
> Large-Scale Unconstrained Continuous Optimization Solvers*, Journal of Computational and Data Exploration
> 2(1), 2026. DOI: [10.33137/codex.v2i1.47829](https://doi.org/10.33137/codex.v2i1.47829)

## Quick start

Requires Python 3.10 – 3.12 (3.13 is not supported by the pinned numpy). Full instructions, including
Conda and troubleshooting: [`docs/INSTALL.md`](docs/INSTALL.md).

<table>
<tr><th>Linux / macOS</th><th>Windows (PowerShell)</th></tr>
<tr><td>

```bash
git clone https://github.com/nishanmudalige/OptimizationParameterTuning.git
cd OptimizationParameterTuning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest
```

</td><td>

```powershell
git clone https://github.com/nishanmudalige/OptimizationParameterTuning.git
cd OptimizationParameterTuning
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest
```

</td></tr>
</table>

Recommend `mem` for your own problems (a CSV with one row per problem; columns listed in
[`docs/USAGE.md`](docs/USAGE.md)):

```bash
python -m memtune recommend my_problems.csv --output recommendations.csv
```

## Results

Recomputed from the committed models and the benchmark data with `python -m memtune evaluate`
(215 problem instances × 100 values of `mem` = 21,500 L-BFGS runs; 172 instances for training, **43 for
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

## Repository layout

```
├── src/memtune/        Python package: data preparation, models, evaluation, recommendation, CLI
├── scripts/            command-line scripts: dataset building, plots, model comparison
├── notebooks/          the exploration and modelling notebooks behind the paper (outputs cleared)
├── models/             trained Stage 1 / Stage 2 models and their checksums (see models/README.md)
├── data/               dataset location (not in git; see data/README.md)
├── results/            generated tables and figures (not in git)
├── tests/              regression tests, a small sample dataset and reference ("golden") results
└── docs/               INSTALL.md (setup on every system), USAGE.md (user guide)
```

## Reproducing the results

1. Install (above) and put the dataset where the code can find it ([`data/README.md`](data/README.md)).
2. `python -m memtune evaluate` recomputes Stage 1 and Stage 2 accuracy and the table above.
3. `python -m pytest` runs the regression tests: the packaged code must reproduce results computed
   independently with the original notebook logic ([`tests/README.md`](tests/README.md)).
   Tests that need the full dataset run when it is available and are skipped otherwise.

## Pre-trained models

Eight scikit-learn / XGBoost models in [`models/`](models/README.md). They are pickles: load them only with
the versions in `requirements-lock.txt` (scikit-learn 1.5.1, **XGBoost 3.2.0**). Other XGBoost versions load
the files but silently return wrong predictions, so `memtune` refuses to run with a different one.
Only load pickles from sources you trust.

## License and authors

MIT License, see [`LICENSE`](LICENSE). Authors: Yuchen (Felix) Gao and Nishan Mudalige, University of
Toronto Mississauga. Data collection uses the Julia [JuliaSmoothOptimizers](https://juliasmoothoptimizers.github.io/)
ecosystem (`JSOSolvers.jl`, `OptimizationProblems.jl`).
