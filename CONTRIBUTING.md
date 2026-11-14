# Contributing

Bug reports, questions and suggestions are welcome as
[issues](https://github.com/nishanmudalige/OptimizationParameterTuning/issues). For a bug, please include the output
of `python -m memtune doctor` and the command that failed.

## Development setup

```bash
git clone https://github.com/nishanmudalige/OptimizationParameterTuning.git
cd OptimizationParameterTuning
./install.sh --yes --with notebooks,dev        # Windows PowerShell: powershell -ExecutionPolicy Bypass -File install.ps1 --yes --with notebooks,dev
source .venv/bin/activate                      # Windows: .venv\Scripts\Activate.ps1
```

## Before you open a pull request

```bash
python -m ruff check .          # lint
python -m pytest                # all tests, about 3 minutes
python -m memtune reproduce     # the paper's results must still be reproduced
```

Continuous integration runs the same checks on Linux, macOS and Windows with Python 3.10 and 3.12, the installer on
all three systems, the Docker image, and the demo page in a browser.

## Rules that keep the results reproducible

- **The paper's numbers do not change.** `tests/golden/*.json` holds reference results computed independently of the
  package. If a change alters them, the reason must be understood and explained in the pull request before the
  reference is regenerated ([`tests/README.md`](tests/README.md)).
- **Models and data are fixed.** Files under `models/` and `data/` have checksums in `SHA256SUMS`. Scripts and
  notebooks write to `results/`, never into `models/`.
- **Pinned versions.** `requirements-lock.txt` is the reference environment. The saved models need scikit-learn 1.5.1
  and XGBoost 3.2.0 exactly.
- **Anything beyond the paper is labelled.** Analyses and models that are not in the paper say so in their code,
  their documentation and the demo, and are evaluated separately from the paper's results.
- **No machine-specific paths.** Locations come from `memtune.paths`; a test checks that no source file contains an
  absolute path.
- **Every system.** Code must run on Linux, macOS and Windows. `install.sh` must work with bash 3.2, the version
  that ships with macOS.

## Where things are

| | |
|---|---|
| `src/memtune/` | the package (`data`, `models`, `evaluate`, `predict`, `one_step`, `robustness`, `reproduce`, `doctor`, `demo`, `paths`) |
| `scripts/` | command-line scripts that are not part of the package |
| `tests/` | tests, the sample dataset and the reference results |
| `site/` | the demo page; its data is generated, not committed |
| `docs/` | user documentation |
