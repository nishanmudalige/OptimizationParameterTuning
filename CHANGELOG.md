# Changelog

All notable changes to this repository. The paper's results are reproduced by every version listed here.

## Unreleased

Added
- `install.sh` / `install.ps1`: one installer for Linux, macOS and Windows that reports the system, recommends a
  setup, asks what to install (virtual environment, conda or Docker; notebooks, demo data, developer tools, tests)
  and checks the result. `scripts/setup.sh` now calls it.
- `python -m memtune reproduce`: recomputes the paper's results and compares all 32 values with the reference.
- `python -m memtune doctor`: checks package versions, the checksums of models and data, and the sample results.
- `python -m memtune demo`: builds the demo data when needed, serves the page locally and opens the browser.
- `scripts/robustness_check.py` and `memtune.robustness` (not in the paper): the evaluation on other train/test
  splits and with whole problems held out.
- A browser test of the demo, a lint check and installer runs on three systems in continuous integration.
- `docs/REPRODUCING.md`, `docs/ONE_STEP_MODEL.md`, `docs/ROBUSTNESS.md`, `CONTRIBUTING.md`, this file.

Changed
- "The framework" now means one model, as in the paper: hybrid Stage 1 and random-forest Stage 2. The XGBoost
  Stage 2 of the journal article is labelled an alternative in the README, the documentation, the demo and the
  output of `reproduce`. Its model, its numbers and `--stage2 gb` are unchanged.
- The fixed `mem = 87` is described as the best single value over all 215 complete instances, as in the paper
  (it was called the best value on the test instances, where the best single value is 86).
- The README was rewritten; the one-step model's description moved to `docs/ONE_STEP_MODEL.md`.
- The demo says "instance" for a problem at one size and states that test instances are new sizes of problems seen
  in training. It opens with a fixed example and explains what to do when its data is missing.

Fixed
- The demo server on Windows no longer starts on a port that another program is using.
- `memtune recommend` without `--output` wrote doubled line endings on Windows (blank lines in some programs).

## 1.0.0

- The `memtune` package, the trained models, the compressed dataset, regression tests against independently computed
  reference results, Docker image, continuous integration on Linux, macOS and Windows.
- The interactive demo (`site/`).
- The one-step model (not in the paper).
- The earlier coursework layout of the repository is kept under the tag `archive/coursework`.
