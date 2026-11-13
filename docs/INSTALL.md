# Installation

`memtune` runs on Linux, macOS and Windows. Nothing in the code depends on where the repository lives
or on the directory you start it from. The only hard requirement is the library versions, because the
trained models are pickles.

| Requirement | Version | Why |
|---|---|---|
| Python | 3.10, 3.11 or 3.12 | numpy 1.26 (needed by scikit-learn 1.5.1) has no wheels for 3.13 |
| scikit-learn | 1.5.1 | version the models were saved with |
| XGBoost | **3.2.0** | other versions load the models but give wrong predictions (checked: 3.0.5 gives R² = −52 instead of 0.86) |
| numpy / pandas / scipy | 1.26.4 / 2.2.3 / 1.13.1 | tested combination, pinned in `requirements-lock.txt` |

Disk space: about 100 MB for the repository and packages, plus the dataset (about 100 MB, see
[`../data/README.md`](../data/README.md)).

## 1. Get the code

```bash
git clone https://github.com/nishanmudalige/OptimizationParameterTuning.git
cd OptimizationParameterTuning
```

## 2. Create an environment and install

### Linux and macOS (bash / zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

macOS only: XGBoost needs the OpenMP runtime. If `import xgboost` fails with a `libomp` error, run
`brew install libomp` and try again.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `cmd` and `.venv\Scripts\activate.bat`.
Windows needs no compiler: all packages ship wheels.

### Conda (any system)

```bash
conda env create -f environment.yml
conda activate memtune
```

### Notebooks (optional)

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name memtune
jupyter lab notebooks/
```

Select the `memtune` kernel. The notebooks import `memtune.paths`, so they find data and models without any
edit.

## 3. Check the installation

```bash
python -m memtune info      # shows the resolved paths and whether the library versions are right
python -m pytest            # about 20 s; uses the small sample dataset that is committed
```

`memtune info` prints `versions : OK` when everything matches. A healthy run of the tests ends with
`passed` (tests that need the full dataset are reported as `skipped` until you add it).

## 4. Add the data (needed for `evaluate`, plots and the full tests)

Copy `lbfgs_runs.csv` to `data/` or point to it from anywhere:

| System | Command |
|---|---|
| Linux / macOS | `export MEMTUNE_DATA_DIR=/path/to/folder` |
| Windows PowerShell | `$env:MEMTUNE_DATA_DIR = "D:\path\to\folder"` |
| Windows cmd | `set MEMTUNE_DATA_DIR=D:\path\to\folder` |

or pass `--data` to any command. Then run `python -m pytest` again: the full-data regression tests now run too.

## Where things are looked for

Resolution order for every location: command-line flag, then environment variable, then the repository
layout (found from the location of the package, not from your current directory).

| What | Flag | Environment variable | Default |
|---|---|---|---|
| Repository root | – | `MEMTUNE_ROOT` | folder containing `pyproject.toml` |
| Dataset | `--data` (file or folder) | `MEMTUNE_DATA_DIR` | `data/lbfgs_runs.csv` |
| Models | `--models` | `MEMTUNE_MODELS_DIR` | `models/` |
| Outputs | `--out` | `MEMTUNE_RESULTS_DIR` | `results/` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `RuntimeError: xgboost==3.2.0 required, found …` | `python -m pip install -r requirements-lock.txt` inside the activated environment |
| `No matching distribution found for numpy==1.26.4` | You are on Python 3.13+; use 3.10–3.12 (`py -3.12`, `pyenv`, `conda create -n memtune python=3.11`) |
| `Dataset not found: …` | See step 4 |
| `XGBoostError … libomp` (macOS) | `brew install libomp` |
| Paths with spaces or `OneDrive` folders | Supported. If a OneDrive file is "online only", make it available offline first; otherwise reads fail with an I/O error |
| Windows: "path too long" while cloning | `git config --global core.longpaths true` |
| `ModuleNotFoundError: memtune` in a notebook | The notebook kernel is not the environment where you ran `pip install -e .` |
| Warning "Trying to unpickle estimator … from version 1.5.1" | Appears only with a different scikit-learn; install the pinned one |
