# Installation

`memtune` runs on Linux, macOS and Windows. Nothing in the code depends on where the repository lives
or on the directory you start it from. The only hard requirement is the library versions, because the
trained models are pickles.

| Requirement | Version | Why |
|---|---|---|
| Python | **3.12** recommended; 3.10 and 3.11 also tested | numpy 1.26 (needed by scikit-learn 1.5.1) has no wheels for 3.13 or newer, so 3.13 and 3.14 do **not** work |
| scikit-learn | 1.5.1 | version the models were saved with |
| XGBoost | **3.2.0** | other versions load the models but give wrong predictions (checked: 3.0.5 gives R² = −52 instead of 0.86) |
| numpy / pandas / scipy | 1.26.4 / 2.2.3 / 1.13.1 | tested combination, pinned in `requirements-lock.txt` |

Disk space: about 600 MB for the environment (1 GB with the notebook tools). The dataset is included (1.9 MB
compressed, 98 MB if you unzip it, see [`../data/README.md`](../data/README.md)).

## Fastest way: the installer

```bash
./install.sh                                              # Linux, macOS, WSL, Git Bash
```

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1      # Windows PowerShell
```

The installer works in four steps and changes nothing before you confirm the plan.

1. **System report.** It detects the operating system, the processor, the free disk space, every Python on the
   `PATH` (and, on Windows, the versions known to the `py` launcher) with a note whether memtune supports it,
   conda, Docker, [uv](https://docs.astral.sh/uv/) and, on macOS, the OpenMP runtime that XGBoost needs.
2. **Recommendation.** A private virtual environment in `.venv` if a supported Python (or uv) is there, otherwise
   conda, otherwise Docker. If none of these is possible, it prints the command that installs Python 3.12 on your
   system and offers to run it.
3. **Your choices.** Where to install (venv, conda or Docker), which Python if several are supported, and the
   optional parts:

   | Extra | What it adds | Default |
   |---|---|---|
   | `notebooks` | JupyterLab and the tools to run `notebooks/` (about 300 MB) | no |
   | `demo` | the interactive demo's data, computed from the models (10 seconds, no download) | yes |
   | `dev` | the linter used by the project (`requirements-dev.txt`) | no |
   | `tests` | a run of the complete test suite after installing (about 3 minutes) | no |

4. **Plan, installation, check.** It lists the steps, asks to proceed, installs, and finishes with
   `python -m memtune doctor`, which verifies the package versions, the checksums of the models and the dataset,
   and that the sample results are reproduced exactly.

Example of the report on Windows:

```text
System
  Operating system : Windows (MINGW64_NT, Git Bash)
  Processor        : x86_64
  Free disk space  : 336 GB in /c/Users/you/OptimizationParameterTuning

Python (memtune needs 3.10 to 3.12)
  py -3.12         : 3.12.10  supported      C:\Users\you\AppData\Local\Programs\Python\Python312\python.exe
  py -3.10         : 3.10.11  supported      C:\Users\you\AppData\Local\Programs\Python\Python310\python.exe

Tools
  conda            : not found
  Docker           : found, running

Recommended for this system: a virtual environment (.venv) with Python 3.12.10.
```

Options for use without questions (scripts, servers, continuous integration):

| Command | Effect |
|---|---|
| `./install.sh --detect` | print the report and the recommendation, then stop |
| `./install.sh --dry-run` | print the plan, change nothing |
| `./install.sh --yes` | install the recommended setup with the default extras |
| `./install.sh --yes --with notebooks,dev` | choose extras (`--without demo`, `--full`, `--minimal`) |
| `./install.sh --env conda` / `--env docker` | use conda or Docker instead of a virtual environment |
| `./install.sh --python python3.11` | use this Python (on Windows for example `--python "py -3.11"`) |
| `./install.sh --venv .venv-linux` | another folder for the environment |

The same options work with `install.ps1`. Without a terminal (a pipe, a CI job) the installer behaves as with
`--yes`.

What the installer will not do: it never deletes an existing environment without asking (with `--yes` it stops and
explains instead), it never installs Python with a package manager unless you agree (or pass `--install-python`),
and apart from that it writes only inside the repository folder. It is safe to run again; an existing `.venv` is
reused and brought up to date.

`install.ps1` runs `install.sh` with the bash of Git for Windows. If Git for Windows is not installed it falls back
to `scripts\setup.ps1`, a plain PowerShell setup without questions. On machines with WSL, do not start the
installer with a bare `bash install.sh` from PowerShell: that `bash` is the Linux one and would build a Linux
environment. `scripts/setup.sh` is kept for older instructions and calls `install.sh --yes --env venv`.

The sections below do the same by hand.

### Which Python do I have?

| System | Check | Install Python 3.12 |
|---|---|---|
| Windows | `py -0p` lists every installed version | `winget install Python.Python.3.12`, or python.org (tick "Add python.exe to PATH") |
| macOS | `python3 --version` | `brew install python@3.12` |
| Ubuntu / Debian | `python3 --version` (24.04 has 3.12) | `sudo apt install python3 python3-venv python3-pip` |
| any | – | `conda create -n memtune python=3.12`, or [uv](https://docs.astral.sh/uv/): `uv venv --python 3.12` |

Several Python versions can live side by side; the virtual environment uses exactly the one that created it.
On Windows `py -3.12 -m venv .venv` picks 3.12 even if your default `python` is a different version.

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

Debian / Ubuntu (including WSL): `python3 -m venv` needs one extra package, otherwise it stops with
"ensurepip is not available". Install it once with `sudo apt install python3-venv python3-pip` and repeat
the commands.

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

### Windows with WSL (Ubuntu or another Linux distribution)

Follow the Linux instructions inside the WSL terminal, and keep **both the repository and the dataset in the
Linux file system** (for example `~/OptimizationParameterTuning`), not under `/mnt/c/...`. Reading Windows
files from WSL goes through a slow cross-system layer, and the tests and notebooks read the 100 MB dataset
and the pickled models repeatedly. Microsoft explains this in
[Working across file systems](https://learn.microsoft.com/en-us/windows/wsl/filesystems#file-storage-and-performance-across-file-systems).
The dataset comes with the repository; if you use a different copy, copy it into the Linux file system too instead of
pointing `MEMTUNE_DATA_DIR` at a Windows folder. Use either the Windows environment or the WSL one for a
given clone, not both: a `.venv` created by Windows Python does not work in WSL.

### Conda (any system)

```bash
conda env create -f environment.yml
conda activate memtune
```

### Docker (nothing to install except Docker)

The repository has a `Dockerfile` (Python 3.12 on Debian, pinned packages, tests and data included). It is the
quickest way to get a working, identical environment on any machine that runs Docker (Windows, macOS, Linux).

```bash
docker build -t memtune https://github.com/nishanmudalige/OptimizationParameterTuning.git#main
# or, from a clone:  docker build -t memtune .

docker run --rm memtune                          # recompute the paper's results (python -m memtune evaluate)
docker run --rm memtune python -m memtune reproduce    # ... and compare every value with the reference
docker run --rm memtune python -m pytest         # run the tests
```

Get the result files out of the container, or recommend `mem` for your own CSV:

```bash
# Linux / macOS
docker run --rm -v "$PWD/out:/app/results" memtune
docker run --rm -v "$PWD/problems.csv:/data/problems.csv:ro" memtune python -m memtune recommend /data/problems.csv
```

```powershell
# Windows PowerShell
docker run --rm -v "${PWD}\out:/app/results" memtune
docker run --rm -v "${PWD}\problems.csv:/data/problems.csv:ro" memtune python -m memtune recommend /data/problems.csv
```

The interactive demo from the container (then open <http://localhost:8000>):

```bash
docker run --rm -p 8000:8000 memtune python -m memtune demo --host 0.0.0.0 --no-browser
```

Another Python version (3.10 – 3.12): `docker build --build-arg PYTHON_VERSION=3.10 -t memtune .`.
The first build takes about two minutes (it downloads the packages); later builds reuse the cache.

### Notebooks (optional)

```bash
python -m pip install -r requirements-notebooks.txt         # JupyterLab, ipykernel, nbclient (pinned)
python -m ipykernel install --user --name memtune --display-name "memtune"
jupyter lab notebooks/                                      # interactive
```

The notebooks use the standard `python3` kernel of the environment you started Jupyter from, so activate the
environment first (or pick the `memtune` kernel). They import `memtune.paths`, so they find data and models
without any edit. To run one without opening Jupyter (this is what the tests do):

```bash
jupyter nbconvert --to notebook --execute --output-dir results/notebooks notebooks/05_hybrid_framework.ipynb
```

Each notebook takes 10 – 20 seconds. Anything they train is written to `results/retrained_models/`, never to
`models/`. On Windows you may see a harmless `Proactor event loop does not implement add_reader` warning from
`zmq`.

## 3. Check the installation

```bash
python -m memtune doctor    # versions, checksums of models and data, and a recomputation of the sample results
python -m memtune reproduce # the paper's results, each value compared with the reference
python -m pytest            # the complete test suite, about 3 minutes
```

`memtune doctor` prints one line per check (`[ ok ]`, `[warn]` or `[FAIL]`) and ends with `Everything works.` when
nothing failed. A healthy run of the tests ends with `passed`. Some tests are reported as `skipped` when an optional
tool is missing: the six notebook tests need `requirements-notebooks.txt`, the browser tests of the demo need
Playwright ([`../tests/README.md`](../tests/README.md)).

## 4. The data

The dataset is already in the repository as `data/lbfgs_runs.csv.gz` (compressed, 1.9 MB). **No unzipping is
needed**: `python -m memtune evaluate` and the whole test suite read it directly, so step 3 already ran the
full regression tests too. To get the plain CSV for another program, see
[`../data/README.md`](../data/README.md) (`gunzip -k data/lbfgs_runs.csv.gz`, or
`python -m gzip -d data/lbfgs_runs.csv.gz` on any system, including Windows).

To use a dataset stored elsewhere, point to it from anywhere:

| System | Command |
|---|---|
| Linux / macOS | `export MEMTUNE_DATA_DIR=/path/to/folder` |
| Windows PowerShell | `$env:MEMTUNE_DATA_DIR = "D:\path\to\folder"` |
| Windows cmd | `set MEMTUNE_DATA_DIR=D:\path\to\folder` |

or pass `--data` (a file or a folder) to any command. The folder may contain `lbfgs_runs.csv` or
`lbfgs_runs.csv.gz`.

## Where things are looked for

Resolution order for every location: command-line flag, then environment variable, then the repository
layout (found from the location of the package, not from your current directory).

| What | Flag | Environment variable | Default |
|---|---|---|---|
| Repository root | – | `MEMTUNE_ROOT` | folder containing `pyproject.toml` |
| Dataset | `--data` (file or folder) | `MEMTUNE_DATA_DIR` | `data/lbfgs_runs.csv.gz` (or `.csv`) |
| Models | `--models` | `MEMTUNE_MODELS_DIR` | `models/` |
| Outputs | `--out` | `MEMTUNE_RESULTS_DIR` | `results/` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Anything unclear | `python -m memtune doctor` names the check that fails and what to run |
| `The folder .venv exists but cannot be used here` | The environment was made by another system (Windows and WSL share the folder) or its Python was removed. Delete `.venv`, or keep it and run `./install.sh --venv .venv-linux` |
| The demo page says the data has not been computed, or it was opened as a file | Start it with `python -m memtune demo`; a browser cannot read the data when the page is opened by double-click |
| `RuntimeError: xgboost==3.2.0 required, found …` | `python -m pip install -r requirements-lock.txt` inside the activated environment |
| `metadata-generation-failed` / `Unknown compiler(s)` while installing numpy | You are on Python 3.13 or newer, which has no numpy 1.26 wheels, so pip tries to compile it. Use Python 3.10–3.12 (`py -3.12`, `pyenv`, `conda create -n memtune python=3.11`) or Docker |
| `Dataset not found: …` | Check that `data/lbfgs_runs.csv.gz` exists (it is part of the repository) or see step 4 |
| `ensurepip is not available` (Debian/Ubuntu) | `sudo apt install python3-venv python3-pip` |
| `XGBoostError … libomp` (macOS) | `brew install libomp` |
| Paths with spaces or `OneDrive` folders | Supported. If a OneDrive file is "online only", make it available offline first; otherwise reads fail with an I/O error |
| Windows: "path too long" while cloning | `git config --global core.longpaths true` |
| Very slow tests or notebooks in WSL | The repository or data is under `/mnt/c`; move both into the Linux home (see the WSL section) |
| `ModuleNotFoundError: memtune` in a notebook | The notebook kernel is not the environment where you ran `pip install -e .` |
| Warning "Trying to unpickle estimator … from version 1.5.1" | Appears only with a different scikit-learn; install the pinned one |
