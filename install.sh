#!/usr/bin/env bash
# memtune installer for Linux, macOS and Windows (Git Bash or WSL).
#
#   ./install.sh                 detect the system, ask a few questions, install
#   ./install.sh --yes           no questions: install the recommended setup
#   ./install.sh --detect        only print the system report and the recommendation
#   ./install.sh --dry-run       show what would be done, change nothing
#
# Options
#   --env venv|conda|docker      where to install (default: the recommendation for this system)
#   --python CMD                 Python to use for the virtual environment, e.g. python3.11 or "py -3.11"
#   --venv DIR                   folder of the virtual environment (default: .venv)
#   --with LIST                  extras, comma separated: notebooks, demo, dev, tests
#   --without LIST               leave extras out (default extras: demo)
#   --full                       all extras (same as --with notebooks,demo,dev,tests)
#   --minimal                    no extras
#   --install-python             with --yes: may install Python with the system's package manager
#   -y, --yes                    accept the defaults, never ask
#   -h, --help                   this text
#
# Extras
#   notebooks   JupyterLab and the tools to run notebooks/        (about 300 MB more)
#   demo        precompute the interactive demo's data            (10 seconds, no downloads)
#   dev         linter used by the project (ruff)
#   tests       run the complete test suite after installing     (about 3 minutes)
#
# The script only writes inside this folder (.venv, site/data, results), unless you choose conda or Docker,
# or agree to install Python with your package manager. It is safe to run again.
# Written for bash 3.2 and newer (the bash that ships with macOS).
set -euo pipefail
cd "$(dirname "$0")"

MIN_MINOR=10   # supported Python: 3.10 ... 3.12 (the pinned numpy has no build for 3.13 and newer)
MAX_MINOR=12

ENV_KIND=""; PY_CMD=""; VENV_DIR=".venv"; ASSUME_YES=0; DRY_RUN=0; DETECT_ONLY=0; INSTALL_PYTHON=0
WITH_NOTEBOOKS=0; WITH_DEMO=1; WITH_DEV=0; WITH_TESTS=0; EXTRAS_GIVEN=0

say()  { printf '%s\n' "$*"; }
warn() { printf 'Warning: %s\n' "$*" >&2; }
die()  { printf 'Error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

set_extras() {  # set_extras LIST 0|1
  local item old_ifs="$IFS"
  IFS=','
  for item in $1; do
    case "$item" in
      notebooks) WITH_NOTEBOOKS=$2 ;;
      demo) WITH_DEMO=$2 ;;
      dev) WITH_DEV=$2 ;;
      tests) WITH_TESTS=$2 ;;
      "") ;;
      *) IFS="$old_ifs"; die "unknown extra '$item' (choose from: notebooks, demo, dev, tests)" ;;
    esac
  done
  IFS="$old_ifs"
  EXTRAS_GIVEN=1
}

while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --detect) DETECT_ONLY=1 ;;
    --install-python) INSTALL_PYTHON=1 ;;
    --env) [ $# -ge 2 ] || die "--env needs a value"; ENV_KIND="$2"; shift ;;
    --env=*) ENV_KIND="${1#*=}" ;;
    --python) [ $# -ge 2 ] || die "--python needs a value"; PY_CMD="$2"; shift ;;
    --python=*) PY_CMD="${1#*=}" ;;
    --venv) [ $# -ge 2 ] || die "--venv needs a value"; VENV_DIR="$2"; shift ;;
    --venv=*) VENV_DIR="${1#*=}" ;;
    --with) [ $# -ge 2 ] || die "--with needs a value"; set_extras "$2" 1; shift ;;
    --with=*) set_extras "${1#*=}" 1 ;;
    --without) [ $# -ge 2 ] || die "--without needs a value"; set_extras "$2" 0; shift ;;
    --without=*) set_extras "${1#*=}" 0 ;;
    --full) set_extras "notebooks,demo,dev,tests" 1 ;;
    --minimal) set_extras "notebooks,demo,dev,tests" 0 ;;
    --notebooks) set_extras "notebooks" 1 ;;   # kept for scripts/setup.sh
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1 (see ./install.sh --help)" ;;
  esac
  shift
done
case "$ENV_KIND" in ""|venv|conda|docker) ;; *) die "--env must be venv, conda or docker" ;; esac
# Without a terminal (a pipe, CI) nobody can answer: use the defaults.
if [ ! -t 0 ]; then ASSUME_YES=1; fi

# ----------------------------------------------------------------------------------------- detection
OS="unknown"; OS_NAME="$(uname -s)"; WSL=0
case "$(uname -s)" in
  Linux*)
    OS="linux"
    if [ -r /etc/os-release ]; then
      OS_NAME="$(. /etc/os-release && printf '%s' "${PRETTY_NAME:-Linux}")"
      DISTRO_ID="$(. /etc/os-release && printf '%s' "${ID:-} ${ID_LIKE:-}")"
    fi
    # WSL, but not a container of Docker Desktop (which also runs on the WSL kernel)
    if grep -qi microsoft /proc/version 2>/dev/null && [ ! -e /.dockerenv ]; then WSL=1; OS_NAME="$OS_NAME (WSL)"; fi ;;
  Darwin*) OS="macos"; OS_NAME="macOS $(sw_vers -productVersion 2>/dev/null || uname -r)" ;;
  MINGW*|MSYS*|CYGWIN*) OS="windows"; OS_NAME="Windows ($(uname -s | cut -d- -f1), Git Bash)" ;;
esac
DISTRO_ID="${DISTRO_ID:-}"
ARCH="$(uname -m)"
FREE_KB="$(df -Pk . 2>/dev/null | awk 'NR==2 {print $4}')"
case "$FREE_KB" in ''|*[!0-9]*) FREE_KB=0 ;; esac
FREE_GB=$((FREE_KB / 1024 / 1024))

VENV_DIR="${VENV_DIR%/}"
if [ "$OS" = "windows" ]; then VENV_PY="$VENV_DIR/Scripts/python.exe"; else VENV_PY="$VENV_DIR/bin/python"; fi

py_version() {  # py_version CMD...  -> "3.12.4" or nothing; CMD may contain arguments ("py -3.12")
  # shellcheck disable=SC2086
  $1 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null | tr -d '\r'
}
py_exe() {
  # shellcheck disable=SC2086
  $1 -c 'import os, sys; print(os.path.realpath(sys.executable))' 2>/dev/null | tr -d '\r'
}
is_supported() {  # is_supported VERSION
  local minor
  case "$1" in 3.*) ;; *) return 1 ;; esac
  minor="${1#3.}"; minor="${minor%%.*}"
  [ "$minor" -ge "$MIN_MINOR" ] && [ "$minor" -le "$MAX_MINOR" ]
}

# One line per interpreter: "command|version|path|yes or no". Newest supported first.
PYTHONS=""; BEST_PY=""; BEST_VER=""
detect_pythons() {
  local candidates cmd ver exe ok seen=""
  PYTHONS=""; BEST_PY=""; BEST_VER=""
  if [ "$OS" = "windows" ]; then
    candidates="py -3.12|py -3.11|py -3.10|python|python3|py"
  else
    candidates="python3.12|python3.11|python3.10|python3|python|python3.13|python3.14"
  fi
  local old_ifs="$IFS"
  IFS='|'
  for cmd in $candidates; do
    IFS="$old_ifs"
    if have "${cmd%% *}"; then
      ver="$(py_version "$cmd" || true)"
      if [ -n "$ver" ]; then
        exe="$(py_exe "$cmd" || true)"
        case "$seen" in *"|$exe|"*) IFS='|'; continue ;; esac
        seen="$seen|$exe|"
        if is_supported "$ver"; then
          ok="yes"
          if [ -z "$BEST_PY" ]; then BEST_PY="$cmd"; BEST_VER="$ver"; fi
        else
          ok="no"
        fi
        PYTHONS="$PYTHONS$cmd|$ver|$exe|$ok
"
      fi
    fi
    IFS='|'
  done
  IFS="$old_ifs"
}
detect_pythons

CONDA=""
for c in conda mamba micromamba; do if have "$c"; then CONDA="$c"; break; fi; done
DOCKER="none"
if have docker; then
  if docker info >/dev/null 2>&1; then DOCKER="running"; else DOCKER="stopped"; fi
fi
UV=0; if have uv; then UV=1; fi

# XGBoost on macOS needs the OpenMP runtime, which Apple does not ship.
LIBOMP="n/a"
if [ "$OS" = "macos" ]; then
  LIBOMP="missing"
  for f in /opt/homebrew/opt/libomp/lib/libomp.dylib /usr/local/opt/libomp/lib/libomp.dylib \
           /opt/local/lib/libomp/libomp.dylib; do
    if [ -f "$f" ]; then LIBOMP="found"; fi
  done
fi

existing_venv_version() { if [ -x "$VENV_PY" ]; then py_version "$VENV_PY" || true; fi; }
VENV_VER="$(existing_venv_version)"

report() {
  say "memtune installer"
  say "================="
  say ""
  say "System"
  say "  Operating system : $OS_NAME"
  say "  Processor        : $ARCH"
  say "  Free disk space  : ${FREE_GB} GB in $(pwd)"
  say ""
  say "Python (memtune needs 3.$MIN_MINOR to 3.$MAX_MINOR)"
  if [ -z "$PYTHONS" ]; then
    say "  none found"
  else
    printf '%s' "$PYTHONS" | while IFS='|' read -r cmd ver exe ok; do
      if [ "$ok" = "yes" ]; then note="supported"; else note="not supported"; fi
      printf '  %-16s : %-8s %-14s %s\n' "$cmd" "$ver" "$note" "$exe"
    done
  fi
  if [ -n "$VENV_VER" ]; then say "  existing $VENV_DIR   : Python $VENV_VER"
  elif [ -e "$VENV_DIR" ]; then say "  existing $VENV_DIR   : not usable here (made on another system or with a removed Python)"; fi
  say ""
  say "Tools"
  say "  conda            : ${CONDA:-not found}"
  case "$DOCKER" in
    running) say "  Docker           : found, running" ;;
    stopped) say "  Docker           : found, but not running (start Docker Desktop or the docker service)" ;;
    *)       say "  Docker           : not found" ;;
  esac
  if [ "$UV" = 1 ]; then say "  uv               : found (can download Python 3.12 without admin rights)"; fi
  if [ "$OS" = "macos" ]; then say "  OpenMP (libomp)  : $LIBOMP (needed by XGBoost)"; fi
  say ""
}

# Recommendation: a private virtual environment whenever a supported Python (or uv) is there.
RECOMMEND=""
recommend() {
  if [ -n "$BEST_PY" ] || [ "$UV" = 1 ]; then RECOMMEND="venv"
  elif [ -n "$CONDA" ]; then RECOMMEND="conda"
  elif [ "$DOCKER" = "running" ]; then RECOMMEND="docker"
  else RECOMMEND=""; fi
}
recommend

python_install_command() {  # the command that installs a supported Python on this system, or nothing
  case "$OS" in
    macos) if have brew; then echo "brew install python@3.12"; fi ;;
    windows) if have winget; then echo "winget install -e --id Python.Python.3.12"; fi ;;
    linux)
      case " $DISTRO_ID " in
        *" ubuntu "*|*" debian "*) echo "sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip" ;;
        *" fedora "*|*" rhel "*|*" centos "*) echo "sudo dnf install -y python3.12" ;;
        *" suse "*|*" opensuse "*) echo "sudo zypper install -y python312" ;;
      esac ;;
  esac
}

no_python_help() {
  say "No supported Python was found, and neither conda nor a running Docker."
  say "Install Python 3.12 in one of these ways, then run ./install.sh again:"
  case "$OS" in
    macos)   say "  Homebrew:        brew install python@3.12        (https://brew.sh)"
             say "  or the installer from https://www.python.org/downloads/macos/" ;;
    windows) say "  winget:          winget install -e --id Python.Python.3.12"
             say "  or the installer from https://www.python.org/downloads/windows/  (tick \"Add python.exe to PATH\")" ;;
    *)       say "  Ubuntu / Debian: sudo apt-get install python3 python3-venv python3-pip"
             say "  Fedora:          sudo dnf install python3.12"
             say "  (if your distribution's Python is 3.13 or newer, use one of the two lines below)" ;;
  esac
  say "  any system:      uv (https://docs.astral.sh/uv/) downloads Python 3.12 without admin rights"
  say "  any system:      Miniforge (https://conda-forge.org/download/), then ./install.sh --env conda"
  say "  any system:      Docker, then ./install.sh --env docker"
}

report
if [ -n "$RECOMMEND" ]; then
  case "$RECOMMEND" in
    venv)   if [ -n "$BEST_PY" ]; then say "Recommended for this system: a virtual environment ($VENV_DIR) with Python $BEST_VER."
            else say "Recommended for this system: a virtual environment ($VENV_DIR); uv will download Python 3.12."; fi ;;
    conda)  say "Recommended for this system: a conda environment named memtune (no supported Python on PATH)." ;;
    docker) say "Recommended for this system: the Docker image (no supported Python and no conda)." ;;
  esac
else
  say "Recommended for this system: install Python 3.12 first (see below)."
fi
say ""
if [ "$DETECT_ONLY" = 1 ]; then
  if [ -z "$RECOMMEND" ]; then no_python_help; fi
  exit 0
fi

# ----------------------------------------------------------------------------------------- questions
ask() {  # ask "question" "default" -> ANSWER
  ANSWER="$2"
  if [ "$ASSUME_YES" = 1 ]; then return 0; fi
  printf '%s [%s]: ' "$1" "$2"
  local reply=""
  IFS= read -r reply || true
  if [ -n "$reply" ]; then ANSWER="$reply"; fi
}
confirm() {  # confirm "question" "y|n" -> status 0 for yes
  local hint="y/N"; if [ "$2" = "y" ]; then hint="Y/n"; fi
  ANSWER="$2"
  if [ "$ASSUME_YES" = 0 ]; then
    printf '%s [%s]: ' "$1" "$hint"
    local reply=""
    IFS= read -r reply || true
    if [ -n "$reply" ]; then ANSWER="$reply"; fi
  fi
  case "$ANSWER" in y|Y|yes|Yes|YES) return 0 ;; *) return 1 ;; esac
}

# No way to install yet: offer to install Python with the system's package manager.
if [ -z "$ENV_KIND" ] && [ -z "$RECOMMEND" ]; then
  INSTALL_CMD="$(python_install_command)"
  if [ -z "$INSTALL_CMD" ]; then no_python_help; exit 1; fi
  say "No supported Python was found. This command installs one:"
  say "  $INSTALL_CMD"
  if [ "$DRY_RUN" = 1 ]; then say "(dry run: not installing)"; exit 0; fi
  if { [ "$ASSUME_YES" = 1 ] && [ "$INSTALL_PYTHON" = 1 ]; } ||
     { [ "$ASSUME_YES" = 0 ] && confirm "Run it now? (it may ask for your password)" "n"; }; then
    sh -c "$INSTALL_CMD"
    hash -r
    detect_pythons
    recommend
    if [ -z "$BEST_PY" ]; then
      say ""
      say "Python was installed, but this terminal does not see a supported version yet."
      say "Open a new terminal and run ./install.sh again. If it still fails:"
      no_python_help
      exit 1
    fi
    say "Found Python $BEST_VER."
  else
    say ""
    no_python_help
    exit 1
  fi
fi

if [ -z "$ENV_KIND" ]; then
  ENV_KIND="$RECOMMEND"
  if [ "$ASSUME_YES" = 0 ]; then
    say "Where should memtune be installed?"
    say "  1) venv    a private environment in $VENV_DIR inside this folder; nothing else on the system changes"
    say "  2) conda   a conda environment named memtune (Python 3.11)"
    say "  3) docker  a container image named memtune; nothing is installed on this system"
    case "$RECOMMEND" in venv) d=1 ;; conda) d=2 ;; *) d=3 ;; esac
    ask "Choice" "$d"
    case "$ANSWER" in 1|venv) ENV_KIND="venv" ;; 2|conda) ENV_KIND="conda" ;; 3|docker) ENV_KIND="docker" ;;
      *) die "please answer 1, 2 or 3" ;; esac
    say ""
  fi
fi

case "$ENV_KIND" in
  venv)
    if [ -n "$PY_CMD" ]; then
      ver="$(py_version "$PY_CMD" || true)"
      [ -n "$ver" ] || die "'$PY_CMD' does not run"
      is_supported "$ver" || die "'$PY_CMD' is Python $ver; memtune needs 3.$MIN_MINOR to 3.$MAX_MINOR"
      BEST_PY="$PY_CMD"; BEST_VER="$ver"
    elif [ -z "$BEST_PY" ] && [ "$UV" = 0 ]; then
      no_python_help; exit 1
    elif [ -n "$VENV_VER" ] && is_supported "$VENV_VER"; then
      :   # an existing, usable environment is kept as it is
    elif [ "$ASSUME_YES" = 0 ] && [ "$(printf '%s' "$PYTHONS" | grep -c '|yes$' || true)" -gt 1 ]; then
      say "Several supported Python versions were found:"
      i=0
      printf '%s' "$PYTHONS" | grep '|yes$' | while IFS='|' read -r cmd ver exe ok; do
        i=$((i + 1)); printf '  %d) Python %-8s %s\n' "$i" "$ver" "$exe"
      done
      ask "Which one? (the results were verified with 3.10 and 3.12)" "1"
      case "$ANSWER" in ''|*[!0-9]*) die "please answer with a number" ;; esac
      line="$(printf '%s' "$PYTHONS" | grep '|yes$' | sed -n "${ANSWER}p")"
      [ -n "$line" ] || die "no such choice: $ANSWER"
      BEST_PY="${line%%|*}"; BEST_VER="$(printf '%s' "$line" | cut -d'|' -f2)"
      say ""
    fi ;;
  conda) [ -n "$CONDA" ] || die "conda was not found. Install Miniforge (https://conda-forge.org/download/) first." ;;
  docker)
    [ "$DOCKER" != "none" ] || die "Docker was not found. Install it from https://docs.docker.com/get-docker/ first."
    [ "$DOCKER" = "running" ] || die "Docker is installed but not running. Start it and run ./install.sh again." ;;
esac

if [ "$ASSUME_YES" = 0 ] && [ "$EXTRAS_GIVEN" = 0 ] && [ "$ENV_KIND" != "docker" ]; then
  say "Optional parts:"
  if confirm "  Notebooks: JupyterLab and the tools to run notebooks/ (about 300 MB more)?" "n"; then WITH_NOTEBOOKS=1; fi
  if confirm "  Demo: precompute the interactive demo's data (10 seconds)?" "y"; then WITH_DEMO=1; else WITH_DEMO=0; fi
  if confirm "  Developer tools: the linter used by the project?" "n"; then WITH_DEV=1; fi
  if confirm "  Run the complete test suite after installing (about 3 minutes)?" "n"; then WITH_TESTS=1; fi
  say ""
fi

REQ="requirements-lock.txt"
if [ "$WITH_NOTEBOOKS" = 1 ]; then REQ="requirements-notebooks.txt"; fi
NEED_GB=2; if [ "$WITH_NOTEBOOKS" = 1 ]; then NEED_GB=3; fi
if [ "$ENV_KIND" = "docker" ]; then NEED_GB=4; fi

# ----------------------------------------------------------------------------------------- plan
RECREATE_VENV=0
if [ "$ENV_KIND" = "venv" ] && [ -e "$VENV_DIR" ]; then
  if [ -z "$VENV_VER" ] || ! is_supported "$VENV_VER"; then
    # Never delete an environment without asking: it may belong to another system that shares this folder
    # (Windows and WSL, for example).
    say "The folder $VENV_DIR exists but cannot be used here (Python ${VENV_VER:-not runnable on this system})."
    if [ "$DRY_RUN" = 1 ]; then RECREATE_VENV=1
    elif [ "$ASSUME_YES" = 0 ] && confirm "Delete it and create a new one?" "n"; then RECREATE_VENV=1
    else
      say "Nothing was changed. Either delete $VENV_DIR yourself, or keep it and choose another folder:"
      say "  ./install.sh --venv .venv-$OS"
      exit 1
    fi
    say ""
  fi
fi
NEED_LIBOMP=0
if [ "$OS" = "macos" ] && [ "$LIBOMP" = "missing" ] && [ "$ENV_KIND" != "docker" ]; then NEED_LIBOMP=1; fi

say "Plan"
n=0
step() { n=$((n + 1)); say "  $n. $*"; }
case "$ENV_KIND" in
  venv)
    if [ "$NEED_LIBOMP" = 1 ]; then
      if have brew; then step "Install the OpenMP runtime that XGBoost needs:  brew install libomp"
      else step "(you) Install the OpenMP runtime that XGBoost needs: install Homebrew (https://brew.sh), then run  brew install libomp"; fi
    fi
    if [ -x "$VENV_PY" ] && [ "$RECREATE_VENV" = 0 ]; then step "Reuse the existing $VENV_DIR (Python $VENV_VER)"
    elif [ "$RECREATE_VENV" = 1 ]; then step "Delete the existing $VENV_DIR and create a new one with Python ${BEST_VER:-3.12}"
    elif [ -n "$BEST_PY" ]; then step "Create $VENV_DIR with Python $BEST_VER ($BEST_PY)"
    else step "Create $VENV_DIR with Python 3.12, downloaded by uv"; fi
    step "Install the pinned packages from $REQ and memtune itself"
    if [ "$WITH_DEV" = 1 ]; then step "Install the developer tools from requirements-dev.txt"; fi ;;
  conda)
    if [ "$NEED_LIBOMP" = 1 ] && have brew; then step "Install the OpenMP runtime that XGBoost needs:  brew install libomp"; fi
    step "Create or update the conda environment 'memtune' from environment.yml (Python 3.11, pinned packages)"
    if [ "$WITH_NOTEBOOKS" = 1 ]; then step "Install the notebook tools from requirements-notebooks.txt"; fi
    if [ "$WITH_DEV" = 1 ]; then step "Install the developer tools from requirements-dev.txt"; fi ;;
  docker)
    step "Build the image 'memtune' from the Dockerfile (Python 3.12, pinned packages)" ;;
esac
if [ "$WITH_DEMO" = 1 ] && [ "$ENV_KIND" != "docker" ]; then step "Precompute the demo data (site/data)"; fi
step "Check the installation (memtune doctor)"
if [ "$WITH_TESTS" = 1 ]; then step "Run the complete test suite"; fi
say ""
if [ "$FREE_GB" -lt "$NEED_GB" ] && [ "$FREE_KB" -gt 0 ]; then
  warn "only ${FREE_GB} GB free here; this setup needs about ${NEED_GB} GB."
fi
if [ "$DRY_RUN" = 1 ]; then say "Dry run: nothing was changed."; exit 0; fi
if ! confirm "Proceed?" "y"; then say "Nothing was changed."; exit 0; fi
say ""

# ----------------------------------------------------------------------------------------- install
run() { say "+ $*"; "$@"; }

if [ "$NEED_LIBOMP" = 1 ]; then
  if have brew; then run brew install libomp
  else warn "libomp is missing and Homebrew was not found; XGBoost will not load until libomp is installed."; fi
fi

case "$ENV_KIND" in
  venv)
    if [ "$RECREATE_VENV" = 1 ]; then run rm -rf "$VENV_DIR"; fi
    if [ ! -x "$VENV_PY" ]; then
      if [ -n "$BEST_PY" ]; then
        say "+ $BEST_PY -m venv $VENV_DIR"
        # shellcheck disable=SC2086
        if ! $BEST_PY -m venv "$VENV_DIR"; then
          rm -rf "$VENV_DIR"
          say ""
          say "Could not create the virtual environment."
          if [ "$OS" = "linux" ]; then
            say "On Debian / Ubuntu the venv module is a separate package. Run:"
            say "  sudo apt-get install -y python3-venv python3-pip"
            say "and then ./install.sh again."
          fi
          exit 1
        fi
      else
        run uv venv --python 3.12 --seed "$VENV_DIR"
      fi
    fi
    PY="$VENV_PY"
    say "Installing $REQ (one to three minutes) ..."
    run "$PY" -m pip install -q --upgrade pip
    run "$PY" -m pip install -q -r "$REQ"
    run "$PY" -m pip install -q -e . --no-deps
    if [ "$WITH_DEV" = 1 ]; then run "$PY" -m pip install -q -r requirements-dev.txt; fi
    RUN_PY() { "$PY" "$@"; } ;;
  conda)
    if "$CONDA" env list | awk '{print $1}' | grep -qx memtune; then
      run "$CONDA" env update -n memtune -f environment.yml
    else
      run "$CONDA" env create -f environment.yml
    fi
    RUN_PY() { "$CONDA" run -n memtune python "$@"; }
    if [ "$WITH_NOTEBOOKS" = 1 ]; then say "+ pip install -r requirements-notebooks.txt"; RUN_PY -m pip install -q -r requirements-notebooks.txt; fi
    if [ "$WITH_DEV" = 1 ]; then say "+ pip install -r requirements-dev.txt"; RUN_PY -m pip install -q -r requirements-dev.txt; fi ;;
  docker)
    run docker build -t memtune .
    RUN_PY() { docker run --rm memtune python "$@"; } ;;
esac

if [ "$WITH_DEMO" = 1 ] && [ "$ENV_KIND" != "docker" ]; then
  say ""
  RUN_PY -m memtune demo --build-only
fi
say ""
RUN_PY -m memtune doctor
if [ "$WITH_TESTS" = 1 ]; then
  say ""
  say "Running the test suite (about 3 minutes) ..."
  RUN_PY -m pytest -q
fi

# ----------------------------------------------------------------------------------------- next steps
say ""
say "Installation finished."
say ""
case "$ENV_KIND" in
  venv)
    say "Activate the environment in every new terminal:"
    if [ "$OS" = "windows" ]; then
      say "  source $VENV_DIR/Scripts/activate          (Git Bash)"
      say "  $VENV_DIR\\Scripts\\Activate.ps1             (PowerShell)"
    else
      say "  source $VENV_DIR/bin/activate"
    fi
    P="python" ;;
  conda)
    say "Activate the environment in every new terminal:"
    say "  conda activate memtune"
    P="python" ;;
  docker)
    say "Run commands in the container, for example:"
    say "  docker run --rm memtune python -m memtune reproduce"
    say "  docker run --rm -p 8000:8000 memtune python -m memtune demo --host 0.0.0.0 --no-browser"
    P="docker run --rm memtune python" ;;
esac
say ""
say "Then try:"
say "  $P -m memtune reproduce        recompute the paper's results and compare them with the reference"
if [ "$ENV_KIND" != "docker" ]; then
  say "  $P -m memtune demo             open the interactive demo"
fi
say "  $P -m memtune recommend FILE   recommend mem for your own problems (docs/USAGE.md)"
say "  $P -m pytest                   run all tests"
