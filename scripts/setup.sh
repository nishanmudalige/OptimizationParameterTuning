#!/usr/bin/env bash
# One-command setup for Linux and macOS.
#
#   ./scripts/setup.sh                # core install (python, tests, CLI)
#   ./scripts/setup.sh --notebooks    # also JupyterLab and the notebook tools
#
# Finds a supported Python (3.10 - 3.12) on its own, creates .venv in the repository, installs the pinned
# packages, and checks the result. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_NOTEBOOKS=0
for arg in "$@"; do
  case "$arg" in
    --notebooks) WITH_NOTEBOOKS=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (use --notebooks or --help)"; exit 2 ;;
  esac
done

find_python() {
  for c in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1 &&
       "$c" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

if ! PY=$(find_python); then
  cat <<'EOF'
No supported Python found. memtune needs Python 3.10, 3.11 or 3.12 (3.13+ is not supported by the pinned numpy).
Install one, then run this script again:
  macOS:           brew install python@3.12
  Ubuntu / Debian: sudo apt install python3 python3-venv python3-pip      (Ubuntu 24.04 ships Python 3.12)
  Fedora:          sudo dnf install python3.12
  any system:      conda create -n memtune python=3.12     (then use conda instead of this script)
EOF
  exit 1
fi
echo "Using $($PY --version) at $(command -v "$PY")"

if [ ! -x .venv/bin/python ]; then
  if ! "$PY" -m venv .venv; then
    echo "Could not create the virtual environment."
    echo "On Debian / Ubuntu run:  sudo apt install python3-venv python3-pip   and try again."
    rm -rf .venv
    exit 1
  fi
fi
# shellcheck disable=SC1091
. .venv/bin/activate

REQ=requirements-lock.txt
[ "$WITH_NOTEBOOKS" = 1 ] && REQ=requirements-notebooks.txt
echo "Installing $REQ (about a minute) ..."
python -m pip install -q --upgrade pip
python -m pip install -q -r "$REQ"
python -m pip install -q -e . --no-deps

echo
python -m memtune info
echo
echo "Smoke test: recomputing the paper's results ..."
python -m memtune evaluate --out "$(mktemp -d)" | sed -n '4,9p'

cat <<'EOF'

Setup finished. Next time, activate the environment first:   source .venv/bin/activate
Then try:   python -m memtune evaluate          python -m pytest          python -m memtune --help
EOF
