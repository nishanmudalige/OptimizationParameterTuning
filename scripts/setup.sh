#!/usr/bin/env bash
# Kept for older instructions: the installer is now install.sh in the repository root.
#
#   ./scripts/setup.sh                # same as  ./install.sh --yes --env venv
#   ./scripts/setup.sh --notebooks    # same as  ./install.sh --yes --env venv --with notebooks
set -euo pipefail
exec bash "$(dirname "$0")/../install.sh" --yes --env venv "$@"
