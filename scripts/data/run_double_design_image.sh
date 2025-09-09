#!/usr/bin/env bash
set -euo pipefail

# Helper to create a uv virtualenv, install minimal deps, and run the generator.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

# Create venv if missing
if [[ ! -d "$REPO_ROOT/.venv" ]]; then
  uv venv "$REPO_ROOT/.venv"
fi

# Activate venv
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

# Install minimal dependency
uv pip install --quiet pillow

# Run the Python script, forwarding all args
python "$SCRIPT_DIR/double_design_image.py" "$@"

