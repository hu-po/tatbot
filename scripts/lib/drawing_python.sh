#!/usr/bin/env bash
# Resolve dependencies before starting work; no uv supervisor remains in the signal path.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if ! command -v uv >/dev/null 2>&1; then
  echo "drawing: uv is required to provision the pinned Python environment" >&2
  exit 3
fi
drawing_python=$(env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT -u UV_NO_SYNC \
  uv run --no-config --no-project --managed-python --locked --script "$repo/scripts/lib/drawing_python.py")
exec "$drawing_python" "$@"
