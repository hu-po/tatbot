#!/usr/bin/env bash
# Logged entry point; physical execution remains tatbot ros draw.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/lib/runlog.sh"
runlog::init research --set "command=${1:-help}"
runlog::run "$REPO/scripts/lib/drawing_python.sh" "$REPO/scripts/research.py" "$@"
