#!/usr/bin/env bash
# Logged entry point for the lab (python/tatbot_lab): studies, trials, plans and cell readings.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/lib/runlog.sh"
runlog::init lab --set "command=${1:-help}"
runlog::run "$REPO/scripts/lib/drawing_python.sh" "$REPO/scripts/lab.py" "$@"
