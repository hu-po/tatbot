#!/usr/bin/env bash
# Logged entry point for the offline art studio (python/tatbot_art).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/lib/runlog.sh"
runlog::init art --set "command=${1:-help}"
runlog::run "$REPO/scripts/lib/drawing_python.sh" "$REPO/scripts/art.py" "$@"
