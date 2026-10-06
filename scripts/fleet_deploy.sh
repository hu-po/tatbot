#!/usr/bin/env bash
# Pushed source -> native builds -> manifested camera services; no arm process.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot deploy"
source "$ROOT/scripts/lib/runlog.sh"
runlog::init deploy
[[ -n "$RUN_DIR" ]] || { echo 'deployment requires a run log' >&2; exit 5; }
runlog::run python3 "$ROOT/scripts/lib/fleet_deploy.py" "$@"
