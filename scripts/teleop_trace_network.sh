#!/usr/bin/env bash
# Capture only controller network traffic; no SDK, serial reader or arm commands.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot teleop trace-network"
# shellcheck source=scripts/lib/runlog.sh
source "$REPO/scripts/lib/runlog.sh"
runlog::init teleop-network
runlog::run python3 "$REPO/scripts/lib/teleop_network_trace.py" "$@"
