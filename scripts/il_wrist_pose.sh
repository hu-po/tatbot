#!/usr/bin/env bash
# Attended parked-wrist camera positioning through the calibration worker.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"
cli_hint::note "tatbot calib pose"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require
# shellcheck source=scripts/lib/estop_guard.sh
source "$REPO/scripts/lib/estop_guard.sh"
estop_guard::reject_overrides "$@"
# shellcheck source=scripts/lib/arm_gate.sh
source "$REPO/scripts/lib/arm_gate.sh"
arm_gate::require
exec python3 "$REPO/scripts/vision/wrist_pose.py" "$@"
