#!/usr/bin/env bash
# Dip the fitted tool into the palette's ink caps — scripted, e-stop
# monitored, written to the ink ledger. Keep the workspace clear and a hand
# near the e-stop.
#
#   scripts/il_dip.sh --ee-tool <tool_id> [--dry-run | --yes] [il_dip.py args...]
#
# First hardware run is the BALLPOINT into DRY caps (a rehearsal): the same
# choreography the 3RL will make, with nothing to spill. A real tool is
# refused until the ledger shows that rehearsal happened.
#
# This is autonomous motion, so it is armed like a rollout: one single-use
# nonce in /tmp/tatbot-arm-token (scripts/lib/arm_gate.sh; `tatbot dip
# --nonce <literal>` writes it). When a launcher's --dip runs this as a child,
# the launcher's own consumed nonce is inherited. --dry-run and --connect-only
# command nothing and are not gated.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot dip"
# shellcheck source=scripts/lib/runlog.sh
source "$REPO/scripts/lib/runlog.sh"
# shellcheck source=scripts/lib/estop_guard.sh
source "$REPO/scripts/lib/estop_guard.sh"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require || exit $?
# shellcheck source=scripts/lib/ee_tool.sh
source "$REPO/scripts/lib/ee_tool.sh"
export TATBOT_CONFIG_DIR="${TATBOT_CONFIG_DIR:-$REPO/config/trossen}"
ee_tool::strip "$@"; set -- "${EE_TOOL_ARGS[@]}"
estop_guard::reject_overrides "$@"
ee_tool::require
moves=1
for a in "$@"; do case "$a" in --dry-run|--connect-only) moves=0 ;; esac; done
if [ "$moves" = 1 ]; then
  # shellcheck source=scripts/lib/arm_gate.sh
  source "$REPO/scripts/lib/arm_gate.sh"
  arm_gate::require || exit $?
fi
runlog::init dip --set tool="$EE_TOOL" --set moves="$moves"
runlog::run uv run --project "$REPO/python/lerobot_robot_tatbot" \
  python "$REPO/scripts/il_dip.py" --ee-tool "$EE_TOOL" --run-id "${TATBOT_RUN_ID:-}" "$@"
