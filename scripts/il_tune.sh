#!/usr/bin/env bash
# Standalone teleop tuning session with the web cockpit.
#
#   scripts/il_tune.sh [--leader-only] [extra tune args...]
#
# Runs the minimal leader→follower mirroring loop (same semantics as
# recording, no cameras/dataset) and serves the tuning cockpit on
# http://<this-host>:8899/ — tune leader feel, follower tracking, grip law
# and the safety envelope live, then "Save to golden" to persist into
# config/trossen/{leader,follower,tatbot}.yaml. Ctrl+C ends the session
# (staged pose, then sleep). See docs/teleop_tuning.md.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot teleop tune"
PROJECT="$REPO/python/lerobot_robot_tatbot"
# shellcheck source=scripts/lib/estop_guard.sh
source "$REPO/scripts/lib/estop_guard.sh"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require || exit $?
estop_guard::reject_overrides "$@"
# shellcheck source=scripts/lib/runlog.sh
source "$REPO/scripts/lib/runlog.sh"
runlog::init tune --set "estop=$TATBOT_ESTOP_DEVICE"

export TATBOT_CONFIG_DIR="${TATBOT_CONFIG_DIR:-$REPO/config/trossen}"

runlog::run uv run --project "$PROJECT" python -m lerobot_robot_tatbot.tune "$@"
