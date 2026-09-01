#!/usr/bin/env bash
# Recover an arm after a controller fault (velocity trip, mode mismatch,
# dead session): clears the error, takes position control where the arm
# stands, then moves slowly staged -> sleep -> idle. Keep the workspace clear.
#
#   il_recover_arm.sh [ip] [leader|follower]
#
# Uses the SAME landing routine the plugins use on a failed disconnect
# (recovery.land_arm), so this script, lerobot-record, lerobot-teleoperate
# and il_tune.sh all recover identically: fresh driver session, carriage held
# where it is (the pen is not ridden along its axis), retries, and
# verification that the arm actually reached the sleep pose.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot arm recover"
export TATBOT_CONFIG_DIR="${TATBOT_CONFIG_DIR:-$REPO/config/trossen}"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require || exit $?
IP="${1:-$TATBOT_FOLLOWER_IP}"
ROLE="${2:-follower}"

exec uv run --project "$REPO/python/lerobot_robot_tatbot" python - "$IP" "$ROLE" <<'EOF'
import logging
import sys

import trossen_arm
from lerobot_robot_tatbot import recovery
from lerobot_robot_tatbot.estop import acquire_estop, release_estop

logging.basicConfig(level=logging.INFO, format="%(message)s")
ip, role = sys.argv[1], sys.argv[2]
end_effector = (
    trossen_arm.StandardEndEffector.wxai_v0_leader if role == "leader"
    else trossen_arm.StandardEndEffector.wxai_v0_follower
)
# The SAME staged pose the plugins use — read from tatbot.yaml, not copied:
# a literal here drifted silently until 2026-08-30 (there were three copies).
from lerobot_robot_tatbot import goldens  # noqa: E402

staged = [float(v) for v in goldens.load_tatbot_yaml()["follower"]["staged_positions"]]
assert len(staged) == 7, staged
import os  # noqa: E402

estop = acquire_estop(os.environ["TATBOT_ESTOP_DEVICE"], required=True)
estop.wait_for_initial_state()
try:
    ok = recovery.land_arm(
        ip, end_effector, staged, name=f"{role}@{ip}", estop=estop
    )
finally:
    release_estop(estop)
sys.exit(0 if ok else 1)
EOF
