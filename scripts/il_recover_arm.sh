#!/usr/bin/env bash
# Recover an arm after a controller fault (velocity trip, mode mismatch,
# dead session): clears the error, takes position control where the arm
# stands, then moves slowly staged -> sleep -> idle. Keep the workspace clear.
#
#   il_recover_arm.sh [ip] [leader|follower]
#
# Runs cpp/teleop/arm_recover, the landing routine on the vendor C++ SDK: no
# Python environment, the same arm stack as the teleop executor. It performs
# the SAME ritual the LeRobot plugins run on a failed disconnect
# (recovery.land_arm): fresh driver session, golden pushed on a fault or a
# pose outside the controller's limits, an arm joint measured past its limits
# refused with nothing commanded (exit 7), carriage held through the staged
# sweep then returned to its configured rest at sleep, retries, and
# verification that the arm and carriage actually reached that pose.
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

# The binary is built by `scripts/check cpp` and staged on the arm node by
# `tatbot deploy`. Bench tests substitute a stand-in here; the
# e-stop device and the staged pose below are never overridable.
ARM_RECOVER="${TATBOT_ARM_RECOVER_BIN:-$REPO/cpp/teleop/build/arm_recover}"
[ -x "$ARM_RECOVER" ] || {
  echo "missing $ARM_RECOVER — build: cmake -B $REPO/cpp/teleop/build -S $REPO/cpp/teleop && cmake --build $REPO/cpp/teleop/build --target arm_recover" >&2
  exit 1
}
# The SAME staged pose the plugins and the executor use — read from
# tatbot.yaml with the stdlib parser, never copied: a literal here drifted
# silently until 2026-08-30 (there were three copies).
STAGED="$(python3 -c '
import sys
sys.path.insert(0, sys.argv[1])
import tool_spec
print(",".join(repr(v) for v in tool_spec.staged_positions(sys.argv[2])))
' "$REPO/scripts/lib" "$REPO")" || { echo "il_recover_arm: cannot read the staged pose from config/trossen/tatbot.yaml" >&2; exit 1; }
GOLDEN="$TATBOT_CONFIG_DIR/$ROLE.yaml"

# shellcheck source=scripts/lib/runlog.sh
source "$REPO/scripts/lib/runlog.sh"
runlog::init arm-recover --set "role=$ROLE"
command -v timeout >/dev/null || { echo "Recovery refused: GNU timeout is required." >&2; exit 3; }
# The SDK's configure timeout covers TCP connect, not every blocking receive.
# Enforce the routine's 45 s budget from outside the process. Foreground
# mode preserves terminal Ctrl+C delivery to the landing's own shield.
RC=0
runlog::run timeout --foreground --signal=TERM --kill-after=2s 45s \
  "$ARM_RECOVER" "$IP" "$ROLE" --staged "$STAGED" --golden "$GOLDEN" \
  --estop "$TATBOT_ESTOP_DEVICE" || RC=$?
if [ "$RC" = 124 ] || [ "$RC" = 137 ]; then
  echo "Recovery timed out: the driver was terminated; arm state is UNKNOWN." >&2
  echo "No landing or release was verified. Support the arms before any controller power cycle." >&2
elif [ "$RC" = 3 ] || [ "$RC" = 5 ] || [ "$RC" = 6 ] || [ "$RC" = 7 ]; then
  : # nothing was commanded: e-stop engaged (3), controller never answered (5), another driver owns the arms (6),
  # an arm joint measured past its limits (7; arm_recover said how to free it)
elif [ "$RC" != 0 ]; then
  echo "Recovery failed or was interrupted; arm state is UNKNOWN. No automatic retry by this launcher." >&2
fi
exit "$RC"
