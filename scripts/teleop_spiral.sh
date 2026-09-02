#!/usr/bin/env bash
# One-shot expanding-spiral capability probe. The operator uses normal C++
# leader/follower teleop to place the fitted tip at the spiral center; tapping
# SPACE after READY transfers control to a fully preflighted joint trajectory.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot teleop spiral"
# shellcheck source=scripts/lib/estop_guard.sh
source "$REPO/scripts/lib/estop_guard.sh"
# shellcheck source=scripts/lib/arm_gate.sh
source "$REPO/scripts/lib/arm_gate.sh"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require || exit $?
# shellcheck source=scripts/lib/ee_tool.sh
source "$REPO/scripts/lib/ee_tool.sh"

estop_guard::reject_overrides "$@"
ee_tool::strip "$@"; set -- "${EE_TOOL_ARGS[@]}"
ee_tool::require || exit $?

RADIUS_MM=6
TURNS=3
DURATION_S=180
EASE_S=2
CARRIAGE_IK=0
REST=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --radius-mm)
      [ "$#" -ge 2 ] || { echo "--radius-mm needs a value" >&2; exit 2; }
      RADIUS_MM="$2"; shift 2 ;;
    --turns)
      [ "$#" -ge 2 ] || { echo "--turns needs a value" >&2; exit 2; }
      TURNS="$2"; shift 2 ;;
    --duration-s)
      [ "$#" -ge 2 ] || { echo "--duration-s needs a value" >&2; exit 2; }
      DURATION_S="$2"; shift 2 ;;
    --ease-s)
      [ "$#" -ge 2 ] || { echo "--ease-s needs a value" >&2; exit 2; }
      EASE_S="$2"; shift 2 ;;
    --carriage-ik)
      CARRIAGE_IK=1; shift ;;
    --)
      shift; REST+=("$@"); break ;;
    *)
      REST+=("$1"); shift ;;
  esac
done

if [ "$CARRIAGE_IK" -eq 1 ]; then
  if [ "$EE_TOOL" != "lutin-ballpoint-dot" ]; then
    echo "--carriage-ik is qualified only for --ee-tool lutin-ballpoint-dot" >&2
    exit 2
  fi
  echo "CARRIAGE-IK A/B CANDIDATE: keep the pen off the paper during the automatic"
  echo "2.0 -> 1.5 -> 2.5 -> 2.0 mm carriage preflight. Hand-guide to contact only"
  echo "after PASS is printed. During the spiral the carriage will move inside 0.5..3.5 mm."
  export TATBOT_CARRIAGE_IK_ARMED=1
  CARRIAGE_IK_ARGS=(--spiral-carriage-ik)
else
  CARRIAGE_IK_ARGS=()
fi

arm_gate::require || exit $?
export TATBOT_SPIRAL_ARMED=1
LAND_SENTINEL="/tmp/tatbot-probe-land-$$"
export TATBOT_PROBE_LAND_SENTINEL="$LAND_SENTINEL"
cleanup_land_sentinel() { rm -f "$LAND_SENTINEL"; }
trap cleanup_land_sentinel EXIT

set +e
"$REPO/scripts/teleop_start.sh" \
  --ee-tool "$EE_TOOL" \
  --spiral-radius-mm "$RADIUS_MM" \
  --spiral-turns "$TURNS" \
  --spiral-duration-s "$DURATION_S" \
  --spiral-ease-s "$EASE_S" \
  "${CARRIAGE_IK_ARGS[@]}" \
  --ff-gain 0 \
  "${REST[@]}"
PROBE_RC=$?
set -e

if [ "$PROBE_RC" -eq 130 ]; then
  echo "Spiral probe ended by emergency release; automatic landing is SKIPPED." >&2
  exit "$PROBE_RC"
fi
if [ ! -f "$LAND_SENTINEL" ]; then
  echo "Spiral probe did not record an operator release; automatic landing is SKIPPED." >&2
  exit "$PROBE_RC"
fi

echo "Spiral probe released: landing follower, then leader, to staged -> sleep -> idle."
echo "Keep both landing paths clear; E-stop operator stay ready."
LANDING_FAILED=0
"$REPO/scripts/il_recover_arm.sh" "$TATBOT_FOLLOWER_IP" follower || LANDING_FAILED=1
"$REPO/scripts/il_recover_arm.sh" "$TATBOT_LEADER_IP" leader || LANDING_FAILED=1
if [ "$LANDING_FAILED" -ne 0 ]; then
  echo "Automatic landing did not verify for both arms; inspect the messages above." >&2
  exit 1
fi
exit "$PROBE_RC"
