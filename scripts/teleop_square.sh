#!/usr/bin/env bash
# One-shot follower Cartesian capability probe. The operator uses the normal
# C++ leader/follower teleop to place the fitted tip at light paper contact;
# tapping SPACE after the READY line transfers control to a slow 6 mm square and
# permanently disables resume for that process.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot teleop square"
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

SIZE_MM=6
EDGE_S=12
REST=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --size-mm)
      [ "$#" -ge 2 ] || { echo "--size-mm needs a value" >&2; exit 2; }
      SIZE_MM="$2"; shift 2 ;;
    --edge-s)
      [ "$#" -ge 2 ] || { echo "--edge-s needs a value" >&2; exit 2; }
      EDGE_S="$2"; shift 2 ;;
    --)
      shift; REST+=("$@"); break ;;
    *)
      REST+=("$1"); shift ;;
  esac
done

# The CLI writes the literal token immediately before launching this wrapper.
# Consume it here, before teleop_start can connect either driver.
arm_gate::require || exit $?
export TATBOT_SQUARE_ARMED=1
LAND_SENTINEL="/tmp/tatbot-probe-land-$$"
export TATBOT_PROBE_LAND_SENTINEL="$LAND_SENTINEL"
cleanup_land_sentinel() { rm -f "$LAND_SENTINEL"; }
trap cleanup_land_sentinel EXIT

set +e
"$REPO/scripts/teleop_start.sh" \
  --ee-tool "$EE_TOOL" \
  --square-probe-mm "$SIZE_MM" \
  --square-edge-s "$EDGE_S" \
  --ff-gain 0 \
  "${REST[@]}"
PROBE_RC=$?
set -e

if [ "$PROBE_RC" -eq 130 ]; then
  echo "Square probe ended by emergency release; automatic landing is SKIPPED." >&2
  exit "$PROBE_RC"
fi
if [ ! -f "$LAND_SENTINEL" ]; then
  echo "Square probe did not record an operator release; automatic landing is SKIPPED." >&2
  exit "$PROBE_RC"
fi

echo "Square probe released: landing follower, then leader, to staged -> sleep -> idle."
echo "Keep both landing paths clear; E-stop operator stay ready."
LANDING_FAILED=0
"$REPO/scripts/il_recover_arm.sh" "$TATBOT_FOLLOWER_IP" follower || LANDING_FAILED=1
"$REPO/scripts/il_recover_arm.sh" "$TATBOT_LEADER_IP" leader || LANDING_FAILED=1
if [ "$LANDING_FAILED" -ne 0 ]; then
  echo "Automatic landing did not verify for both arms; inspect the messages above." >&2
  exit 1
fi
exit "$PROBE_RC"
