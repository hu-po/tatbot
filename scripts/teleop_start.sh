#!/usr/bin/env bash
# Start the bare 400 Hz C++ leader→follower teleop (wxai_teleop) with the
# rig's canonical arguments — the process an operator used to paste an ssh
# line for whenever another workflow (the calibration sweep, a
# camera session) needed the arms live under a human's hands.
#
#   scripts/teleop_start.sh --ee-tool <tool_id> [--touchoff] [extra wxai_teleop args...]
#   tatbot --on <arm-node> --ee-tool <tool_id> teleop start [--touchoff]
#
# --touchoff is the bootstrap for a tool that has no touch-off yet: the teleop
# refuses a tool that config/workspace.yaml was not measured with, but the
# tip phase that would measure it needs this teleop running. It maps to
# wxai_teleop --tool-uncalibrated: grip force from the datasheet, workspace
# constants untouched and announced as the other tool's. Nothing else needs it.
#
# It is INTERACTIVE and stays in the foreground on purpose: the teleop asks
# before the follower moves to meet the leader (Enter), and after an e-stop
# or a fault it holds the arms and asks you to support them before Enter
# releases them to idle. Those prompts are the operator's; nothing here
# backgrounds them. Ctrl+C is the normal way to end it.
#
# Joint telemetry goes to the viewer node's live URDF (--telemetry-udp, default
# the profile telemetry endpoint, override TATBOT_TELEMETRY_UDP) as a literal argument.
# No cameras, no recording beyond the teleop's own .wxtl flight log
# under the resolved log root (teleop/) — `tatbot live cockpit` on the operator
# node shows every camera and the animated URDF beside it.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Physical roles are validated before profiles, tools, probes or driver access.
# The wrist-calibration mode changes control roles inside the executor.
# Controller addresses, profiles and SDK end-effectors stay physically bound.
WRIST_CALIBRATION=0
LEADER_ARM=""
FOLLOWER_ARM=""
ROLE_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --wrist-calibration) WRIST_CALIBRATION=1; ROLE_ARGS+=("$1"); shift ;;
    --leader|--follower)
      [ "$#" -ge 2 ] || { echo "$1 needs a physical arm" >&2; exit 2; }
      [ -n "$2" ] || { echo "$1 needs a physical arm" >&2; exit 2; }
      if [ "$1" = --leader ]; then LEADER_ARM="$2"; else FOLLOWER_ARM="$2"; fi
      shift 2 ;;
    --leader=|--follower=) echo "$1 needs a physical arm" >&2; exit 2 ;;
    --leader=*) LEADER_ARM="${1#--leader=}"; shift ;;
    --follower=*) FOLLOWER_ARM="${1#--follower=}"; shift ;;
    *) ROLE_ARGS+=("$1"); shift ;;
  esac
done
if [ "$WRIST_CALIBRATION" -eq 1 ]; then
  LEADER_ARM="${LEADER_ARM:-right}"
  FOLLOWER_ARM="${FOLLOWER_ARM:-left}"
fi
python3 - "$REPO" "$LEADER_ARM" "$FOLLOWER_ARM" "$WRIST_CALIBRATION" <<'PY'
import sys
from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "scripts/lib"))
from tatbot_cli.arms import require_current_executor, select_roles
try:
    roles = select_roles(sys.argv[2] or None, sys.argv[3] or None)
except ValueError as exc:
    print(f"teleop_start: {exc}", file=sys.stderr)
    sys.exit(2)
try:
    require_current_executor(repo, roles, wrist_calibration=sys.argv[4] == "1")
except ValueError as exc:
    print(f"teleop_start: {exc}", file=sys.stderr)
    sys.exit(3)
PY
set -- "${ROLE_ARGS[@]}"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot teleop start"
# shellcheck source=scripts/lib/estop_guard.sh
source "$REPO/scripts/lib/estop_guard.sh"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require || exit $?
# shellcheck source=scripts/lib/paths.sh
source "$REPO/scripts/lib/paths.sh"
export TATBOT_LOG_ROOT="${TATBOT_LOG_ROOT:-$(tatbot_paths::log_root)}"
# shellcheck source=scripts/lib/ee_tool.sh
source "$REPO/scripts/lib/ee_tool.sh"
estop_guard::reject_overrides "$@"
ee_tool::strip "$@"; set -- "${EE_TOOL_ARGS[@]}"
ee_tool::require || exit $?
TOOL_ARGS=()
REST=()
for a in "$@"; do
  case "$a" in
    --touchoff) TOOL_ARGS+=(--tool-uncalibrated) ;;
    *) REST+=("$a") ;;
  esac
done
set -- "${REST[@]}"

TELEOP="$REPO/cpp/teleop/build/wxai_teleop"
# Telemetry endpoint comes from the profile (endpoints.teleop_telemetry_udp,
# exported as TATBOT_TELEMETRY_UDP); empty disables the stream.
TELEMETRY="${TATBOT_TELEMETRY_UDP:-}"
if [ -z "$TELEMETRY" ]; then
  echo "teleop_start: no telemetry endpoint in the profile (endpoints.teleop_telemetry_udp) — joint-telemetry stream DISABLED" >&2
fi
[ -x "$TELEOP" ] || { echo "missing $TELEOP — build: cd $REPO/cpp/teleop && cmake -B build -S . && cmake --build build" >&2; exit 1; }
for ip in "$TATBOT_LEADER_IP" "$TATBOT_FOLLOWER_IP"; do
  ping -c1 -W1 "$ip" >/dev/null 2>&1 || {
    echo "Arm at $ip is not reachable — is it powered on? (arms take ~20 s to boot)" >&2
    exit 5
  }
done
# The arm driver is exclusive: a second teleop would fail at connect, after
# the first had already been disturbed. Say so before touching anything.
if pgrep -f '[w]xai_teleop' >/dev/null; then
  echo "a wxai_teleop is already running (pid $(pgrep -o -f '[w]xai_teleop')) — it is yours; Ctrl+C it there first" >&2
  exit 6
fi

# shellcheck source=scripts/lib/runlog.sh
source "$REPO/scripts/lib/runlog.sh"
runlog::init teleop --set stack=cpp --set "estop=$TATBOT_ESTOP_DEVICE" --set tool="$EE_TOOL" --set telemetry="$TELEMETRY"
cd "$REPO"
# Every teleop session lands both arms inside its own live session at Enter
# (staged -> sleep -> idle): a bare release idled the motors and left the
# operator to `tatbot arm recover` after every reference capture. A caller that
# already names the pose keeps its value.
LAND_ARGS=()
case " $* " in
  *" --staged-positions "*) ;;
  *)
    # shellcheck source=scripts/lib/staged_pose.sh
    source "$REPO/scripts/lib/staged_pose.sh"
    STAGED="$(staged_pose::csv "$REPO")" || exit 1
    LAND_ARGS=(--staged-positions "$STAGED") ;;
esac
RC=0
runlog::run "$TELEOP" "$TATBOT_LEADER_IP" "$TATBOT_FOLLOWER_IP" \
  config/trossen/leader.yaml config/trossen/follower.yaml \
  --estop "$TATBOT_ESTOP_DEVICE" \
  --telemetry-udp "$TELEMETRY" \
  --ee-tool "$EE_TOOL" "${TOOL_ARGS[@]}" "${LAND_ARGS[@]}" \
  "$@" || RC=$?
# Exit 3 from wxai_teleop is its scheduling refusal, raised before either arm
# driver is constructed. It reports the priority and the limit it measured;
# this adds the account of WHY the policy did not reach this session — group
# membership, PAM/SSH login handling, an inherited limit — from the same
# read-only model `tatbot teleop check` renders. It reads files and changes
# nothing, and it never turns a refusal into a success.
if [ "$RC" = 3 ] && [ -f "$REPO/scripts/lib/teleop_readiness.py" ]; then
  python3 "$REPO/scripts/lib/teleop_readiness.py" --explain-realtime || true
fi
exit "$RC"
