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
# the profile telemetry endpoint, override TATBOT_TELEMETRY_UDP) as a literal argument,
# which is what scripts/vision/calib_sweep.sh looks for to know it can
# attach. No cameras, no recording beyond the teleop's own .wxtl flight log
# under the resolved log root (teleop/) — `tatbot teleop run` is the full session.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
runlog::run "$TELEOP" "$TATBOT_LEADER_IP" "$TATBOT_FOLLOWER_IP" \
  config/trossen/leader.yaml config/trossen/follower.yaml \
  --estop "$TATBOT_ESTOP_DEVICE" \
  --telemetry-udp "$TELEMETRY" \
  --ee-tool "$EE_TOOL" "${TOOL_ARGS[@]}" \
  "$@"
