#!/usr/bin/env bash
# LeRobot teleoperation sanity check with the tatbot plugin (no recording).
# Leader = left arm, follower = right arm, both wrist RealSenses displayed.
# Extra args pass through to lerobot-teleoperate.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot teleop lerobot"
# shellcheck source=scripts/lib/estop_guard.sh
source "$REPO/scripts/lib/estop_guard.sh"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require || exit $?
# Wrist depth cameras by ROLE from the visiond sensor registry.
WRIST_CAMERAS="$(python3 - "$REPO" <<'PY'
import sys, tomllib
from pathlib import Path
reg = tomllib.loads((Path(sys.argv[1]) / "rust/visiond/config/vision.toml").read_text())
by_role = {c["role"]: c["serial"] for c in reg.get("cameras", {}).get("realsense", []) if c.get("role")}
missing = [r for r in ("wrist_upper", "wrist_lower") if r not in by_role]
if missing:
    sys.exit(f"sensor registry has no role= for {', '.join(missing)}")
print("{" + ", ".join(
    f"{r}: {{type: intelrealsense, serial_number_or_name: '{by_role[r]}', "
    "width: 640, height: 480, fps: 30}}" for r in ("wrist_upper", "wrist_lower")) + "}")
PY
)"

# shellcheck source=scripts/lib/ee_tool.sh
source "$REPO/scripts/lib/ee_tool.sh"
estop_guard::reject_overrides "$@"
# Strip --ee-tool before the positionals; the rest passes through untouched.
ee_tool::strip "$@"; set -- "${EE_TOOL_ARGS[@]}"
# shellcheck source=scripts/lib/runlog.sh
source "$REPO/scripts/lib/runlog.sh"
runlog::init teleop --set stack=lerobot --set "estop=$TATBOT_ESTOP_DEVICE"
export TATBOT_CONFIG_DIR="${TATBOT_CONFIG_DIR:-$REPO/config/trossen}"

# Over SSH, show the Rerun panel on this machine's own screen (the operator
# stands next to it); recording control keys still come from this terminal.
if [ -z "${DISPLAY:-}" ] && [ -e "/run/user/$(id -u)/gdm/Xauthority" ]; then
  XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
  export DISPLAY=:0 XAUTHORITY
fi


ee_tool::require || exit $?

# Fail fast and friendly if the arms are not powered on.
for ip in "$TATBOT_LEADER_IP" "$TATBOT_FOLLOWER_IP"; do
  ping -c1 -W1 "$ip" >/dev/null 2>&1 || {
    echo "Arm at $ip is not reachable — is it powered on? (arms take ~20 s to boot)" >&2
    exit 1
  }
done

# The RealSenses are exclusive: whoever opens them first owns them. Set
# TATBOT_TELEOP_CAMERAS={} to teleop without them, freeing the cameras for
# another tool (e.g. scripts/depth_probe.py) while the arms stay compliant.
if [ -n "${TATBOT_TELEOP_CAMERAS+set}" ]; then
  CAMERAS="$TATBOT_TELEOP_CAMERAS"
else
  CAMERAS="$WRIST_CAMERAS"
fi

runlog::run uv run --project "$REPO/python/lerobot_robot_tatbot" lerobot-teleoperate \
  --robot.type=tatbot_follower \
  --robot.ee_tool="$EE_TOOL" \
  --robot.ip_address="$TATBOT_FOLLOWER_IP" \
  --robot.id=tatbot_follower_right \
  --robot.estop_required=true \
  --robot.cameras="$CAMERAS" \
  --teleop.type=tatbot_leader_teleop \
  --teleop.ip_address="$TATBOT_LEADER_IP" \
  --teleop.id=tatbot_leader_left \
  --teleop.estop_required=true \
  --display_data=true \
  "$@"
