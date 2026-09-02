#!/usr/bin/env bash
# Run a trained policy on the tatbot follower arm (leader arm not used).
#
#   scripts/il_rollout.sh policy_dir [duration_s] [extra lerobot-rollout args...]
#
# This legacy absolute-policy path requires an explicit checkpoint; the GR00T
# flagship uses il_rollout_async.sh. Duration defaults to 60 s with no
# recording (--strategy.type=base). The robot runs through the SAME
# tatbot_follower class used for recording (bounded grip, staged homing on
# connect, staged->sleep on exit). Ctrl+C stops the rollout; keep the
# workspace clear and a hand near the e-stop.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot rollout sync"
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
unset DISPLAY

# shellcheck source=scripts/lib/dip_hook.sh
source "$REPO/scripts/lib/dip_hook.sh"
# Strip --ee-tool, --dip and --no-ink before the positionals; the rest passes through untouched.
ee_tool::strip "$@"; set -- "${EE_TOOL_ARGS[@]}"
dip_hook::strip "$@" || exit $?; set -- "${DIP_HOOK_ARGS[@]}"

[ $# -ge 1 ] || {
  echo "usage: scripts/il_rollout.sh policy_dir [duration_s] [extra args...]" >&2
  echo "GR00T flagship checkpoints use scripts/il_rollout_async.sh" >&2
  exit 2
}
POLICY="$1"
DURATION="${2:-60}"
shift $(( $# < 2 ? $# : 2 ))
estop_guard::reject_overrides "$@"
for arg in "$@"; do
  case "$arg" in
    --policy.path|--policy.path=*|--robot.include_external_effort|\
    --robot.include_external_effort=*|--robot.mask_external_effort|\
    --robot.mask_external_effort=*|--robot.depth_policy_encoding|\
    --robot.depth_policy_encoding=*|--robot.cameras|--robot.cameras=*)
      echo "checkpoint-controlled rollout option cannot be overridden: $arg" >&2
      exit 2
      ;;
  esac
done

# GR00T relative checkpoints explicitly reject select_action(): cached delta
# rows cannot be decoded against newer observations. This synchronous launcher
# uses that single-action path, so fail before the operator gate or any hardware
# check and route the checkpoint to the full-chunk async stack.
POLICY_CONFIG="$POLICY/config.json"
[ -f "$POLICY_CONFIG" ] || POLICY_CONFIG="$POLICY/train_config.json"
[ -f "$POLICY_CONFIG" ] || {
  echo "checkpoint has no config.json or train_config.json: $POLICY" >&2
  exit 2
}
IFS='|' read -r POLICY_TYPE USE_DEPTH_INT STATE_SIZE DEPTH_ENCODING USE_RELATIVE MASK_EXT_EFF < <(
  python3 "$REPO/scripts/eval/checkpoint_contract.py" --format fields "$POLICY_CONFIG"
)
if [ "$POLICY_TYPE" = groot ] && [ "$USE_RELATIVE" = 1 ]; then
  echo "relative-action GR00T requires full-chunk async inference; use:" >&2
  echo "  scripts/il_rollout_async.sh <duration> <server-policy-path> groot" >&2
  exit 2
fi
[ "$STATE_SIZE" = 7 ] || [ "$STATE_SIZE" = 14 ] || {
  echo "unsupported checkpoint state width $STATE_SIZE" >&2
  exit 2
}
if [ "$MASK_EXT_EFF" = 1 ] && [ "$STATE_SIZE" != 14 ]; then
  echo "masked external effort requires a 14-wide checkpoint state" >&2
  exit 2
fi
USE_DEPTH=$([ "$USE_DEPTH_INT" = 1 ] && echo true || echo false)
INCLUDE_EXT_EFF=$([ "$STATE_SIZE" = 14 ] && echo true || echo false)
MASK_EXT_EFF_BOOL=$([ "$MASK_EXT_EFF" = 1 ] && echo true || echo false)

# Wrist depth cameras by ROLE from the visiond sensor registry. The checkpoint
# contract, not the current launcher default, decides whether depth is present.
WRIST_CAMERAS="$(USE_DEPTH="$USE_DEPTH" python3 - "$REPO" <<'PY'
import os, sys, tomllib
from pathlib import Path
reg = tomllib.loads((Path(sys.argv[1]) / "rust/visiond/config/vision.toml").read_text())
by_role = {c["role"]: c["serial"] for c in reg.get("cameras", {}).get("realsense", []) if c.get("role")}
missing = [r for r in ("wrist_upper", "wrist_lower") if r not in by_role]
if missing:
    sys.exit(f"sensor registry has no role= for {', '.join(missing)}")
depth = os.environ["USE_DEPTH"]
print("{" + ", ".join(
    f"{r}: {{type: intelrealsense, serial_number_or_name: '{by_role[r]}', "
    f"width: 640, height: 480, fps: 30, use_depth: {depth}}}"
    for r in ("wrist_upper", "wrist_lower")) + "}")
PY
)"
# Tool identity is a hardware concern: it gates here, with the other ones,
# after the checkpoint contract has had its say.
ee_tool::require || exit $?
# shellcheck source=scripts/lib/arm_gate.sh
source "$REPO/scripts/lib/arm_gate.sh"
arm_gate::require || exit $?

TASK=$(python3 -c "import json;print(json.load(open('$POLICY/train_config.json'))['dataset'].get('single_task') or 'run the demonstrated task')" 2>/dev/null || echo "run the demonstrated task")
echo "policy: $POLICY  duration: ${DURATION}s  task: $TASK"
echo "checkpoint contract: type=$POLICY_TYPE depth=$USE_DEPTH state=$STATE_SIZE external effort=$INCLUDE_EXT_EFF masked=$MASK_EXT_EFF_BOOL"


# Open the run before the preflight, so a rollout that never starts still
# leaves a record of when and why.
runlog::init rollout \
  --set policy="$POLICY" --set duration_s="$DURATION" --set task="$TASK" \
  --set inference=sync
# --dip / --no-ink: the ink hook (scripts/lib/dip_hook.sh), after the run is
# open so a dip's ledger events are mirrored into this run's ink.jsonl.
dip_hook::run || exit $?

# Fail fast and friendly if the arms are not powered on.
ping -c1 -W1 "$TATBOT_FOLLOWER_IP" >/dev/null 2>&1 || {
  echo "Arm at $TATBOT_FOLLOWER_IP is not reachable — is it powered on? (arms take ~20 s to boot)" >&2
  exit 1
}

# Upstream rollout drops .eff/.ext_eff observations from the policy state
# (lerobot v0.6.1 TODO); apply/verify the one-line fix (idempotent).
uv run --project "$REPO/python/lerobot_robot_tatbot" python "$REPO/scripts/il_patch_lerobot.py"

# il_client_shield.py: first Ctrl+C starts the graceful landing, repeats are
# swallowed so they cannot abort the staged->sleep moves mid-motion.
# Refuse to start while another client still holds the arm. On 2026-08-21 a
# rollout blocked in the driver connect behind an orphan and then silently
# started driving the arm when that orphan was finally cleared. Bracketed
# pattern so this does not match its own launching shell (see 189db8a).
if pgrep -f "[i]l_client_shield.py" >/dev/null 2>&1; then
  echo "another rollout client is already running — refusing to start:" >&2
  pgrep -af "[i]l_client_shield.py" >&2
  echo "stop it (kill -INT <pid>); scripts/il_recover_arm.sh if the arm is stranded" >&2
  exit 1
fi

# runlog::run, not exec: this shell outlives the rollout so it can finalize the
# run and analyze the flight log. Unlike the async path this keeps the client
# in the FOREGROUND, so a terminal Ctrl+C reaches it directly through the
# process group and il_client_shield.py owns the landing. lerobot bounds the
# run itself via --duration, so there is no timeout wrapper here — which is
# what spared this script the orphaning incident the async path had.
STATUS=0
runlog::run env --chdir="${RUN_DIR:-$PWD}" uv run --project "$REPO/python/lerobot_robot_tatbot" python "$REPO/scripts/il_client_shield.py" lerobot.scripts.lerobot_rollout:main \
  --strategy.type=base \
  --policy.path="$POLICY" \
  --robot.type=tatbot_follower \
  --robot.ee_tool="$EE_TOOL" \
  --robot.ip_address="$TATBOT_FOLLOWER_IP" \
  --robot.id=tatbot_follower_right \
  --robot.estop_required=true \
  --robot.cameras="$WRIST_CAMERAS" \
  --robot.include_external_effort=$INCLUDE_EXT_EFF \
  --robot.mask_external_effort=$MASK_EXT_EFF_BOOL \
  --robot.depth_policy_encoding="$DEPTH_ENCODING" \
  --task="$TASK" \
  --duration="$DURATION" \
  --inference.type=sync \
  --display_data=false \
  "$@" || STATUS=$?

# A report, never a gate: bounded, non-fatal, cannot change the exit code.
if [ "${TATBOT_ANALYZE:-1}" != 0 ] && [ -n "${RUN_DIR:-}" ]; then
  timeout 60 uv run --project "$REPO/python/lerobot_robot_tatbot" python \
    "$REPO/scripts/il_analyze_rollout.py" "$RUN_DIR" \
    || echo "analysis failed (non-fatal) — rerun: scripts/il_analyze_rollout.py $RUN_DIR" >&2
fi

# Nothing of ours may outlive this script while still holding the arm.
if pgrep -f "[i]l_client_shield.py" >/dev/null 2>&1; then
  echo "WARNING: a rollout client is STILL RUNNING after this script finished:" >&2
  pgrep -af "[i]l_client_shield.py" >&2
  echo "  stop it with: kill -INT <pid>" >&2
fi

exit "$STATUS"
