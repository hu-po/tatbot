#!/usr/bin/env bash
# Run a trained policy on the follower arm with GPU inference on the serve node.
#
#   scripts/il_rollout_async.sh [duration] [server_policy_dir] [policy_type] [extra robot_client args...]
#
# policy_type is derived from the checkpoint; an explicitly supplied value
# must match. Architectures get their own chunk sizing (bounded by each
# policy's n_action_steps; evo1 needs ~1.1 s per
# chunk on the GB10, so it refills earlier to avoid queue starvation).
# The task string matters for language-conditioned policies (multi_task_dit,
# evo1) and is ignored by act — override with TATBOT_TASK="...".
#
# Requires a checkpoint-pinned policy server on the node with the `serve`
# role (config/nodes.json):
#   tatbot --on <serve-node> serve start --policy <server checkpoint>
#
# The client streams observations to the server and receives action chunks;
# overlapping chunks are blended (weighted average), which removes the
# chunk-boundary jolts of local CPU inference. The robot runs through the
# same tatbot_follower class as recording. Ctrl+C stops the client (the
# robot is then disconnected and returns staged -> sleep).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot rollout run"
# shellcheck source=scripts/lib/runlog.sh
source "$REPO/scripts/lib/runlog.sh"
# shellcheck source=scripts/lib/estop_guard.sh
source "$REPO/scripts/lib/estop_guard.sh"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require || exit $?
# shellcheck source=scripts/lib/nodes.sh
source "$REPO/scripts/lib/nodes.sh"
# shellcheck source=scripts/lib/ee_tool.sh
source "$REPO/scripts/lib/ee_tool.sh"
# shellcheck source=scripts/lib/dip_hook.sh
source "$REPO/scripts/lib/dip_hook.sh"
# Strip --ee-tool, --dip and --no-ink before the positionals; the rest passes through untouched.
ee_tool::strip "$@"; set -- "${EE_TOOL_ARGS[@]}"
dip_hook::strip "$@" || exit $?; set -- "${DIP_HOOK_ARGS[@]}"

# shellcheck source=scripts/il_audio_record.sh
source "$REPO/scripts/il_audio_record.sh"
export TATBOT_CONFIG_DIR="${TATBOT_CONFIG_DIR:-$REPO/config/trossen}"

# The venv interpreter, used for everything below. Deliberately NOT `uv run`:
# uv has to be on PATH, which it is not under cron, systemd, ssh without a
# login shell, or an agent — and more importantly it forks a process that then
# sits between this shell and the client, which is what let a rollout outlive
# its supervisor and drive the arm unattended on 2026-08-21.
VENV_PY="$REPO/python/lerobot_robot_tatbot/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "no interpreter at $VENV_PY" >&2
  echo "run: uv sync --project $REPO/python/lerobot_robot_tatbot" >&2
  exit 1
fi
unset DISPLAY

DURATION="${1:-10}"
# Path AS SEEN BY THE SERVER. The stable flagship link is
# installed only after a checkpoint passes wire and robot evaluation.
POLICY="${2:-${TATBOT_SERVE_ROOT:-$HOME/il-serve}/models/flagship}"
# Must match the policy dir: act, multi_task_dit, evo1, ... (see docs/imitation_learning.md)
REQUESTED_POLICY_TYPE="${3:-${TATBOT_POLICY_TYPE:-}}"
shift $(( $# < 3 ? $# : 3 ))
estop_guard::reject_overrides "$@"
for arg in "$@"; do
  case "$arg" in
    --policy_type|--policy_type=*|--pretrained_name_or_path|--pretrained_name_or_path=*|\
    --robot.include_external_effort|--robot.include_external_effort=*|\
    --robot.mask_external_effort|--robot.mask_external_effort=*|\
    --robot.depth_policy_encoding|--robot.depth_policy_encoding=*|\
    --robot.cameras|--robot.cameras=*)
      echo "checkpoint-controlled rollout option cannot be overridden: $arg" >&2
      exit 2
      ;;
  esac
done

# Default: the serve node from config/nodes.json (empty -> must be stated).
SERVER="${TATBOT_POLICY_SERVER:-$(tatbot_nodes::target serve | sed "s/.*@//"):8080}"
SSH_TARGET="${TATBOT_POLICY_SSH:-${TATBOT_POLICY_USER:-$(tatbot_nodes::target serve | sed "s/@.*//")}@${SERVER%%:*}}"
TASK="${TATBOT_TASK:-draw a continuous squiggle using pen tip on the grid lines of the paper pad.}"

# Observation wiring must match what the CHECKPOINT was trained on, and the
# server fails (loudly, arm never leaves staged) when it does not:
# - TATBOT_DEPTH=1 streams the wrist D405 depth planes. RGBD policies
#   (squiggle_h64_*_rgbd, squiggle_approach_*) require it; RGB-only policies
#   must NOT get it — the server indexes the policy's features by every
#   camera key the robot sends, so an unexpected depth key is a KeyError.
# - TATBOT_EXT_EFF=0 drops external effort from the wire so observation.state
#   is the 7 joint positions. The squiggle-era policies are 7-state; the
#   draw-square era trained on 14 (pos+ext_eff), hence the default stays 1.
# - A checkpoint with mask_external_effort keeps the 14-wide state but zeros
#   all seven .ext_eff values. The safety watchdog still reads measured force
#   directly from the driver; only the policy observation is masked.
# Neither include/drop nor masking is a safety parameter: the overforce guard
# and e-stop read the driver directly, not the observation dict.
# - TATBOT_OBS_HISTORY=1 (opt-in) bundles the previous sent frame with each
#   observation so the policy's n_obs=2 history is a ~250 ms pair instead of
#   the ~1.6 s stale cross-chunk pair (33 ms in training; this is 6x closer).
#   Untested on-robot as of 2026-08-24 — bench-verified only. Flows to the
#   client via the environment; patches 12/13 implement it.
POLICY_CONFIG_JSON=""
SIDECAR_JSON="{}"
if [ -r "$POLICY/config.json" ]; then
  POLICY_CONFIG_JSON="$(cat "$POLICY/config.json")"
  [ ! -r "$POLICY/tatbot_contract.json" ] || SIDECAR_JSON="$(cat "$POLICY/tatbot_contract.json")"
else
  POLICY_CONFIG_JSON="$(ssh -n -o BatchMode=yes -o ConnectTimeout=3 "$SSH_TARGET" \
    "cat '$POLICY/config.json'" 2>/dev/null)" || {
      echo "cannot read checkpoint config at $SSH_TARGET:$POLICY/config.json" >&2
      exit 2
    }
  SIDECAR_JSON="$(ssh -n -o BatchMode=yes -o ConnectTimeout=3 "$SSH_TARGET" \
    "if [ -f '$POLICY/tatbot_contract.json' ]; then cat '$POLICY/tatbot_contract.json'; else printf '{}'; fi" \
    2>/dev/null)" || {
      echo "cannot read checkpoint sidecar at $SSH_TARGET:$POLICY/tatbot_contract.json" >&2
      exit 2
    }
fi
IFS='|' read -r POLICY_TYPE CONTRACT_DEPTH STATE_SIZE DEPTH_ENCODING _USE_RELATIVE MASK_EXT_EFF < <(
  python3 "$REPO/scripts/eval/checkpoint_contract.py" --format fields \
    --sidecar-json "$SIDECAR_JSON" - <<<"$POLICY_CONFIG_JSON"
)
if [ -n "$REQUESTED_POLICY_TYPE" ] && [ "$REQUESTED_POLICY_TYPE" != "$POLICY_TYPE" ]; then
  echo "requested policy type $REQUESTED_POLICY_TYPE does not match checkpoint type $POLICY_TYPE" >&2
  exit 2
fi
[ "$STATE_SIZE" = 7 ] || [ "$STATE_SIZE" = 14 ] || {
  echo "unsupported checkpoint state width $STATE_SIZE" >&2
  exit 2
}
EXPECTED_EXT_EFF=$([ "$STATE_SIZE" = 14 ] && echo 1 || echo 0)
if [ -n "${TATBOT_DEPTH+x}" ] && [ "$TATBOT_DEPTH" != "$CONTRACT_DEPTH" ]; then
  echo "TATBOT_DEPTH=$TATBOT_DEPTH contradicts checkpoint depth=$CONTRACT_DEPTH" >&2
  exit 2
fi
if [ -n "${TATBOT_DEPTH_ENCODING+x}" ] && [ "$TATBOT_DEPTH_ENCODING" != "$DEPTH_ENCODING" ]; then
  echo "TATBOT_DEPTH_ENCODING=$TATBOT_DEPTH_ENCODING contradicts checkpoint encoding=$DEPTH_ENCODING" >&2
  exit 2
fi
if [ -n "${TATBOT_EXT_EFF+x}" ] && [ "$TATBOT_EXT_EFF" != "$EXPECTED_EXT_EFF" ]; then
  echo "TATBOT_EXT_EFF=$TATBOT_EXT_EFF contradicts checkpoint state width $STATE_SIZE" >&2
  exit 2
fi
if [ -n "${TATBOT_MASK_EXT_EFF+x}" ] && [ "$TATBOT_MASK_EXT_EFF" != "$MASK_EXT_EFF" ]; then
  echo "TATBOT_MASK_EXT_EFF=$TATBOT_MASK_EXT_EFF contradicts checkpoint mask=$MASK_EXT_EFF" >&2
  exit 2
fi
if [ "$MASK_EXT_EFF" = 1 ] && [ "$STATE_SIZE" != 14 ]; then
  echo "masked external effort requires a 14-wide checkpoint state" >&2
  exit 2
fi
USE_DEPTH=$([ "$CONTRACT_DEPTH" = 1 ] && echo true || echo false)
INCLUDE_EXT_EFF=$([ "$EXPECTED_EXT_EFF" = 1 ] && echo true || echo false)
MASK_EXT_EFF_BOOL=$([ "$MASK_EXT_EFF" = 1 ] && echo true || echo false)

# Wrist depth cameras by ROLE from the visiond sensor registry (never file
# order); use_depth follows the checkpoint contract resolved above.
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

# All checkpoint/wire checks precede the tool and single-use motion gates, so
# a malformed policy cannot consume an operator nonce.
ee_tool::require || exit $?
# shellcheck source=scripts/lib/arm_gate.sh
source "$REPO/scripts/lib/arm_gate.sh"
arm_gate::require || exit $?
TARGET_VELOCITY="${TATBOT_POLICY_TARGET_VELOCITY:-0.25}"
CONTROLLER_VELOCITY="${TATBOT_POLICY_CONTROLLER_VELOCITY:-0.75}"

# How many of the served actions to execute, and when to refill. Only the
# FRACTION and the THRESHOLD are judgement calls; the chunk length itself is
# the policy's own n_action_steps and is read from the checkpoint below.
# ACT takes 60% because its late-chunk actions are stale by the time they run.
# INFER_MS is the measured server-side cost used for the budget check.
case "$POLICY_TYPE" in
  act)            FRACTION=0.60; THRESH=0.5; FALLBACK=60;  INFER_MS=14  ;;
  groot)          FRACTION=1.00; THRESH=0.6; FALLBACK=16;  INFER_MS=250 ;;
  multi_task_dit) FRACTION=1.00; THRESH=0.6; FALLBACK=24;  INFER_MS=350 ;;
  evo1)           FRACTION=1.00; THRESH=0.5; FALLBACK=50;  INFER_MS=382 ;;
  *)              FRACTION=1.00; THRESH=0.6; FALLBACK=24;  INFER_MS=350 ;;
esac
INFER_MS="${TATBOT_INFER_MS:-$INFER_MS}"

# The policy path is a SERVER-side path, so this host may not be able to stat
# it. Try hardest-to-softest and record which one won.
read_n_action_steps() {
  python3 -c '
import json,sys
try: c=json.load(sys.stdin)
except Exception: sys.exit(1)
for k in ("n_action_steps","chunk_size","horizon"):
    if isinstance(c.get(k),int): print(c[k]); break
else: sys.exit(1)
' 2>/dev/null
}

SERVED=""; CHUNK_SOURCE=""
if [ -n "${TATBOT_N_ACTION_STEPS:-}" ]; then
  SERVED="$TATBOT_N_ACTION_STEPS"; CHUNK_SOURCE="env"
else
  SERVED="$(read_n_action_steps <<<"$POLICY_CONFIG_JSON")" && CHUNK_SOURCE="checkpoint_config"
fi

if [ -n "$SERVED" ]; then
  CHUNK=$(python3 -c "import sys;s=int(sys.argv[1]);print(max(8,min(s,round(float(sys.argv[2])*s))))" \
          "$SERVED" "$FRACTION")
else
  CHUNK="$FALLBACK"; CHUNK_SOURCE="fallback"
  echo "WARN: could not read n_action_steps from the checkpoint." >&2
  echo "      path:  $POLICY" >&2
  echo "      tried: ${SSH_TARGET:-<no ssh target>} (set TATBOT_POLICY_SSH to change)" >&2
  echo "      using the per-type fallback $CHUNK. If this checkpoint serves a" >&2
  echo "      different horizon, set TATBOT_N_ACTION_STEPS=<n> and re-run." >&2
fi

# Explicit overrides still win over everything resolved above.
[ -n "${TATBOT_CHUNK:-}" ] && { CHUNK="$TATBOT_CHUNK"; CHUNK_SOURCE="env_chunk"; }
THRESH="${TATBOT_THRESH:-$THRESH}"

# Asking for more than the policy serves is actively harmful: the server
# truncates (chunk[:, :actions_per_chunk, :]) while the client keeps dividing
# the queue by the size it ASKED for, so it believes it is starving and uploads
# harder.
if [ -n "$SERVED" ] && [ "$CHUNK" -gt "$SERVED" ]; then
  echo "WARN: requested $CHUNK actions but the policy serves $SERVED — clamping." >&2
  CHUNK="$SERVED"
fi

# Refill budget must clear the server's inference cost or the action queue
# drains and the control loop stalls. Measured 2026-08-21: at CHUNK=24 the
# queue sat empty 23% of the time and the loop ran 25.7 Hz losing 14% of
# ticks; at CHUNK=48 it was empty 3%, ran a clean 30.0 Hz and stalled zero
# times. The client is not the bottleneck (its own work is 1.3 ms median) —
# this budget is. See docs/imitation_learning.md.
BUDGET_MS=$(python3 -c "print(int(float('$THRESH')*int('$CHUNK')/30*1000))")
if [ "$BUDGET_MS" -lt "$((INFER_MS * 3 / 2))" ]; then
  echo "WARN: refill budget ${BUDGET_MS} ms vs ~${INFER_MS} ms inference — the action" >&2
  echo "      queue will drain and the control loop will stall. Raise" >&2
  echo "      TATBOT_THRESH (max 1.0), serve a policy with a longer action" >&2
  echo "      horizon, or use a faster server." >&2
fi
echo "duration: ${DURATION}s  policy(server-side): $POLICY  type: $POLICY_TYPE  server: $SERVER"
echo "task: \"$TASK\"  actions_per_chunk: $CHUNK ($CHUNK_SOURCE)  chunk_size_threshold: $THRESH"
echo "depth: $USE_DEPTH  depth encoding: ${DEPTH_ENCODING:-raw/none}  external effort on the wire: $INCLUDE_EXT_EFF  masked: $MASK_EXT_EFF_BOOL"
echo "obs history bundle: ${TATBOT_OBS_HISTORY:-0}"
echo "commissioning speed: target ${TARGET_VELOCITY} rad/s; controller ${CONTROLLER_VELOCITY} rad/s"
echo "refill budget: ${BUDGET_MS} ms vs ~${INFER_MS} ms server inference"


# Open the run before the preflight, so a run that never starts still leaves
# a record of when and why. Everything from here — the arm check, the patch
# report, the client's output, the landing narration — lands in console.log.
runlog::init rollout_async \
  --set policy="$POLICY" --set policy_type="$POLICY_TYPE" --set server="$SERVER" \
  --set duration_s="$DURATION" --set task="$TASK" --set chunk="$CHUNK" \
  --set chunk_source="$CHUNK_SOURCE" --set n_action_steps="${SERVED:-unknown}" \
  --set thresh="$THRESH" --set refill_budget_ms="$BUDGET_MS" \
  --set infer_ms="$INFER_MS" --set use_depth="$USE_DEPTH" \
  --set depth_encoding="${DEPTH_ENCODING:-none}" --set include_ext_eff="$INCLUDE_EXT_EFF" \
  --set mask_ext_eff="$MASK_EXT_EFF_BOOL"
# --dip / --no-ink: the ink hook (scripts/lib/dip_hook.sh), after the run is
# open so a dip's ledger events are mirrored into this run's ink.jsonl.
dip_hook::run || exit $?

# Fail fast and friendly if the arms are not powered on.
ping -c1 -W1 "$TATBOT_FOLLOWER_IP" >/dev/null 2>&1 || {
  echo "Arm at $TATBOT_FOLLOWER_IP is not reachable — is it powered on? (arms take ~20 s to boot)" >&2
  exit 1
}

"$VENV_PY" "$REPO/scripts/il_patch_lerobot.py"

# The client has no duration flag; SIGINT triggers its clean teardown
# (disconnect -> staged -> sleep), so this script bounds the run by sending one
# itself. il_client_shield.py turns the first SIGINT into the landing and
# swallows repeats until the arm is down, so an operator Ctrl+C and the
# deadline take exactly the same path.
# lerobot's async client writes its own DEBUG log to logs/<name>_<epoch>.log
# RELATIVE TO THE CWD (async_inference/helpers.py does os.makedirs("logs")).
# Run from inside the run directory so that lands with the rest of this run's
# evidence instead of scattering copies through the repo — four had piled up
# in the checkout before anyone noticed, and it is a 2,700-line-per-run record of
# exactly the client-side timing we spent 2026-08-21 reconstructing by hand.
# Every path passed below is absolute, so the cwd is free to move.
CLIENT_CWD="${RUN_DIR:-$PWD}"

# THE SHELL OWNS THE CLIENT'S PID. Nothing sits between them.
#
# This used to be `timeout ... uv run ... python`, and on 2026-08-21 that
# arrangement drove the arm for TEN MINUTES after the terminal said the run had
# ended. `timeout --foreground` signals only its direct child — `uv run` — which
# does not relay SIGINT to the python it forked, so the client never saw it;
# timeout then SIGKILLed `uv run`, and python was reparented to init and kept
# streaming actions, unsupervised, with no deadline left to stop it. The run log
# closed at 82 s while the flight recorder went on to 656 s.
#
# So: no `timeout`, and no `uv run`. We invoke the venv interpreter directly, so
# the client is this shell's own child, and this shell does the timing and the
# signalling itself. uv is only a venv manager here; it has no business in the
# signal path of a machine that moves.
# Refuse to start while another client still holds the arm. Without this, a
# second rollout blocks in the driver connect and then SILENTLY STARTS when the
# first one lets go — which is how a zombie run from ten minutes earlier began
# driving the arm on its own.
# Bracketed pattern: a bare `pgrep -f il_client_shield.py` also matches any
# shell whose command line merely CONTAINS that string — the wrapper that
# launched this script, an ssh one-liner, a grep. calib_sweep.sh learned this
# the hard way and its fix is the same; see 189db8a. A false positive here
# refuses a legitimate rollout, which is annoying rather than dangerous, but
# it is still wrong.
if pgrep -f "[i]l_client_shield.py" >/dev/null 2>&1; then
  echo "another rollout client is already running — refusing to start:" >&2
  pgrep -af "[i]l_client_shield.py" >&2
  echo "stop it (kill -INT <pid>) and re-run; scripts/il_recover_arm.sh if the arm is stranded" >&2
  exit 1
fi

# Contact-mic capture (opt-in). Started only after every preflight above has
# passed, so an aborted launch never leaves a recorder running; between here
# and audio::stop there is no exit path, and the -d cap inside audio::start
# bounds the recorder even if this shell is SIGKILLed. Evidence, never a gate.
if [ "${TATBOT_AUDIO:-0}" = 1 ]; then
  audio::start "${RUN_DIR:-$PWD}" "$(( DURATION + 75 ))"
  runlog::event audio_start device="${AUDIO_DEV:-none}"
fi

STATUS=0
cd "$CLIENT_CWD"
"$VENV_PY" "$REPO/scripts/il_client_shield.py" lerobot.async_inference.robot_client \
  --server_address="$SERVER" \
  --policy_type="$POLICY_TYPE" \
  --pretrained_name_or_path="$POLICY" \
  --policy_device=cuda \
  --robot.type=tatbot_follower \
  --robot.ee_tool="$EE_TOOL" \
  --robot.ip_address="$TATBOT_FOLLOWER_IP" \
  --robot.id=tatbot_follower_right \
  --robot.estop_required=true \
  --robot.abort_on_estop=true \
  --robot.require_z_floor=true \
  --robot.max_joint_velocity="$TARGET_VELOCITY" \
  --robot.controller_velocity_limit="$CONTROLLER_VELOCITY" \
  --robot.cameras="$WRIST_CAMERAS" \
  --robot.include_external_effort=$INCLUDE_EXT_EFF \
  --robot.mask_external_effort=$MASK_EXT_EFF_BOOL \
  --robot.depth_policy_encoding="$DEPTH_ENCODING" \
  --task="$TASK" \
  --fps=30 \
  --actions_per_chunk="$CHUNK" \
  --chunk_size_threshold="$THRESH" \
  --aggregate_fn_name=weighted_average \
  --robot.target_filter_tau=0.3 \
  "$@" &
CLIENT_PID=$!
cd "$REPO"

# An operator Ctrl+C takes the same path as the deadline: ask the client to
# land. No kill here — repeats are the shield's job to swallow.
trap 'echo "interrupt — landing the arm (do not press again)" >&2; \
      kill -INT "$CLIENT_PID" 2>/dev/null || true; INTERRUPTED=1' INT
INTERRUPTED=0
DEADLINE_FIRED=0

# Grace covers connect + homing before the clock matters.
DEADLINE=$(( $(date +%s) + DURATION + 14 ))
while kill -0 "$CLIENT_PID" 2>/dev/null; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && break
  sleep 0.5
done

if kill -0 "$CLIENT_PID" 2>/dev/null; then
  if [ "$INTERRUPTED" = 0 ]; then
    echo "reached the ${DURATION}s limit — landing the arm"
    DEADLINE_FIRED=1
  fi
  kill -INT "$CLIENT_PID" 2>/dev/null || true
fi

# Let the landing finish. staged->sleep takes ~8 s; 45 s is generous.
LAND_BY=$(( $(date +%s) + 45 ))
while kill -0 "$CLIENT_PID" 2>/dev/null && [ "$(date +%s)" -lt "$LAND_BY" ]; do
  sleep 0.5
done
trap - INT

if kill -0 "$CLIENT_PID" 2>/dev/null; then
  # It would not land. Kill the client and anything it spawned — an abandoned
  # client still commanding the arm is strictly worse than a hard stop — then
  # say so loudly, because the arm is now wherever it stopped.
  #
  # NOT a process-group kill. The client is this shell's own child and shares
  # its process group, which also contains this script and the operator's
  # shell; `kill -KILL -$PGID` would take the terminal down with it (verified).
  # The orphaning this replaced came from uv/timeout intermediaries that no
  # longer exist, so the PID and its children are the whole subtree.
  echo "CLIENT DID NOT LAND after 45 s — killing it" >&2
  pkill -KILL -P "$CLIENT_PID" 2>/dev/null || true
  kill -KILL "$CLIENT_PID" 2>/dev/null || true
  runlog::event error msg="client failed to land within 45s; killed"
  STATUS=137
  echo "ARM MAY BE STRANDED MID-MOTION — run: scripts/il_recover_arm.sh" >&2
else
  wait "$CLIENT_PID" 2>/dev/null || STATUS=$?
fi

# The client exits on a signal in BOTH the normal and the interrupted case, so
# its raw code cannot distinguish them — only this shell knows which happened.
if [ "$INTERRUPTED" = 1 ]; then
  STATUS=130
elif [ "$DEADLINE_FIRED" = 1 ] && { [ "$STATUS" = 130 ] || [ "$STATUS" = 143 ]; }; then
  # Only WE stopping it at the deadline counts as a normal finish. A client
  # that died on its own keeps its real exit code, so an early crash is never
  # dressed up as a completed rollout.
  echo "rollout reached its ${DURATION}s limit and landed normally"
  STATUS=0
fi

# A hardware stop is a failed run even if the client later releases, lands and
# returns zero. The incident logger emitted status=ok because only the final
# process code was considered. The structured follower event is authoritative.
if [ -n "${RUN_DIR:-}" ] && [ -f "$RUN_DIR/run.jsonl" ] \
   && grep -q '"kind": "estop"' "$RUN_DIR/run.jsonl"; then
  echo "E-stop event recorded — rollout failed regardless of client exit" >&2
  runlog::event error msg="E-stop event makes the rollout a failure"
  [ "$STATUS" -ne 0 ] || STATUS=70
fi

# Nothing of ours may outlive this script while still holding the arm.
if pgrep -f "[i]l_client_shield.py" >/dev/null 2>&1; then
  echo "ROLLOUT CLIENT SURVIVED ITS OWNER — terminating it and failing the run:" >&2
  mapfile -t SURVIVORS < <(pgrep -f "[i]l_client_shield.py")
  ps -o pid=,ppid=,etime=,args= -p "$(IFS=,; echo "${SURVIVORS[*]}")" >&2 || true
  kill -TERM "${SURVIVORS[@]}" 2>/dev/null || true
  for _ in {1..20}; do
    STILL=()
    for pid in "${SURVIVORS[@]}"; do kill -0 "$pid" 2>/dev/null && STILL+=("$pid"); done
    [ "${#STILL[@]}" -eq 0 ] && break
    sleep 0.1
  done
  [ "${#STILL[@]}" -eq 0 ] || kill -KILL "${STILL[@]}" 2>/dev/null || true
  runlog::event error msg="client survived the run; terminated"
  STATUS=137
fi

if [ "${TATBOT_AUDIO:-0}" = 1 ]; then
  audio::stop
  if [ -n "${RUN_DIR:-}" ] && [ -f "$RUN_DIR/audio.wav" ]; then
    runlog::artifact "$RUN_DIR/audio.wav"
    runlog::artifact "$RUN_DIR/audio_start.json"
  fi
fi

# Analysis is a REPORT, never a gate: bounded, non-fatal, and it cannot change
# the exit code. It runs here rather than in the robot class because that code
# path is the emergency landing — see scripts/il_analyze_rollout.py.
if [ "${TATBOT_ANALYZE:-1}" != 0 ] && [ -n "${RUN_DIR:-}" ]; then
  timeout 60 "$VENV_PY" "$REPO/scripts/il_analyze_rollout.py" "$RUN_DIR" \
    || echo "analysis failed (non-fatal) — rerun: scripts/il_analyze_rollout.py $RUN_DIR" >&2
fi
if [ "${TATBOT_ANALYZE:-1}" != 0 ] && [ -n "${RUN_DIR:-}" ] && [ -f "$RUN_DIR/audio.wav" ]; then
  timeout 60 "$VENV_PY" "$REPO/scripts/il_analyze_audio.py" "$RUN_DIR" \
    || echo "audio analysis failed (non-fatal) — rerun: scripts/il_analyze_audio.py $RUN_DIR" >&2
fi

exit "$STATUS"
