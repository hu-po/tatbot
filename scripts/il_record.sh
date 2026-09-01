#!/usr/bin/env bash
# Record imitation-learning episodes with the tatbot LeRobot plugin.
#
#   scripts/il_record.sh <dataset-name> "<task description>" [num_episodes] [extra lerobot-record args...]
#
# Leader and follower addresses come from the hardware profile; the follower carries the
# bounded-grip gripper; observations = both wrist RealSenses + joints +
# external effort. During recording: right arrow / n = next episode,
# left arrow / r = re-record, ESC / q = stop and finalize.
#
# The dataset is pushed to the Hugging Face hub when `hf auth login` has been
# done; otherwise it stays local under ~/.cache/huggingface/lerobot/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot record"
# shellcheck source=scripts/lib/estop_guard.sh
source "$REPO/scripts/lib/estop_guard.sh"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
# shellcheck source=scripts/lib/nodes.sh
source "$REPO/scripts/lib/nodes.sh"
# The PoE-camera node and its checkout, from config/nodes.json (bulk capture
# uses its LAN path); empty = no such node here and PoE evidence is skipped.
POE_SSH="$(tatbot_nodes::target poe-cameras lan)"
POE_NODE="${POE_SSH%@*}"
# The camera node's OWN checkout path (its tree, not ours).
POE_CHECKOUT="$(tatbot_nodes::checkout poe-cameras)"

# Wrist depth cameras: names and serials from the visiond sensor registry
# (rust/visiond/config/vision.toml) — the same file the vision daemon reads,
# so a swapped camera is a config edit, not a launcher edit.
WRIST_CAMERAS="$(python3 - "$REPO" <<'PY'
import sys, tomllib
from pathlib import Path
cams = tomllib.loads((Path(sys.argv[1]) / "rust/visiond/config/vision.toml").read_text())
rs = cams.get("cameras", {}).get("realsense", [])
by_role = {c["role"]: c["serial"] for c in rs if c.get("role")}
missing = [r for r in ("wrist_upper", "wrist_lower") if r not in by_role]
if missing:
    sys.exit(f"sensor registry has no role= for {', '.join(missing)}")
parts = [
    f"{role}: {{type: intelrealsense, serial_number_or_name: '{by_role[role]}', "
    "width: 640, height: 480, fps: 30, use_depth: true}"
    for role in ("wrist_upper", "wrist_lower")
]
print("{" + ", ".join(parts) + "}")
PY
)"
[ "$WRIST_CAMERAS" = "{}" ] && { echo "il_record: no RealSense cameras in the sensor registry" >&2; exit 1; }
profile_env::require || exit $?
# --push: publish this dataset to the hub (default: local only).
WANT_PUSH="${TATBOT_PUSH:-0}"
_args=()
for a in "$@"; do
  case "$a" in
    --push) WANT_PUSH=1 ;;
    --no-push) WANT_PUSH=0 ;;
    *) _args+=("$a") ;;
  esac
done
set -- "${_args[@]+"${_args[@]}"}"
# shellcheck source=scripts/lib/ee_tool.sh
source "$REPO/scripts/lib/ee_tool.sh"
# shellcheck source=scripts/lib/runlog.sh
source "$REPO/scripts/lib/runlog.sh"

# No DISPLAY on purpose: with X available, lerobot's keyboard listener grabs
# global X keys (on the machine's own screen) instead of reading n/r/q from
# this terminal. Headless keeps the terminal fallback working over SSH.
unset DISPLAY
PROJECT="$REPO/python/lerobot_robot_tatbot"

# Golden arm/tatbot configs (loaded at connect) + tuning cockpit on :8899.
export TATBOT_CONFIG_DIR="${TATBOT_CONFIG_DIR:-$REPO/config/trossen}"

# shellcheck source=scripts/lib/dip_hook.sh
source "$REPO/scripts/lib/dip_hook.sh"
# Strip --ee-tool, --dip and --no-ink before the positionals; the rest passes through untouched.
ee_tool::strip "$@"; set -- "${EE_TOOL_ARGS[@]}"
dip_hook::strip "$@" || exit $?; set -- "${DIP_HOOK_ARGS[@]}"
NAME="${1:?usage: il_record.sh <dataset-name> \"<task>\" [num_episodes]}"
TASK="${2:?usage: il_record.sh <dataset-name> \"<task>\" [num_episodes]}"
EPISODES="${3:-5}"
ee_tool::require || exit $?
shift $(( $# < 3 ? $# : 3 ))
estop_guard::reject_overrides "$@"
runlog::init record --set stack=lerobot --set "estop=$TATBOT_ESTOP_DEVICE" \
  --set dataset="$NAME" --set episodes="$EPISODES"
# --dip / --no-ink: the ink hook (scripts/lib/dip_hook.sh); the run is open,
# so a dip's ledger events are mirrored into this run's ink.jsonl.
dip_hook::run || exit $?

# Hub identity: the repo's git-ignored .env (HF_TOKEN, HF_USER) fills what the
# environment does not already carry; `hf auth login` is the fallback. The
# whoami output differs by huggingface_hub version (`user=<name> orgs=...` on
# one line, or `✓ Logged in` then an indented `user: <name>`); the old parse
# read only line 1 with a colon, matched nothing on 1.x, and every recording
# fell back to local/ even on a logged-in node. Match the user line wherever
# it is, either separator.
if [ -f "$REPO/.env" ]; then
  set -a; # shellcheck source=/dev/null
  source "$REPO/.env"; set +a
fi
if [ -z "${HF_USER:-}" ]; then
  HF_USER=$(NO_COLOR=1 uv run --project "$PROJECT" hf auth whoami 2>/dev/null \
    | sed -n 's/^[[:space:]]*user[=:][[:space:]]*\([^[:space:]]*\).*/\1/p' | head -1) || true
fi
# A dead token makes lerobot's push fail SILENTLY — a progress line, zero
# files, no error (2026-08-31: a rotated token cost a session its hub copy).
# push:true only with a token that actually answers whoami.
if [ -n "${HF_USER:-}" ] && ! NO_COLOR=1 uv run --project "$PROJECT" hf auth whoami >/dev/null 2>&1; then
  echo "WARNING: Hugging Face token is invalid (rotated?). Recording locally only;" >&2
  echo "         refresh HF_TOKEN in $REPO/.env, then: tatbot data push -- --root <dataset>" >&2
  HF_USER=""
fi
if [ -n "${HF_USER:-}" ] && [ "${WANT_PUSH:-0}" = 1 ]; then
  # Private by default: this is wrist-camera footage of a private workshop,
  # and a PRO subscription meters private storage on the USER namespace.
  REPO_ID="${HF_DATASET_NAMESPACE:-$HF_USER}/$NAME"
  PUSH=true
  PRIVATE="${HF_DATASET_PUBLIC:+false}"; PRIVATE="${PRIVATE:-true}"
elif [ -n "${HF_USER:-}" ]; then
  # Publishing is OPT-IN (2026-08-31): a recording session should never put
  # workshop footage on the hub as a side effect. Pass --push (tatbot record
  # --push) or set TATBOT_PUSH=1 when you actually mean to publish.
  echo "recording locally only — pass --push to publish (or later: tatbot data push -- --root <dataset dir>)"
  REPO_ID="${HF_DATASET_NAMESPACE:-$HF_USER}/$NAME"
  PUSH=false
  PRIVATE=true
else
  echo "WARNING: no Hugging Face identity (put HF_TOKEN/HF_USER in $REPO/.env or run:"
  echo "         uv run --project $PROJECT hf auth login). Recording locally only;"
  echo "         push later with: tatbot data push -- --root <dataset dir>"
  REPO_ID="local/$NAME"
  PUSH=false
  PRIVATE=true
fi
echo "dataset: $REPO_ID  episodes: $EPISODES  task: $TASK  push: $PUSH (private: $PRIVATE)"

# The sim generates its prompt from the tool datasheet and the substrate
# registry; a hand-typed one here can drift from it, and a language-conditioned
# policy needs only that difference to tell the two domains apart. Warn, never
# block: the operator may be recording something the sim does not model.
CANON="$(python3 "$REPO/scripts/lib/canonical_removal_task.py" "$REPO" "${EE_TOOL:-}" 2>/dev/null || true)"
if [ -n "$CANON" ] && [ "$TASK" != "$CANON" ]; then
  echo "WARNING: the sim words this task differently, so a policy could tell the"
  echo "         two apart by the prompt alone. Consider:"
  echo "           $CANON"
fi


# Fail fast and friendly if the arms are not powered on.
for ip in "$TATBOT_LEADER_IP" "$TATBOT_FOLLOWER_IP"; do
  ping -c1 -W1 "$ip" >/dev/null 2>&1 || {
    echo "Arm at $ip is not reachable — is it powered on? (arms take ~20 s to boot)" >&2
    exit 1
  }
done

# Apply/verify local lerobot patches (empty-episode crash guard, etc).
uv run --project "$PROJECT" python "$REPO/scripts/il_patch_lerobot.py"

# --- evidence sidecars -------------------------------------------------------
# Record everything the rig can hear and see BESIDE the dataset (operator
# decision 2026-08-31): the policy's observations stay wrist-only, but the EE
# microphone and the five Amcrest PoE cameras are evidence. Same contract as
# every audio:: function — evidence, never a gate: a missing device or an
# an unreachable camera node warns and the session records anyway.
# shellcheck source=scripts/il_audio_record.sh
source "$REPO/scripts/il_audio_record.sh"
EVIDENCE_CAP_S="${TATBOT_IL_EVIDENCE_CAP_S:-7200}"  # backstop; stopped at session end
audio::start "${RUN_DIR:-$PWD}" "$EVIDENCE_CAP_S"

POE_OUT=""
# the camera node's NVMe, never its small eMMC root: that root hit 100% on 2026-08-31 and
# the capture died at mkdir while this launch still reported success.
# No --decoded: that flag stores raw BGR8 (14.8 MB/frame — a 100 s session
# wrote 14 GB at ~2 fps); encoded .h264 keeps the camera's full ~20 fps at
# ~35x less disk (measured 2026-08-31).
POE_DIR="${TATBOT_IL_POE_DIR:-/mnt/tatbot-logs/vision}"
if [ "${TATBOT_IL_POE:-1}" = 1 ]; then
  POE_OUT="$POE_DIR/il-${NAME}-$(date +%Y%m%d_%H%M%S)-poe"
  if ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$POE_SSH" \
      "set -a; source ~/.config/tatbot/cameras.env; set +a; mkdir -p $POE_DIR; \
       nohup $POE_CHECKOUT/rust/target/release/tatbot-visiond capture-poe-all \
         $POE_CHECKOUT/rust/visiond/config/vision.toml \
         --stream ${TATBOT_IL_POE_STREAM:-main} \
         --duration-seconds $EVIDENCE_CAP_S \
         --output $POE_OUT >$POE_OUT.log 2>&1 &" 2>/dev/null; then
    # nohup+& makes ssh exit 0 no matter what — trust only the capture's own
    # per-camera directory, which visiond creates at startup.
    sleep 3
    if ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$POE_SSH" \
        "test -d $POE_OUT/camera1" 2>/dev/null; then
      echo "PoE evidence: $POE_NODE:$POE_OUT (PoE cameras; stops with this session)"
    else
      echo "WARNING: PoE capture did not start on $POE_NODE — recording without it:" >&2
      ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$POE_SSH" \
        "tail -3 $POE_OUT.log 2>/dev/null" >&2 || true
      POE_OUT=""
    fi
  else
    echo "WARNING: PoE evidence skipped — $POE_NODE not reachable from this node" >&2
    echo "         (one-time: ssh-keygen here, then ssh-copy-id $POE_SSH)" >&2
    POE_OUT=""
  fi
fi
evidence_stop() {
  audio::stop
  [ -n "$POE_OUT" ] || return 0
  # Anchored to the binary: an unanchored pattern matches this wrapper's own
  # argv and pkill kills the shell (scripts/live/cockpit.sh, 2026-08-31).
  ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$POE_SSH" \
    "pkill -f -- '^[^ ]*/tatbot-visiond capture-poe-all.*il-${NAME}-'" 2>/dev/null || true
}
# EXIT only, deliberately: the INT/TERM path belongs to lerobot's own arm
# landing; the sidecars' -d/--duration caps bound them even on a SIGKILL.
trap evidence_stop EXIT

# Recording-only reversal loosening (operator, 2026-08-31): drawing a 6 mm
# square is four wrist reversals per shape, and the rollout-tuned guard
# (0.2 rad/s, 4 in 1 s) tripped twice in two hand-guided sessions. Rollouts
# keep the strict defaults — a policy has no business reversing that fast.
runlog::run uv run --project "$PROJECT" lerobot-record \
  --robot.type=tatbot_follower \
  --robot.ee_tool="$EE_TOOL" \
  --robot.ip_address="$TATBOT_FOLLOWER_IP" \
  --robot.id=tatbot_follower_right \
  --robot.estop_required=true \
  --robot.reversal_min_velocity=0.5 \
  --robot.reversal_abort_count=6 \
  --robot.cameras="$WRIST_CAMERAS" \
  --teleop.type=tatbot_leader_teleop \
  --teleop.ip_address="$TATBOT_LEADER_IP" \
  --teleop.id=tatbot_leader_left \
  --teleop.estop_required=true \
  --display_data=false \
  --dataset.repo_id="$REPO_ID" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.single_task="$TASK" \
  --dataset.push_to_hub="$PUSH" \
  --dataset.private="$PRIVATE" \
  --dataset.tags='["tatbot", "real", "trossen"]' \
  --dataset.reset_time_s=8 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  "$@"

# Stamp the dataset with the tool that made it. LeRobot records only
# `robot_type`, which cannot tell one tattoo machine from another; without
# this a dataset stops being self-describing the moment a pen is swapped.
#
# Resolve the directory rather than trusting REPO_ID: lerobot's
# DatasetConfig.stamp_repo_id() appends _YYYYMMDD_HHMMSS at creation so each
# session is unique, so the recording lands at "<REPO_ID>_<stamp>" and never
# at REPO_ID. This step had been looking at the unstamped path and failing
# silently since it was added — on 2026-08-26 every dataset in the local cache
# turned out to be unstamped. Take the newest match, which is this run's.
LEROBOT_CACHE="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
DATASET_ROOT=$(ls -1dt "$LEROBOT_CACHE/${REPO_ID}"_* 2>/dev/null | head -1)
if [ -z "$DATASET_ROOT" ] && [ -d "$LEROBOT_CACHE/$REPO_ID" ]; then
  DATASET_ROOT="$LEROBOT_CACHE/$REPO_ID"   # --dataset.no_stamp=true was passed
fi
TOOL_META_ARGS=(--tool-id "$EE_TOOL")
if [ -n "$DATASET_ROOT" ]; then
  # Both: --root is the local directory to stamp, --repo-id is where --push
  # sends the file. The hub repo carries the STAMPED name too, so deriving it
  # from the resolved path is what keeps the two pointing at one dataset.
  TOOL_META_ARGS+=(--root "$DATASET_ROOT"
                   --repo-id "${DATASET_ROOT#"$LEROBOT_CACHE/"}")
else
  TOOL_META_ARGS+=(--repo-id "$REPO_ID")   # let it report the miss
fi
if [ "$PUSH" = true ]; then
  TOOL_META_ARGS+=(--push)
fi
uv run --project "$PROJECT" python "$REPO/scripts/il_tool_meta.py" "${TOOL_META_ARGS[@]}" \
  || echo "WARNING: dataset recorded but not stamped with its tool (see above)" >&2

# Evidence travels with the local dataset: session audio beside the episodes,
# and a pointer to where the camera node holds the PoE footage (too big to copy here).
evidence_stop
if [ -n "$DATASET_ROOT" ]; then
  if [ -f "${RUN_DIR:-$PWD}/audio.wav" ]; then
    mkdir -p "$DATASET_ROOT/audio"
    cp "${RUN_DIR:-$PWD}/audio.wav" "${RUN_DIR:-$PWD}/audio_start.json" \
       "$DATASET_ROOT/audio/" 2>/dev/null || true
    echo "audio evidence: $DATASET_ROOT/audio/audio.wav"
  fi
  if [ -n "$POE_OUT" ]; then
    mkdir -p "$DATASET_ROOT/meta"
    printf '{"poe": "%s:%s"}\n' "$POE_NODE" "$POE_OUT" > "$DATASET_ROOT/meta/evidence.json" || true
    echo "PoE evidence pointer: $DATASET_ROOT/meta/evidence.json"
  fi
fi

# meta/hub.json: which hub repo and commit hold this recording, so a training
# run that consumes it (by hub pull or by rsync) can name its provenance.
if [ "$PUSH" = true ] && [ -n "$DATASET_ROOT" ]; then
  # `--` is load-bearing: the tatbot-CLI shim behind dataset_hub.sh reorders
  # bare flags (push --root X became push X --root) — everything after `--`
  # passes through verbatim.
  "$REPO/scripts/dataset_hub.sh" push -- --root "$DATASET_ROOT" \
      --repo-id "${DATASET_ROOT#"$LEROBOT_CACHE/"}" \
    || echo "WARNING: hub copy not reconciled; run: tatbot data push -- --root $DATASET_ROOT" >&2
fi
