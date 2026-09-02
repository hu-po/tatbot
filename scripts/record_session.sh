#!/usr/bin/env bash
# Record a full tatbot session: all 7 cameras + leader/follower teleop,
# merged into one Rerun recording on a shared timeline.
#
#   scripts/record_session.sh [DURATION_S] [extra wxai_teleop args...]
#
# DURATION_S (default 30) is the camera recording window — start hand-guiding
# as soon as the teleop banner appears. The teleop itself runs until you
# Ctrl+C (then support both arms and press Enter to release them).
# Extra args pass through to wxai_teleop (e.g. --contact-cap 12 --ff-gain 0).
#
# Outputs under ~/tatbot-logs/vision/session-<timestamp>*:
#   -poe/ -rs/   raw camera evidence (full resolution, heavy: ~1.4 GB/s total)
#   .rrd         merged JPEG-encoded Rerun recording (auto-opened on a display)
# The teleop flight log lands in ~/tatbot-logs/teleop/ as usual.
#
# Pass --no-view anywhere to skip opening the viewer at the end.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot teleop run"
# shellcheck source=scripts/lib/estop_guard.sh
source "$REPO/scripts/lib/estop_guard.sh"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require || exit $?
estop_guard::reject_overrides "$@"
for a in "$@"; do
  case "$a" in
    --square-probe-mm|--square-probe-mm=*|--square-edge-s|--square-edge-s=*)
      echo "REFUSING Cartesian square passthrough from teleop run." >&2
      echo "  Use: tatbot --ee-tool <id> teleop square --nonce <fresh-literal>" >&2
      exit 3 ;;
    --spiral-radius-mm|--spiral-radius-mm=*|--spiral-turns|--spiral-turns=*|--spiral-duration-s|--spiral-duration-s=*|--spiral-ease-s|--spiral-ease-s=*|--spiral-carriage-ik)
      echo "REFUSING Cartesian spiral passthrough from teleop run." >&2
      echo "  Use: tatbot --ee-tool <id> teleop spiral --nonce <fresh-literal>" >&2
      exit 3 ;;
  esac
done
# shellcheck source=scripts/lib/runlog.sh
source "$REPO/scripts/lib/runlog.sh"
# shellcheck source=scripts/lib/paths.sh
source "$REPO/scripts/lib/paths.sh"
LOG_ROOT="$(tatbot_paths::log_root)"
export TATBOT_LOG_ROOT="$LOG_ROOT"
runlog::init record --set "estop=$TATBOT_ESTOP_DEVICE"
VISIOND="$REPO/rust/target/release/tatbot-visiond"
TELEOP="$REPO/cpp/teleop/build/wxai_teleop"
CAMS_ENV="$HOME/.config/tatbot/cameras.env"
V="$LOG_ROOT/vision"
TS=$(date +%Y%m%d_%H%M%S)
SESSION_EPOCH=$(date +%s)
RERUN_MAX_FPS="${RERUN_MAX_FPS:-6}"
# 1.0: the recorder stores encoded frames since the 2026-08-27 vision work, and
# replay-rerun can only rescale decoded BGR/RGB ("visual scaling requires decoded
# BGR/RGB frames" failed every merge on 2026-08-30 at 0.5).
RERUN_IMAGE_SCALE="${RERUN_IMAGE_SCALE:-1.0}"
RERUN_JPEG_QUALITY="${RERUN_JPEG_QUALITY:-80}"
VIEWER_MEMORY_LIMIT="${VIEWER_MEMORY_LIMIT:-3GB}"

DURATION=30
VIEW=1
TELEOP_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --no-view) VIEW=0 ;;
    *) if [[ "$arg" =~ ^[0-9]+$ && ${#TELEOP_ARGS[@]} -eq 0 ]]; then DURATION=$arg; else TELEOP_ARGS+=("$arg"); fi ;;
  esac
done

[ -x "$VISIOND" ] || { echo "missing $VISIOND — build: cd $REPO/rust && cargo build --release --features 'gstreamer,realsense,rerun'"; exit 1; }
[ -x "$TELEOP" ] || { echo "missing $TELEOP — build: cd $REPO/cpp/teleop && cmake -B build -S . && cmake --build build"; exit 1; }
[ -f "$CAMS_ENV" ] || { echo "missing $CAMS_ENV (Amcrest credentials)"; exit 1; }
set -a
# shellcheck disable=SC1090
source "$CAMS_ENV"
set +a
grep -q FILL_ME "$CAMS_ENV" && { echo "$CAMS_ENV still has FILL_ME placeholders"; exit 1; }

mkdir -p "$V"
echo "=== tatbot session $TS: cameras rolling for ${DURATION}s — start guiding the leader ==="

"$VISIOND" capture-poe-all "$REPO/rust/visiond/config/vision.toml" \
  --stream main --decoded --duration-seconds "$DURATION" \
  --output "$V/session-$TS-poe" > "$V/session-$TS-poe.log" 2>&1 &
POE=$!
"$VISIOND" capture-realsense-all "$REPO/rust/visiond/config/vision.toml" \
  --duration-seconds "$DURATION" \
  --output "$V/session-$TS-rs" > "$V/session-$TS-rs.log" 2>&1 &
RS=$!

# Teleop in the foreground: Ctrl+C stops it, then support both arms + Enter.
TELEOP_STATUS=0
runlog::run "$TELEOP" "$TATBOT_LEADER_IP" "$TATBOT_FOLLOWER_IP" \
  "$REPO/config/trossen/leader.yaml" "$REPO/config/trossen/follower.yaml" \
  --estop "$TATBOT_ESTOP_DEVICE" \
  "${TELEOP_ARGS[@]+"${TELEOP_ARGS[@]}"}" || TELEOP_STATUS=$?

# From here on Ctrl+C must never corrupt session artifacts: the first press
# in a phase warns, the second aborts that phase cleanly.
ABORT=0
trap 'echo; echo "(Ctrl+C — press again to abort the current phase)"; ABORT=$((ABORT+1))' INT

echo "=== waiting for camera recorders to finish ==="
BASE=$ABORT
while kill -0 "$POE" 2>/dev/null || kill -0 "$RS" 2>/dev/null; do
  wait "$POE" "$RS" 2>/dev/null && break
  if [ "$ABORT" -gt $((BASE + 1)) ]; then
    echo "abandoning camera recorders"
    kill "$POE" "$RS" 2>/dev/null || true
    break
  fi
done
grep -h -E "synchronized_sets|errors=" "$V/session-$TS-poe.log" "$V/session-$TS-rs.log" || true
[ "$TELEOP_STATUS" -ne 0 ] && echo "WARNING: teleop exited with status $TELEOP_STATUS"

# Only merge a teleop log recorded by THIS session, never a stale one.
WXTL=$(find "$LOG_ROOT"/teleop -name "teleop_*.wxtl" -newermt "@$SESSION_EPOCH" 2>/dev/null | sort | tail -1)
echo "=== merging (teleop log: ${WXTL:-none}) ==="
RRD="$V/session-$TS.rrd"
MERGE_CMD=("$VISIOND" replay-rerun "$V/session-$TS-poe" "$V/session-$TS-rs"
  ${WXTL:+--teleop-log "$WXTL"}
  --urdf "$REPO/urdf/tatbot.urdf" --rerun-layout full
  --max-fps "$RERUN_MAX_FPS" --image-scale "$RERUN_IMAGE_SCALE"
  --jpeg-quality "$RERUN_JPEG_QUALITY" --output "$RRD.tmp")
"${MERGE_CMD[@]}" &
MERGE=$!
BASE=$ABORT
while kill -0 "$MERGE" 2>/dev/null; do
  wait "$MERGE" 2>/dev/null && break
  if [ "$ABORT" -gt $((BASE + 1)) ]; then
    kill "$MERGE" 2>/dev/null || true
    rm -f "$RRD.tmp"
    echo "merge aborted — regenerate later with:"
    printf '  %q ' "${MERGE_CMD[@]}"; echo
    echo "  (then rename the .tmp output to $RRD)"
    exit 130
  fi
done
# Reap for the real exit status (a trap-interrupted wait above returns >128
# even when the merge later succeeds).
wait "$MERGE" 2>/dev/null || { rm -f "$RRD.tmp"; echo "merge failed"; exit 1; }
mv "$RRD.tmp" "$RRD"
trap - INT

echo "=== recording: $RRD ==="
if [ "$VIEW" -eq 1 ]; then
  # Open maximized: on the local desktop if this shell has one, else on the
  # machine's own display (e.g. when run over SSH).
  if [ -z "${DISPLAY:-}" ] && [ -e "/run/user/$(id -u)/gdm/Xauthority" ]; then
    XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
    export DISPLAY=:0 XAUTHORITY
  fi
  source "$REPO/scripts/rerun_maximized.bash"
  rerun --memory-limit "$VIEWER_MEMORY_LIMIT" "$V/session-$TS.rrd"
fi
