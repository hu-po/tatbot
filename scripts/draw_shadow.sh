#!/usr/bin/env bash
# `tatbot draw shadow <dir>`: open the shadow of a draw dir in a capped Rerun
# viewer — the mapped surface, its normals, the orbit, the compiled path and
# the raw D405 captures (docs/draw.md). Nothing here touches the arm.
#
#   scripts/draw_shadow.sh <draw-dir> [extra draw_shadow.py args...]
#
# Ctrl+C (or closing the viewer) ends it. Viewer memory is capped; keep it so.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot draw shadow"
# shellcheck source=scripts/vision/rerun_session.sh
source "$REPO/scripts/vision/rerun_session.sh"

[ "$#" -ge 1 ] || { echo "usage: draw_shadow.sh <draw-dir> [draw_shadow.py args]" >&2; exit 2; }
DRAW_DIR="$1"; shift
[ -d "$DRAW_DIR" ] || { echo "not a draw dir: $DRAW_DIR" >&2; exit 2; }
VIEWER_MEMORY_LIMIT="${VIEWER_MEMORY_LIMIT:-3GB}"

draw_python() {
  local home="${HOME:-~}" p
  for p in "$REPO/python/lerobot_robot_tatbot/.venv/bin/python" \
           "${TATBOT_SERVE_ROOT:-$home/il-serve}/.venv/bin/python" \
           "${TATBOT_TRAIN_ROOT:-$home/il-train}/.venv/bin/python"; do
    if [ -x "$p" ]; then echo "$p"; return 0; fi
  done
  return 1
}
DRAW_PY="$(draw_python)" || {
  echo "no LeRobot interpreter found (python/lerobot_robot_tatbot/.venv, ~/il-serve, ~/il-train)" >&2
  exit 1
}

cleanup() { rerun_session::stop_viewer "${RERUN_VIEWER_PID:-}"; }
trap cleanup EXIT INT TERM

rerun_session::start_viewer "$VIEWER_MEMORY_LIMIT" 512MB
LAN_IP="$(rerun_session::lan_ip)"
PROXY="rerun+http://${LAN_IP:-127.0.0.1}:9876/proxy"
echo "=== draw shadow: $DRAW_DIR -> $PROXY (viewer cap $VIEWER_MEMORY_LIMIT; Ctrl+C to stop) ==="
"$DRAW_PY" "$REPO/scripts/draw_shadow.py" "$DRAW_DIR" --connect "$PROXY" "$@"
wait "$RERUN_VIEWER_PID"
