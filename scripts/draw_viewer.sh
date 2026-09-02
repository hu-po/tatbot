#!/usr/bin/env bash
# `tatbot draw viewer`: a capped Rerun viewer on THIS node, bound on every
# interface, for a draw session launched from here with `--on <arm-node>`
# (docs/draw.md). The arm node finds it through SSH_CONNECTION and streams the
# shadow (surface, normals, orbit, path, captures) here. Nothing touches the arm.
#
#   scripts/draw_viewer.sh [--memory-limit 3GB]
#
# Ctrl+C (or closing the viewer) ends it. Viewer memory is capped; keep it so.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot draw viewer"
# shellcheck source=scripts/vision/rerun_session.sh
source "$REPO/scripts/vision/rerun_session.sh"

VIEWER_MEMORY_LIMIT="${VIEWER_MEMORY_LIMIT:-3GB}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --memory-limit) [ "$#" -ge 2 ] || { echo "--memory-limit needs a value" >&2; exit 2; }; VIEWER_MEMORY_LIMIT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
  echo "no display: run this in a graphical session on the node you launch the draw from" >&2
  exit 1
fi

cleanup() { rerun_session::stop_viewer "${RERUN_VIEWER_PID:-}"; }
trap cleanup EXIT INT TERM

rerun_session::start_viewer "$VIEWER_MEMORY_LIMIT" 512MB
echo "Rerun viewer up (memory cap $VIEWER_MEMORY_LIMIT), bound on every interface, port 9876."
echo "A draw session launched from this node with --on <arm-node> streams here by itself; explicit form:"
ip -4 -o addr show 2>/dev/null | awk '!/ lo /{split($4, a, "/"); print "  --rerun-viewer rerun+http://" a[1] ":9876/proxy   (" $2 ")"}'
echo "Ctrl+C to stop."
wait "$RERUN_VIEWER_PID"
