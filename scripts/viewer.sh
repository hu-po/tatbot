#!/usr/bin/env bash
# `tatbot viewer <start|stop|status|open|view|install|serve>`:
# the one fleet Rerun viewer every workflow streams into (docs/vision.md).
#
#   start     on the rerun-server node: headless server on :9876 (+ web :9090)
#   open      attach a capped window on THIS node to the fleet viewer
#   stop      here: close this node's window; on the server node: also the server
#   status    fleet reachability / local server / window / version / proxy URL
#   view F    stream a recorded .rrd/.wxtl (or a teleop run dir) into the fleet viewer
#   install   on the rerun-server node: enable the headless tatbot-viewer@ (sudo)
#   serve     run the server in the foreground (what tatbot-viewer@ execs)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot viewer"
# shellcheck source=scripts/vision/rerun_session.sh
source "$REPO/scripts/vision/rerun_session.sh"

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }
cmd="${1:-status}"; shift || true
SERVER="$(rerun_session::fleet_server)" || true
SERVER_NODE="${SERVER%% *}"

need_server_node() {
  if [ -n "$SERVER" ] && ! rerun_session::is_server_node; then
    echo "tatbot viewer $cmd runs on the rerun-server node ($SERVER_NODE): tatbot --on $SERVER_NODE viewer $cmd" >&2
    exit 4
  fi
}

serve_args() {
  local -a a=(--serve-grpc)
  [ "${TATBOT_VIEWER_WEB:-0}" = 1 ] && a=(--serve-web --web-viewer-port "${RERUN_WEB_PORT:-9090}")
  printf '%s\n' "${a[@]}"
}

case "$cmd" in
  start)
    need_server_node
    if systemctl is-enabled "tatbot-viewer@$USER.service" >/dev/null 2>&1 && ! rerun_session::server_pid >/dev/null; then
      sudo -n systemctl start "tatbot-viewer@$USER.service" 2>/dev/null || systemctl start "tatbot-viewer@$USER.service"
      for _ in $(seq 1 20); do rerun_session::server_pid >/dev/null && break; sleep 0.5; done
    fi
    TATBOT_RERUN_LOCAL=1 rerun_session::ensure_viewer "$@"
    ;;
  open)
    proxy="$(rerun_session::fleet_proxy)"
    if rerun_session::is_server_node || [ -z "$SERVER" ]; then
      rerun_session::start_server
      proxy="rerun+http://127.0.0.1:$RERUN_PORT/proxy"
    elif ! rerun_session::reachable "$proxy"; then
      echo "fleet viewer $proxy not answering (tatbot --on $SERVER_NODE viewer start)" >&2; exit 1
    fi
    rerun_session::open_gui "${1:-$VIEWER_MEMORY_LIMIT}" "$proxy" borrow \
      || { echo "no display on this node; the web viewer still shows everything" >&2; exit 1; }
    echo "window attached to $proxy (cap ${1:-$VIEWER_MEMORY_LIMIT})"
    ;;
  stop)
    if rerun_session::is_server_node && systemctl is-active "tatbot-viewer@$USER.service" >/dev/null 2>&1; then
      sudo -n systemctl stop "tatbot-viewer@$USER.service" 2>/dev/null || systemctl stop "tatbot-viewer@$USER.service"
    fi
    rerun_session::stop_viewer
    echo "viewer stopped here"
    ;;
  status)
    rerun_session::status
    if rerun_session::is_server_node; then
      u="tatbot-viewer@$USER"
      if systemctl cat "$u.service" >/dev/null 2>&1; then
        echo "unit:   $u.service $(systemctl is-active "$u.service" 2>/dev/null || true) ($(systemctl is-enabled "$u.service" 2>/dev/null || true))"
      else
        echo "unit:   $u.service not installed (tatbot viewer install)"
      fi
    fi
    ;;
  blueprint)
    # Internal display bootstrap. Workflows never call this or choose a view.
    VISIOND="${VISIOND:-$REPO/rust/target/release/tatbot-visiond}"
    if ! [ -x "$VISIOND" ] || ! "$VISIOND" send-blueprint --help >/dev/null 2>&1; then
      echo "the fixed blueprint lives in visiond (rust/visiond/src/rerun_viewer.rs); build it here:" >&2
      echo "  cd $REPO/rust && cargo build --release --features rerun" >&2
      exit 1
    fi
    need_server_node
    proxy="rerun+http://127.0.0.1:$RERUN_PORT/proxy"
    for _ in $(seq 1 40); do
      rerun_session::reachable "$proxy" && break
      sleep 0.25
    done
    rerun_session::reachable "$proxy" || { echo "viewer $proxy not answering" >&2; exit 1; }
    "$VISIOND" send-blueprint --connect "$proxy" --recording-id live-cameras
    ;;
  view)
    [ "$#" -ge 1 ] || { echo "usage: tatbot viewer view <file.rrd|file.wxtl|teleop-run-dir>" >&2; exit 2; }
    target="${1/#\~/$HOME}"; shift   # a quoted ~ from an ssh hop
    if [ -d "$target" ]; then
      # A teleop run dir (runlog: meta.json, console.log) holds no log itself;
      # the flight log is the .wxtl written beside it when the run started.
      found="$(python3 - "$target" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
d = Path(sys.argv[1])
inside = sorted(d.glob("**/*.wxtl"))
if inside:
    print(inside[-1]); sys.exit(0)
meta = d / "meta.json"
if meta.is_file():
    m = json.loads(meta.read_text())
    t0 = datetime.fromisoformat(m["started_at"].replace("Z", "+00:00")).timestamp()
    t1 = t0 + float(m.get("duration_s") or 0) + 120
    hits = [p for p in d.parent.glob("*.wxtl") if t0 - 5 <= p.stat().st_mtime <= t1]
    if hits:
        print(min(hits, key=lambda p: p.stat().st_mtime)); sys.exit(0)
sys.exit(1)
PY
)" || { echo "no .wxtl in or beside $target (a run dir's log is the .wxtl written next to it)" >&2; exit 2; }
      target="$found"
      echo "=== run dir -> $target"
    fi
    [ -f "$target" ] || { echo "not a file: $target" >&2; exit 2; }
    proxy="$(rerun_session::fleet_proxy)"
    if rerun_session::is_server_node || [ -z "$SERVER" ]; then rerun_session::start_server; proxy="rerun+http://127.0.0.1:$RERUN_PORT/proxy"; fi
    if ! rerun_session::reachable "$proxy"; then
      echo "fleet viewer unavailable at $proxy; restore it with 'tatbot viewer start', then retry" >&2
      exit 5
    fi
    # A stamped workflow id: the viewer lists it as "Replay <file> HH:MM".
    rid="replay-$(basename "${target%.*}")-$(date -u +%Y%m%dT%H%M%SZ)"
    case "$target" in
      *.wxtl)
        VISIOND="${VISIOND:-$REPO/rust/target/release/tatbot-visiond}"
        [ -x "$VISIOND" ] || { echo "needs $VISIOND (cargo build --release --features rerun)" >&2; exit 1; }
        echo "=== replaying $target -> $proxy (recording $rid)"
        exec "$VISIOND" replay-rerun --teleop-log "$target" --urdf "$REPO/urdf/tatbot.urdf" \
          --teleop-leader "${TELEOP_LEADER:-left}" --connect "$proxy" --recording-id "$rid" "$@"
        ;;
      *.rrd)
        echo "=== forwarding data-only $target -> $proxy (recording $rid)"
        export PATH="$HOME/.local/bin:$PATH"
        (
        import_tmp="$(mktemp -d)"
        trap 'rm -rf -- "$import_tmp"' EXIT
        VISIOND="${VISIOND:-$REPO/rust/target/release/tatbot-visiond}"
        [ -x "$VISIOND" ] || {
          echo "needs $VISIOND (cargo build --release --features rerun)" >&2
          exit 1
        }
        "$VISIOND" sanitize-rrd --input "$target" --output "$import_tmp/data-only.rrd" \
          --recording-id "$rid"
        "$RERUN_BIN" rrd verify "$import_tmp/data-only.rrd"
        PYTHONPATH="$REPO/scripts/lib" uv run --no-project --with "rerun-sdk==0.36.0" python - \
          "$import_tmp/data-only.rrd" "$proxy" "$rid" <<'PY'
import sys, rerun as rr
from tatbot_rerun import recording_name
path, proxy, rid = sys.argv[1:4]
rr.init("tatbot", recording_id=rid)
rr.connect_grpc(proxy)
rr.send_recording_name(recording_name(rid))
rr.log_file_from_path(path)
rr.disconnect()
print("forwarded", path)
PY
        )
        ;;
      *) echo "unsupported file type: $target (.rrd, .wxtl, or a run dir)" >&2; exit 2 ;;
    esac
    ;;
  install)
    need_server_node
    VISIOND="${VISIOND:-$REPO/rust/target/release/tatbot-visiond}"
    [ -x "$VISIOND" ] && "$VISIOND" send-blueprint --help >/dev/null 2>&1 || {
      echo 'build the fixed-blueprint sender: cargo build --manifest-path rust/Cargo.toml -p tatbot-visiond --features rerun --release' >&2
      exit 1
    }
    sudo install -m 644 "$REPO/config/systemd/tatbot-viewer@.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    # Restart once so this installed generation runs its ExecStartPost and gets
    # the canonical blueprint. The Python CLI can briefly outlive systemd's
    # main process; wait for both listeners to disappear before starting the
    # replacement so an install does not rely on RestartSec for recovery.
    sudo systemctl enable "tatbot-viewer@$USER.service"
    sudo systemctl stop "tatbot-viewer@$USER.service"
    for _ in $(seq 1 40); do
      if ! ss -H -ltn "sport = :$RERUN_PORT" 2>/dev/null | grep -q . \
          && ! ss -H -ltn "sport = :${RERUN_WEB_PORT:-9090}" 2>/dev/null | grep -q .; then
        break
      fi
      sleep 0.25
    done
    if ss -H -ltn "sport = :$RERUN_PORT" 2>/dev/null | grep -q . \
        || ss -H -ltn "sport = :${RERUN_WEB_PORT:-9090}" 2>/dev/null | grep -q .; then
      echo "Rerun did not release ports $RERUN_PORT and ${RERUN_WEB_PORT:-9090}" >&2
      sudo -n ss -H -ltnp "( sport = :$RERUN_PORT or sport = :${RERUN_WEB_PORT:-9090} )" >&2 || true
      exit 1
    fi
    sudo systemctl start "tatbot-viewer@$USER.service"
    ;;
  serve)
    if rerun_session::server_pid >/dev/null; then
      echo "a Rerun server already listens on :$RERUN_PORT" >&2; exit 1
    fi
    mapfile -t serve < <(serve_args)
    exec "$RERUN_BIN" "${serve[@]}" --bind 127.0.0.1 --port "$RERUN_PORT" --server-memory-limit "$SERVER_MEMORY_LIMIT"
    ;;
  -h|--help|help) usage ;;
  *) echo "unknown viewer command: $cmd" >&2; usage >&2; exit 2 ;;
esac
