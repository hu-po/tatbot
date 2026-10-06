#!/usr/bin/env bash
# The fleet Rerun viewer (docs/vision.md). Source this file.
#
# One viewer per FLEET: the node with role `rerun-server` in config/nodes.json
# runs a headless gRPC server (`rerun --serve-grpc`, port 9876); every other
# node streams to it and may attach a capped window (`rerun --connect`). Workflows call
# `rerun_session::ensure_viewer` and point their producers at `$RERUN_PROXY`;
# they never start, own, or stop a viewer of their own. `tatbot viewer
# start|stop|status|open|view` drives the same functions (scripts/viewer.sh).
#
# Resolution order for the proxy: TATBOT_RERUN_CONNECT, else the rerun-server
# node's `lan` address, else (on that node, or with TATBOT_RERUN_LOCAL=1, or
# with no server configured) a local server on this node.
#
# MEMORY (a viewer OOM froze a workstation on 2026-08-20): the viewer keeps every row
# until --memory-limit, then drops the oldest; the server buffers up to
# --server-memory-limit for late-joining windows. Both caps are mandatory here
# and every producer carries its own rate cap (`scripts/check rerun-caps`).
# The window cap is 1 GB because a window shares an operator node with
# everything else (measured 2026-08-30: RSS ~3x the raw payload).

RERUN_PORT="${RERUN_PORT:-9876}"
VIEWER_MEMORY_LIMIT="${VIEWER_MEMORY_LIMIT:-1GB}"
SERVER_MEMORY_LIMIT="${SERVER_MEMORY_LIMIT:-256MB}"
RERUN_RUNTIME_DIR="${RERUN_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/tmp}/tatbot-viewer}"
# RERUN_BIN picks the binary. SDK and viewer must be the same version: every
# Rerun in the project is 0.36.0 (since 2026-09-02), and `tatbot viewer status`
# prints the server's version next to the producers' pins so drift is visible.
RERUN_BIN="${RERUN_BIN:-$(command -v rerun 2>/dev/null || echo "$HOME/.local/bin/rerun")}"
RERUN_SESSION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

rerun_session::listener_pids() {
  local port="$1"
  ss -H -ltnp "sport = :$port" 2>/dev/null \
    | grep -o 'pid=[0-9][0-9]*' \
    | cut -d= -f2 \
    | sort -u
}

rerun_session::process_is_viewer() {
  local pid="$1" comm cmdline
  [ -r "/proc/$pid/comm" ] && [ -r "/proc/$pid/cmdline" ] || return 1
  comm="$(tr -d '\n' < "/proc/$pid/comm")"
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [ "$comm" = rerun ] || [[ "$cmdline" =~ (^|[/[:space:]])rerun([[:space:]]|$) ]]
}

# The pid of the Rerun process listening on the proxy port, if any.
rerun_session::server_pid() {
  local port="${1:-$RERUN_PORT}" pid
  for pid in $(rerun_session::listener_pids "$port"); do
    rerun_session::process_is_viewer "$pid" && { echo "$pid"; return 0; }
  done
  return 1
}

# Stop whatever Rerun listens on the port (an unrelated listener is a hard error).
rerun_session::release_port() {
  local port="${1:-$RERUN_PORT}" pid
  local -a pids=()
  mapfile -t pids < <(rerun_session::listener_pids "$port")
  if [ "${#pids[@]}" -eq 0 ]; then
    if ss -H -ltn "sport = :$port" 2>/dev/null | grep -q .; then
      echo "port $port is busy, but its owner is not visible:" >&2
      ss -H -ltnp "sport = :$port" >&2 || true
      return 1
    fi
    return 0
  fi
  for pid in "${pids[@]}"; do
    if ! rerun_session::process_is_viewer "$pid"; then
      echo "refusing to stop non-Rerun listener pid $pid on port $port:" >&2
      ps -o pid=,comm=,args= -p "$pid" >&2 || true
      return 1
    fi
  done
  kill "${pids[@]}" 2>/dev/null || true
  local attempt
  for ((attempt = 0; attempt < 5; attempt++)); do
    ss -H -ltn "sport = :$port" 2>/dev/null | grep -q . || return 0
    sleep 1
  done
  echo "Rerun did not release port $port:" >&2
  ss -H -ltnp "sport = :$port" >&2 || true
  return 1
}

# This host's rig-LAN address: one implementation, tatbot_cli.nodes.lan_ip
# (the `__rig__` subnet in config/nodes.json, or TATBOT_RERUN_LAN_IP). The old
# literal 192.168.1. match picked the viewer node's Wi-Fi lease after the rig
# moved to its own subnet.
rerun_session::lan_ip() {
  PYTHONPATH="$RERUN_SESSION_REPO/scripts/lib" python3 -c \
    'import sys; from pathlib import Path; from tatbot_cli import nodes; print(nodes.lan_ip(Path(sys.argv[1])) or "")' \
    "$RERUN_SESSION_REPO" 2>/dev/null
}

# The proxy URL producers on other nodes dial: this host's LAN address.
rerun_session::proxy_url() {
  local ip
  ip="$(rerun_session::lan_ip)"
  echo "rerun+http://${ip:-127.0.0.1}:${1:-$RERUN_PORT}/proxy"
}

# <prefix>-<UTC stamp>, the same clock as a run id and the viewer's time panel;
# the viewer names the recording from it (tatbot_rerun.recording_name).
rerun_session::recording_id() { echo "${1:?prefix}-$(date -u +%Y%m%dT%H%M%SZ)"; }

# "<node> <lan ip>" of the fleet viewer (config/nodes.json role rerun-server), or nothing.
rerun_session::fleet_server() {
  python3 - "$RERUN_SESSION_REPO" <<'PY' 2>/dev/null
import json, re, sys
from pathlib import Path
path = Path(sys.argv[1]) / "config" / "nodes.json"
if path.is_file():
    data = json.loads(re.sub(r"^\s*//.*$", "", path.read_text(), flags=re.M))
    for name, rec in data.items():
        if isinstance(rec, dict) and "rerun-server" in rec.get("roles", []) and rec.get("lan"):
            print(name, rec["lan"]); break
PY
}

# True when this host is the fleet viewer node (by hostname or LAN address).
rerun_session::is_server_node() {
  local server node lan
  server="$(rerun_session::fleet_server)" || true
  [ -n "$server" ] || return 1
  node="${server%% *}"; lan="${server##* }"
  [ "$(hostname -s 2>/dev/null || hostname)" = "$node" ] && return 0
  ip -4 -o addr show 2>/dev/null | grep -q " $lan/"
}

# The proxy producers dial: TATBOT_RERUN_CONNECT, else the fleet server, else
# this host (local mode). Never empty.
rerun_session::fleet_proxy() {
  local port="${1:-$RERUN_PORT}" server
  if [ -n "${TATBOT_RERUN_CONNECT:-}" ]; then echo "$TATBOT_RERUN_CONNECT"; return 0; fi
  if [ "${TATBOT_RERUN_LOCAL:-0}" != 1 ] && ! rerun_session::is_server_node; then
    server="$(rerun_session::fleet_server)" || true
    if [ -n "$server" ]; then echo "rerun+http://${server##* }:$port/proxy"; return 0; fi
  fi
  rerun_session::proxy_url "$port"
}

# TCP reachability of a proxy URL (3 s), so a launcher fails fast and clearly.
rerun_session::reachable() {
  local url="$1" hostport host port
  hostport="${url#*://}"; hostport="${hostport%%/*}"
  host="${hostport%%:*}"; port="${hostport##*:}"
  timeout 3 bash -c "exec 3<>/dev/tcp/$host/$port" 2>/dev/null
}

rerun_session::_pid_alive() {
  local file="$1" pid
  [ -r "$file" ] || return 1
  pid="$(cat "$file" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && rerun_session::process_is_viewer "$pid"
}

# Start the headless gRPC server if nothing Rerun listens on the port yet.
# Detached (setsid) so it outlives the workflow that started it.
rerun_session::start_server() {
  local server_memory_limit="${1:-$SERVER_MEMORY_LIMIT}" port="${2:-$RERUN_PORT}"
  case "$server_memory_limit" in ""|0|0B|0GB|0MB) echo "SERVER_MEMORY_LIMIT must be a real cap" >&2; return 2 ;; esac
  if rerun_session::server_pid "$port" >/dev/null; then
    return 0
  fi
  if ss -H -ltn "sport = :$port" 2>/dev/null | grep -q .; then
    echo "port $port is held by something that is not Rerun:" >&2
    ss -H -ltnp "sport = :$port" >&2 || true
    return 1
  fi
  mkdir -p "$RERUN_RUNTIME_DIR"
  local -a serve=()
  # TATBOT_VIEWER_WEB=1 also serves the browser viewer (a phone in the lab);
  # browser tabs ignore the memory cap, so it is opt-in.
  if [ "${TATBOT_VIEWER_WEB:-0}" = 1 ]; then
    serve=(--serve-web --web-viewer-port "${RERUN_WEB_PORT:-9090}")
  else
    serve=(--serve-grpc)
  fi
  setsid "$RERUN_BIN" "${serve[@]}" --bind "${TATBOT_RERUN_BIND:-127.0.0.1}" --port "$port" \
    --server-memory-limit "$server_memory_limit" \
    > "$RERUN_RUNTIME_DIR/server.log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$RERUN_RUNTIME_DIR/server.pid"
  "$RERUN_BIN" --version 2>/dev/null | head -1 > "$RERUN_RUNTIME_DIR/server.version" || true
  local attempt
  for ((attempt = 0; attempt < 20; attempt++)); do
    ss -H -ltn "sport = :$port" 2>/dev/null | grep -q . && return 0
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Rerun server exited before listening on port $port:" >&2
      tail -n 5 "$RERUN_RUNTIME_DIR/server.log" >&2 || true
      return 1
    fi
    sleep 0.5
  done
  echo "Rerun server did not listen on port $port within 10 seconds" >&2
  kill "$pid" 2>/dev/null || true
  return 1
}

rerun_session::has_display() {
  [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]
}

# Over SSH on a node with a desktop: borrow its X session for the window.
rerun_session::_borrow_display() {
  rerun_session::has_display && return 0
  local xauth
  xauth="/run/user/$(id -u)/gdm/Xauthority"
  if [ -e "$xauth" ]; then
    export DISPLAY=:0 XAUTHORITY="$xauth"
    return 0
  fi
  return 1
}

# Attach a native window to a server (one per node; a second call is a no-op).
# Only where this shell already has a display: a launcher run over ssh on a
# headless producer node must never borrow that node's console session
# — the operator's window is the viewer. `tatbot viewer
# open` passes `borrow` to use the node's own desktop deliberately.
# Returns 1 without a display: the server keeps buffering for a later `open`.
rerun_session::open_gui() {
  local memory_limit="${1:-$VIEWER_MEMORY_LIMIT}" proxy="${2:-rerun+http://127.0.0.1:$RERUN_PORT/proxy}" borrow="${3:-}"
  case "$memory_limit" in ""|0|0B|0GB|0MB) echo "VIEWER_MEMORY_LIMIT must be a real cap" >&2; return 2 ;; esac
  mkdir -p "$RERUN_RUNTIME_DIR"
  if rerun_session::_pid_alive "$RERUN_RUNTIME_DIR/gui.pid"; then
    return 0
  fi
  # One window per node, whoever started it: a window from an earlier shell
  # whose pid file is gone, anything. Match the binary
  # (bracketed so this shell's own argv never matches) attached to any proxy.
  if pgrep -f "rerun_cli/[r]erun .* --memory-limit " >/dev/null 2>&1 || pgrep -f "bin/[r]erun --connect " >/dev/null 2>&1; then
    echo "a Rerun window already runs on this node; not opening a second one" >&2
    return 0
  fi
  if [ "$borrow" = borrow ]; then rerun_session::_borrow_display || return 1; else rerun_session::has_display || return 1; fi
  setsid "$RERUN_BIN" --connect "$proxy" \
    --memory-limit "$memory_limit" --hide-welcome-screen \
    > "$RERUN_RUNTIME_DIR/gui.log" 2>&1 < /dev/null &
  echo "$!" > "$RERUN_RUNTIME_DIR/gui.pid"
  ( sleep 2.5; rerun_session::maximize_window ) >/dev/null 2>&1 &
  return 0
}

# Maximize the viewer window where an X11 window manager can be reached
# (rerun-cli has no maximize flag); silently does nothing elsewhere.
rerun_session::maximize_window() {
  [ -z "${WAYLAND_DISPLAY:-}" ] && [ -n "${DISPLAY:-}" ] || return 0
  command -v xdotool >/dev/null 2>&1 || return 0
  local wid=""
  for _ in $(seq 1 40); do
    wid=$(xdotool search --class rerun 2>/dev/null | head -n1)
    [ -n "$wid" ] && break
    sleep 0.25
  done
  [ -n "$wid" ] || return 0
  xprop -id "$wid" _NET_WM_STATE 2>/dev/null | grep -q MAXIMIZED && return 0
  if command -v wmctrl >/dev/null 2>&1; then
    wmctrl -i -r "$wid" -b add,maximized_vert,maximized_horz 2>/dev/null && return 0
  fi
  for _ in 1 2 3; do
    xdotool windowactivate --sync "$wid" 2>/dev/null
    sleep 0.4
    xdotool key --clearmodifiers super+Up 2>/dev/null
    sleep 0.6
    xprop -id "$wid" _NET_WM_STATE 2>/dev/null | grep -q MAXIMIZED && return 0
  done
  return 0
}

# What every launcher calls. On the fleet viewer node (or in local mode):
# server up here, window attached when there is a display. Elsewhere: the
# fleet server must answer; NO window opens here (a window costs the
# launching node CPU/GPU — operator decision 2026-09-02; `tatbot viewer open`
# attaches one deliberately).
# Either way RERUN_PROXY is exported for the producers and one budget line
# is printed.
rerun_session::ensure_viewer() {
  local memory_limit="${1:-$VIEWER_MEMORY_LIMIT}" server_memory_limit="${2:-$SERVER_MEMORY_LIMIT}" port="${3:-$RERUN_PORT}"
  local gui="window attached" server
  RERUN_PROXY="$(rerun_session::fleet_proxy "$port")"
  export RERUN_PROXY
  if rerun_session::is_server_node || [ "${TATBOT_RERUN_LOCAL:-0}" = 1 ] || [ -z "$(rerun_session::fleet_server)" ] \
     && [ -z "${TATBOT_RERUN_CONNECT:-}" ]; then
    rerun_session::start_server "$server_memory_limit" "$port" || return 1
    rerun_session::open_gui "$memory_limit" "rerun+http://127.0.0.1:$port/proxy" \
      || gui="no display here; attach one with: tatbot viewer open"
    echo "=== Rerun viewer $RERUN_PROXY (this node; window cap $memory_limit, server buffer $server_memory_limit; $gui)"
    return 0
  fi
  if ! rerun_session::reachable "$RERUN_PROXY"; then
    server="$(rerun_session::fleet_server)" || true
    echo "fleet Rerun viewer $RERUN_PROXY is not answering (${server%% *}: tatbot viewer status;" \
         "start it with: tatbot --on ${server%% *} viewer start — or TATBOT_RERUN_LOCAL=1 for a viewer here)" >&2
    return 1
  fi
  server="$(rerun_session::fleet_server)" || true
  echo "=== Rerun viewer $RERUN_PROXY (fleet: ${server%% *}; a window here: tatbot viewer open)"
}

# Stop this node's window, and the server if one runs here (`tatbot viewer stop`).
rerun_session::stop_viewer() {
  local port="${1:-$RERUN_PORT}" f pid
  for f in gui server; do
    if [ -r "$RERUN_RUNTIME_DIR/$f.pid" ]; then
      pid="$(cat "$RERUN_RUNTIME_DIR/$f.pid")"
      [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
      rm -f "$RERUN_RUNTIME_DIR/$f.pid"
    fi
  done
  pkill -f -- "[r]erun --connect rerun\+http://[^ ]*:$port/proxy" 2>/dev/null || true
  if rerun_session::server_pid "$port" >/dev/null; then
    rerun_session::release_port "$port"
  fi
}

rerun_session::status() {
  local port="${1:-$RERUN_PORT}" pid server proxy
  server="$(rerun_session::fleet_server)" || true
  proxy="$(rerun_session::fleet_proxy "$port")"
  if [ -n "$server" ] && ! rerun_session::is_server_node; then
    if rerun_session::reachable "$proxy"; then
      echo "fleet:  ${server%% *} answers at $proxy (web viewer on http://${server##* }:${RERUN_WEB_PORT:-9090})"
    else
      echo "fleet:  ${server%% *} NOT answering at $proxy (tatbot --on ${server%% *} viewer start)"
    fi
  fi
  if pid="$(rerun_session::server_pid "$port")"; then
    echo "server: pid $pid on :$port ($(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -oE -- '--serve-[a-z]+|--server-memory-limit [^ ]+' | tr '\n' ' '))"
    echo "        $(cat "$RERUN_RUNTIME_DIR/server.version" 2>/dev/null || "$RERUN_BIN" --version 2>/dev/null | head -1)"
    echo "        proxy $(rerun_session::proxy_url "$port"); up $(ps -o etime= -p "$pid" | tr -d ' ')"
  else
    echo "server: not running (tatbot viewer start)"
  fi
  if rerun_session::_pid_alive "$RERUN_RUNTIME_DIR/gui.pid"; then
    pid="$(cat "$RERUN_RUNTIME_DIR/gui.pid")"
    echo "window: pid $pid ($(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -oE -- '--memory-limit [^ ]+'))"
  else
    echo "window: none (tatbot viewer open)"
  fi
  [ -d "$RERUN_RUNTIME_DIR" ] && echo "logs:   $RERUN_RUNTIME_DIR/{server,gui}.log"
  return 0
}
