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
# node's address on a direct cable to this node while that cable is live
# (config/nodes.json `links`, tatbot_cli.viewer_path), else its `lan` address,
# else (on that node, or with TATBOT_RERUN_LOCAL=1, or with no server
# configured) a local server on this node.
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
RERUN_PIN=0.36.0
RERUN_SESSION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The operator's window runs as this transient user unit (`tatbot viewer open`).
VIEWER_WINDOW_UNIT="${VIEWER_WINDOW_UNIT:-tatbot-viewer-window}"

# Exit 3 unless $RERUN_BIN reports the fleet pin: a unit installed without it
# restarts every few seconds forever on a missing binary.
rerun_session::require_pin() {
  local got
  got="$("$RERUN_BIN" --version 2>/dev/null | head -1 | awk '{print $2}')"
  [ "$got" = "$RERUN_PIN" ] && return 0
  echo "rerun ${got:-is not installed} at $RERUN_BIN; the fleet runs $RERUN_PIN: uv tool install 'rerun-sdk==$RERUN_PIN'" >&2
  return 3
}

# The Python of the pinned rerun tool: the `rerun` that `uv tool install`
# links is that tool venv's entry point, and the venv holds the SDK.
rerun_session::sdk_python() {
  local bin
  bin="$(readlink -f "$RERUN_BIN" 2>/dev/null)" || bin=""
  if [ -n "$bin" ] && [ -x "${bin%/*}/python" ]; then
    echo "${bin%/*}/python"
    return 0
  fi
  echo "no rerun SDK interpreter beside $RERUN_BIN: uv tool install 'rerun-sdk==$RERUN_PIN'" >&2
  return 1
}

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

# True when this host is the fleet viewer node (by node name or LAN address).
rerun_session::is_server_node() {
  local server node lan
  server="$(rerun_session::fleet_server)" || true
  [ -n "$server" ] || return 1
  node="${server%% *}"; lan="${server##* }"
  # shellcheck source=scripts/lib/nodes.sh
  declare -F tatbot_nodes::this_node >/dev/null || source "$RERUN_SESSION_REPO/scripts/lib/nodes.sh"
  [ "$(tatbot_nodes::this_node)" = "$node" ] && return 0
  ip -4 -o addr show 2>/dev/null | grep -q " $lan/"
}

# tatbot_cli.viewer_path: `endpoint [PORT]` prints "<address>\t<path>" where
# this node dials the fleet server (its direct cable here when that routes off
# the rig interface and answers, else its rig address, with why); `describe
# ADDRESS` names the path to an address; `clients SS ADDRS` lists the hub's.
rerun_session::_viewer_path() {
  PYTHONPATH="$RERUN_SESSION_REPO/scripts/lib" python3 -m tatbot_cli.viewer_path "$@" 2>/dev/null
}

# What a proxy URL rides from this node, in words (the direct link, the rig LAN).
rerun_session::proxy_path() {
  local host="${1#*://}"
  host="${host%%/*}"; host="${host%:*}"
  rerun_session::_viewer_path describe "$host"
}

# "<address>\t<path>" of the fleet server from this node; its rig address alone
# if the path cannot be worked out. Nothing without a fleet server.
rerun_session::fleet_endpoint() {
  local port="${1:-$RERUN_PORT}" server found
  server="$(rerun_session::fleet_server)" || true
  [ -n "$server" ] || return 1
  found="$(rerun_session::_viewer_path endpoint "$port")" || true
  echo "${found:-${server##* }}"
}

# The proxy producers dial: TATBOT_RERUN_CONNECT, else the fleet server (on its
# direct cable to this node while that is live), else this host (local mode).
# Never empty.
rerun_session::fleet_proxy() {
  local port="${1:-$RERUN_PORT}" found
  if [ -n "${TATBOT_RERUN_CONNECT:-}" ]; then echo "$TATBOT_RERUN_CONNECT"; return 0; fi
  if [ "${TATBOT_RERUN_LOCAL:-0}" != 1 ] && ! rerun_session::is_server_node; then
    found="$(rerun_session::fleet_endpoint "$port")" || true
    if [ -n "$found" ]; then echo "rerun+http://${found%%$'\t'*}:$port/proxy"; return 0; fi
  fi
  rerun_session::proxy_url "$port"
}

# The proxy a running window process dials (its --connect), or nothing.
rerun_session::_connect_of() {
  [ -n "${1:-}" ] && [ -r "/proc/$1/cmdline" ] || return 0
  tr '\0' ' ' < "/proc/$1/cmdline" | grep -oE -- '--connect [^ ]+' | cut -d' ' -f2
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

# Over SSH on a node with a desktop: borrow its X session for the window,
# as the desktop's user manager records it (GNOME imports DISPLAY and
# XAUTHORITY there; the station's panel is not always :0), else the console's.
rerun_session::_borrow_display() {
  rerun_session::has_display && return 0
  local xauth line
  while IFS= read -r line; do
    case "$line" in DISPLAY=?*|WAYLAND_DISPLAY=?*|XAUTHORITY=?*) export "${line?}" ;; esac
  done < <(systemctl --user show-environment 2>/dev/null)
  rerun_session::has_display && return 0
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
# A recorded .rrd/.wxtl in a window of its own, capped like every window and maximized once it maps: the
# operator shell's `rerun file.rrd` (scripts/rerun_maximized.bash). Its file, not the fleet viewer.
rerun_session::view_file() {
  [ "$#" -ge 1 ] || { echo "usage: rerun_session::view_file <file.rrd|file.wxtl> [rerun args]" >&2; return 2; }
  rerun_session::_borrow_display || true
  rerun_session::window_state maximized >/dev/null 2>&1 &
  "$RERUN_BIN" --memory-limit "$VIEWER_MEMORY_LIMIT" "$@"
}

rerun_session::open_gui() {
  local memory_limit="${1:-$VIEWER_MEMORY_LIMIT}" proxy="${2:-rerun+http://127.0.0.1:$RERUN_PORT/proxy}" borrow="${3:-}"
  local state="${4:-maximized}"
  case "$memory_limit" in ""|0|0B|0GB|0MB) echo "VIEWER_MEMORY_LIMIT must be a real cap" >&2; return 2 ;; esac
  mkdir -p "$RERUN_RUNTIME_DIR"
  if rerun_session::_pid_alive "$RERUN_RUNTIME_DIR/gui.pid"; then
    return 0
  fi
  rerun_session::_window_running && return 0
  if [ "$borrow" = borrow ]; then rerun_session::_borrow_display || return 1; else rerun_session::has_display || return 1; fi
  setsid "$RERUN_BIN" --connect "$proxy" \
    --memory-limit "$memory_limit" --hide-welcome-screen \
    > "$RERUN_RUNTIME_DIR/gui.log" 2>&1 < /dev/null &
  echo "$!" > "$RERUN_RUNTIME_DIR/gui.pid"
  rerun_session::window_state "$state" >/dev/null 2>&1 &
  return 0
}

# One window per node, whoever started it: a window from an earlier shell
# whose pid file is gone, anything. Match the binary
# (bracketed so this shell's own argv never matches) attached to any proxy.
rerun_session::_window_running() {
  if pgrep -f "rerun_cli/[r]erun .* --memory-limit " >/dev/null 2>&1 || pgrep -f "bin/[r]erun --connect " >/dev/null 2>&1; then
    echo "a Rerun window already runs on this node; not opening a second one" >&2
    return 0
  fi
  return 1
}

# The window unit's hard ceiling: the viewer's own --memory-limit plus 1 GiB
# for the process itself (4GB -> 5120M). An absolute size only; nothing for a
# percentage or a zero cap.
rerun_session::window_memory_max() {
  awk -v s="$1" 'BEGIN {
    if (!match(s, /^[0-9]+(\.[0-9]+)?/)) exit 1
    n = substr(s, 1, RLENGTH); u = toupper(substr(s, RLENGTH + 1)); sub(/^ +/, "", u); sub(/I?B$/, "", u)
    f = (u == "") ? 1 / 1048576 : (u == "K") ? 1 / 1024 : (u == "M") ? 1 : (u == "G") ? 1024 : (u == "T") ? 1048576 : -1
    if (f < 0 || n * f < 1) exit 1
    printf "%dM\n", n * f + 1024.5
  }'
}

# `tatbot viewer open`: the window as the transient user unit
# $VIEWER_WINDOW_UNIT. The viewer drops its oldest rows at --memory-limit; the
# unit's MemoryMax is the hard ceiling for the whole process. A running unit
# or window is left as it is, unless the third argument is `restart`: a window
# never reconnects to a hub that restarted, so it is closed and opened again.
# The fourth is the window's state once it maps, `maximized` (the default) or
# `fullscreen` (rerun_session::window_state); a running unit is made
# fullscreen too, and one that already is is not touched. Without a user
# manager (no systemd login session) the window starts detached instead.
# 1 = no display here.
rerun_session::open_window() {
  local memory_limit="$1" proxy="$2" state="${4:-maximized}" max var
  max="$(rerun_session::window_memory_max "$memory_limit")" || {
    echo "--memory-limit wants an absolute size such as 4GB, not '$memory_limit'" >&2
    return 2
  }
  [ "${3:-}" = restart ] && rerun_session::close_window
  if ! command -v systemd-run >/dev/null 2>&1 || ! systemctl --user show-environment >/dev/null 2>&1; then
    rerun_session::open_gui "$memory_limit" "$proxy" borrow "$state"
    return
  fi
  if systemctl --user is-active --quiet "$VIEWER_WINDOW_UNIT"; then
    echo "the window already runs as $VIEWER_WINDOW_UNIT; leaving it as it is" >&2
    local dials
    dials="$(rerun_session::_connect_of "$(systemctl --user show -p MainPID --value "$VIEWER_WINDOW_UNIT" 2>/dev/null)")"
    if [ -n "$dials" ] && [ "$dials" != "$proxy" ]; then
      echo "it dials $dials, not $proxy: tatbot viewer open --restart moves it" >&2
    fi
    if [ "$state" = fullscreen ] && rerun_session::_borrow_display; then rerun_session::window_state fullscreen; fi
    return 0
  fi
  rerun_session::_window_running && return 0
  rerun_session::_borrow_display || return 1
  local -a env=()
  for var in DISPLAY WAYLAND_DISPLAY XAUTHORITY; do
    [ -n "${!var:-}" ] && env+=("--setenv=$var=${!var}")
  done
  mkdir -p "$RERUN_RUNTIME_DIR"
  systemctl --user reset-failed "$VIEWER_WINDOW_UNIT" >/dev/null 2>&1 || true
  systemd-run --user --quiet --collect --unit "$VIEWER_WINDOW_UNIT" -p MemoryMax="$max" \
    -p StandardOutput="append:$RERUN_RUNTIME_DIR/gui.log" -p StandardError="append:$RERUN_RUNTIME_DIR/gui.log" \
    "${env[@]}" "$RERUN_BIN" --connect "$proxy" --memory-limit "$memory_limit" --hide-welcome-screen || return 2
  rerun_session::window_state "$state" >/dev/null 2>&1 &
  return 0
}

# The browser viewer on the fleet server, opened on the standing proxy:
# http://<lan>:9090/?url=rerun%2Bhttp%3A%2F%2F<lan>%3A9876%2Fproxy
rerun_session::web_url() {
  local server lan
  server="$(rerun_session::fleet_server)" || true
  [ -n "$server" ] || return 1
  lan="${server##* }"
  echo "http://$lan:${RERUN_WEB_PORT:-9090}/?url=rerun%2Bhttp%3A%2F%2F$lan%3A${1:-$RERUN_PORT}%2Fproxy"
}

# Put this node's viewer window in a state once it maps: `maximized` or
# `fullscreen` (rerun-cli has neither flag). scripts/lib/x11_window.py sends the
# window manager the EWMH request, so no key is typed and no focus is needed; a
# window already in the state is not asked again. Fullscreen is the demo page:
# the focus goes back to the window that held it (the station's agent terminal),
# and the operator alt-tabs to the viewer. Silently nothing off X11.
rerun_session::window_state() {
  [ -z "${WAYLAND_DISPLAY:-}" ] && [ -n "${DISPLAY:-}" ] || return 0
  local -a keep=()
  [ "${1:-maximized}" = fullscreen ] && keep=(--keep-focus)
  python3 "$RERUN_SESSION_REPO/scripts/lib/x11_window.py" rerun "${1:-maximized}" "${keep[@]}"
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
# This node's window (the unit, or one started detached), never a server.
rerun_session::close_window() {
  local port="${1:-$RERUN_PORT}" pid
  systemctl --user stop "$VIEWER_WINDOW_UNIT" >/dev/null 2>&1 || true
  # A stopped transient unit can stay loaded a moment, and systemd-run then refuses its name ("already loaded or
  # has a fragment file"): `viewer open --restart` left the station with no window on 2026-10-19. Wait it out.
  for _ in $(seq 1 20); do
    [ "$(systemctl --user show -p LoadState --value "$VIEWER_WINDOW_UNIT" 2>/dev/null)" = loaded ] || break
    systemctl --user reset-failed "$VIEWER_WINDOW_UNIT" >/dev/null 2>&1 || true
    sleep 0.25
  done
  if [ -r "$RERUN_RUNTIME_DIR/gui.pid" ]; then
    pid="$(cat "$RERUN_RUNTIME_DIR/gui.pid")"
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    rm -f "$RERUN_RUNTIME_DIR/gui.pid"
  fi
  pkill -f -- "[r]erun --connect rerun\+http://[^ ]*:$port/proxy" 2>/dev/null || true
  for _ in $(seq 1 20); do
    pgrep -f -- "[r]erun --connect rerun\+http://[^ ]*:$port/proxy" >/dev/null 2>&1 || return 0
    sleep 0.25
  done
}

rerun_session::stop_viewer() {
  local port="${1:-$RERUN_PORT}" pid
  rerun_session::close_window "$port"
  if [ -r "$RERUN_RUNTIME_DIR/server.pid" ]; then
    pid="$(cat "$RERUN_RUNTIME_DIR/server.pid")"
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    rm -f "$RERUN_RUNTIME_DIR/server.pid"
  fi
  if rerun_session::server_pid "$port" >/dev/null; then
    rerun_session::release_port "$port"
  fi
}

# Status lines: what this node's window dials and over which path, and whether
# that is the fleet path (the second argument) a fresh window would take.
rerun_session::_window_dials() {
  local url
  url="$(rerun_session::_connect_of "$1")"
  [ -n "$url" ] || return 0
  echo "        dials $url ($(rerun_session::proxy_path "$url"))"
  if [ -n "${2:-}" ] && [ "$url" != "$2" ]; then
    echo "        not the fleet path above: tatbot viewer open --restart moves it"
  fi
}

# Status lines on the hub: who is connected, over which path.
rerun_session::_hub_clients() {
  local port="$1" lines line lead="clients:"
  lines="$(rerun_session::_viewer_path clients "$(ss -H -tn state established "( sport = :$port )" 2>/dev/null)" \
    "$(ip -4 -o addr 2>/dev/null)")" || return 0
  [ -n "$lines" ] || lines="none"
  while IFS= read -r line; do
    echo "        $lead $line"
    lead="        "
  done <<<"$lines"
}

rerun_session::status() {
  local port="${1:-$RERUN_PORT}" pid server proxy="" path="" found max
  server="$(rerun_session::fleet_server)" || true
  if [ -n "$server" ] && ! rerun_session::is_server_node; then
    if [ -n "${TATBOT_RERUN_CONNECT:-}" ]; then
      proxy="$TATBOT_RERUN_CONNECT"; path="TATBOT_RERUN_CONNECT"
    else
      found="$(rerun_session::fleet_endpoint "$port")" || true
      proxy="rerun+http://${found%%$'\t'*}:$port/proxy"
      [[ "$found" == *$'\t'* ]] && path="${found#*$'\t'}"
    fi
    if rerun_session::reachable "$proxy"; then
      echo "fleet:  ${server%% *} answers at $proxy${path:+ ($path)}"
    else
      echo "fleet:  ${server%% *} NOT answering at $proxy${path:+ ($path)}; tatbot --on ${server%% *} viewer start"
    fi
  fi
  if pid="$(rerun_session::server_pid "$port")"; then
    echo "server: pid $pid on :$port ($(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -oE -- '--serve-[a-z]+|--server-memory-limit [^ ]+' | tr '\n' ' '))"
    echo "        $(cat "$RERUN_RUNTIME_DIR/server.version" 2>/dev/null || "$RERUN_BIN" --version 2>/dev/null | head -1)"
    echo "        proxy $(rerun_session::proxy_url "$port"); up $(ps -o etime= -p "$pid" | tr -d ' ')"
    rerun_session::_hub_clients "$port"
  else
    echo "server: not running (tatbot viewer start)"
  fi
  [ -n "$server" ] && echo "web:    $(rerun_session::web_url "$port")"
  if systemctl --user is-active --quiet "$VIEWER_WINDOW_UNIT" 2>/dev/null; then
    pid="$(systemctl --user show -p MainPID --value "$VIEWER_WINDOW_UNIT" 2>/dev/null)"
    max="$(systemctl --user show -p MemoryMax --value "$VIEWER_WINDOW_UNIT" 2>/dev/null)"
    [[ "$max" =~ ^[0-9]+$ ]] && max="$((max / 1048576))M"
    echo "window: $VIEWER_WINDOW_UNIT pid $pid ($(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -oE -- '--memory-limit [^ ]+'), MemoryMax $max)"
    rerun_session::_window_dials "$pid" "$proxy"
  elif rerun_session::_pid_alive "$RERUN_RUNTIME_DIR/gui.pid"; then
    pid="$(cat "$RERUN_RUNTIME_DIR/gui.pid")"
    echo "window: pid $pid ($(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -oE -- '--memory-limit [^ ]+'))"
    rerun_session::_window_dials "$pid" "$proxy"
  else
    echo "window: none (tatbot viewer open)"
  fi
  [ -d "$RERUN_RUNTIME_DIR" ] && echo "logs:   $RERUN_RUNTIME_DIR/{server,gui}.log"
  return 0
}
