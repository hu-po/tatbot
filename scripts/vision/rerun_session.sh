#!/usr/bin/env bash
# Shared lifecycle helpers for Tatbot live Rerun workflows.
#
# Source this file. It deliberately only stops the exact listener PID after
# proving that PID is a Rerun viewer; an unrelated service on the port is a
# hard error. Callers own remote producer cleanup because their recording ids
# and socket paths are the safest process selectors.

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

rerun_session::release_port() {
  local port="${1:-9876}" pid
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
  echo "port $port has an earlier Rerun viewer; stopping pid(s) ${pids[*]}"
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

rerun_session::start_viewer() {
  local memory_limit="${1:-3GB}"
  local server_memory_limit="${2:-512MB}"
  local port="${3:-9876}"
  rerun_session::release_port "$port"
  command rerun --bind 0.0.0.0 --port "$port" \
    --memory-limit "$memory_limit" \
    --server-memory-limit "$server_memory_limit" &
  RERUN_VIEWER_PID=$!
  local attempt
  for ((attempt = 0; attempt < 20; attempt++)); do
    if ss -H -ltn "sport = :$port" 2>/dev/null | grep -q .; then
      export RERUN_VIEWER_PID
      return 0
    fi
    if ! kill -0 "$RERUN_VIEWER_PID" 2>/dev/null; then
      wait "$RERUN_VIEWER_PID" 2>/dev/null || true
      echo "Rerun viewer exited before listening on port $port" >&2
      return 1
    fi
    sleep 0.5
  done
  echo "Rerun viewer did not listen on port $port within 10 seconds" >&2
  kill "$RERUN_VIEWER_PID" 2>/dev/null || true
  return 1
}

rerun_session::stop_viewer() {
  local pid="${1:-${RERUN_VIEWER_PID:-}}"
  [ -n "$pid" ] || return 0
  kill "$pid" 2>/dev/null || true
}

rerun_session::lan_ip() {
  ip -4 -o addr show \
    | awk '/192\.168\.1\./ {split($4, address, "/"); print address[1]; exit}'
}
