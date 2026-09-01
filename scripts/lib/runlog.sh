#!/usr/bin/env bash
# tatbot run logging, shell side. SOURCE this file, do not execute it.
#
#   source "$REPO/scripts/lib/runlog.sh"
#   runlog::init rollout --set policy="$POLICY" --set server="$SERVER"
#   runlog::run <the command that used to follow `exec`>
#
# Everything the run prints lands in $RUN_DIR/console.log AND on the terminal.
#
# This library installs an EXIT trap and nothing else. It never traps INT or
# TERM, so the coordinated-landing traps in record_session.sh and the
# first-Ctrl+C-lands / repeats-swallowed semantics of il_client_shield.py keep
# working exactly as they did. Nothing here runs while the arm is moving.
#
# shellcheck shell=bash

RUNLOG_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNLOG_PY="${TATBOT_RUNLOG_PY:-$RUNLOG_LIB_DIR/tatbot_runlog.py}"
RUNLOG_PYTHON="${TATBOT_RUNLOG_PYTHON:-python3}"
RUNLOG_ACTIVE=0
RUN_DIR="${RUN_DIR:-}"
RUN_ID="${RUN_ID:-}"

# runlog::init <workflow> [--set key=value ...]
runlog::init() {
  [ "${TATBOT_RUNLOG:-1}" = 0 ] && return 0
  local workflow="$1"; shift
  # Python mints the id, writes meta.json, appends the index row and runs
  # retention. One implementation of those rules, not two — a second copy of
  # the delete logic in bash is how a 300 GB directory becomes an incident.
  if ! RUN_DIR="$("$RUNLOG_PYTHON" "$RUNLOG_PY" begin \
        --workflow "$workflow" --argv0 "${BASH_SOURCE[1]:-$0}" \
        --parent-pid "$$" "$@")"; then
    echo "runlog: unavailable — continuing without a run directory" >&2
    RUN_DIR=""
    return 0                     # a broken logger never stops a run
  fi
  RUN_ID="$(basename "$RUN_DIR")"
  export TATBOT_RUN_DIR="$RUN_DIR" TATBOT_RUN_ID="$RUN_ID" \
         TATBOT_RUN_WORKFLOW="$workflow" TATBOT_RUN_CONSOLE=shell \
         TATBOT_RUNLOG_PY="$RUNLOG_PY"
  RUNLOG_ACTIVE=1
  runlog::_capture
  trap 'runlog::finalize $?' EXIT
}

runlog::_capture() {
  # Keep the real terminal on fds 3/4 so finalize can still talk to a human
  # after the capture is torn down.
  exec 3>&1 4>&2
  # Process substitution, NOT `main | tee`: a pipeline makes every exit code a
  # PIPESTATUS puzzle and fights `set -e`.
  # tee -i, because a terminal Ctrl+C is delivered to the whole foreground
  # process group. Without -i, tee dies first and the landing narration — the
  # part you actually need in the log — never reaches the file.
  exec > >(exec tee -i -a "$RUN_DIR/console.log") 2>&1
  RUNLOG_TEE_PID=$!
}

# Bounded wait for tee to flush. Never unbounded: a teardown must not hang.
runlog::_reap_tee() {
  local pid="${RUNLOG_TEE_PID:-}" i=0
  [ -n "$pid" ] || return 0
  while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 50 ]; do
    sleep 0.1; i=$((i + 1))
  done
}

runlog::event() {   # runlog::event <kind> [key=value ...]
  [ "$RUNLOG_ACTIVE" = 1 ] || return 0
  local kind="$1"; shift
  "$RUNLOG_PYTHON" "$RUNLOG_PY" event --dir "$RUN_DIR" --kind "$kind" "$@" \
    >/dev/null 2>&1 || true
}

runlog::artifact() {   # runlog::artifact <path>
  [ "$RUNLOG_ACTIVE" = 1 ] || return 0
  "$RUNLOG_PYTHON" "$RUNLOG_PY" artifact --dir "$RUN_DIR" --path "$1" \
    >/dev/null 2>&1 || true
}

# runlog::path <name> [fallback] — an artifact path inside the run dir, or the
# fallback when there is no run (bench runs keep working unchanged).
runlog::path() {
  if [ "$RUNLOG_ACTIVE" = 1 ]; then echo "$RUN_DIR/$1"; else echo "${2:-}"; fi
}

# runlog::run <cmd...> — drop-in replacement for a trailing `exec`.
runlog::run() {
  local rc=0
  # No `exec`, so the EXIT trap survives to finalize the run. Ctrl+C still
  # reaches the child directly (the tty signals the whole foreground process
  # group) and bash defers its own handling until the foreground child
  # returns, so the staged->sleep landing always completes before we finalize.
  "$@" || rc=$?
  return "$rc"
}

runlog::finalize() {   # EXIT trap; idempotent
  local rc="${1:-$?}"
  [ "$RUNLOG_ACTIVE" = 1 ] || return 0
  RUNLOG_ACTIVE=0
  # Restore the terminal fds FIRST: closing tee's stdin is what makes it flush
  # and exit, and `end` must not stat console.log before that has happened.
  exec 1>&3 2>&4 3>&- 4>&-
  runlog::_reap_tee
  "$RUNLOG_PYTHON" "$RUNLOG_PY" end --dir "$RUN_DIR" --exit-code "$rc" || true
}
