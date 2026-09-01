#!/usr/bin/env bash
# Background lifecycle for the design generator on THIS node (the CLI hops here).
#   scripts/inkgen_ctl.sh start [--port 8600]   nohup app.py, pidfile + log under ~/.cache/tatbot/inkgen, wait for /api/health
#   scripts/inkgen_ctl.sh stop                  SIGTERM the pid in the pidfile
#   scripts/inkgen_ctl.sh status [--port 8600]  pid + /api/health
#   scripts/inkgen_ctl.sh logs [-n 40]
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="${INKGEN_HOME:-$HOME/.cache/tatbot/inkgen}"
pidfile="$root/inkgen.pid"; log="$root/inkgen.log"
cmd="${1:-status}"; shift || true
port=8600; lines=40
while [ $# -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    -n) lines="$2"; shift 2 ;;
    *) echo "inkgen_ctl: unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$root"
alive() { [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; }
health() { curl -fsS --max-time 5 "http://127.0.0.1:$port/api/health" 2>/dev/null; }
case "$cmd" in
  start)
    if alive; then echo "inkgen: already running (pid $(cat "$pidfile")) on :$port"; health; echo; exit 0; fi
    nohup "$repo/scripts/inkgen_serve.sh" --port "$port" > "$log" 2>&1 &
    echo $! > "$pidfile"
    echo "inkgen: starting on $(hostname -s):$port (pid $!, log $log) — loading the model takes ~30 s, more on a first run"
    for _ in $(seq 1 120); do
      if out=$(health); then echo "inkgen: live — $out"; exit 0; fi
      alive || { echo "inkgen: exited early; last log lines:" >&2; tail -n 20 "$log" >&2; exit 5; }
      sleep 5
    done
    echo "inkgen: still not answering after 10 min; see $log" >&2; exit 5 ;;
  stop)
    if alive; then pid=$(cat "$pidfile"); pkill -TERM -P "$pid" 2>/dev/null || true; kill -TERM "$pid" 2>/dev/null || true; sleep 2; rm -f "$pidfile"; echo "inkgen: stopped (pid $pid)"; else rm -f "$pidfile"; echo "inkgen: not running"; fi ;;
  status)
    if alive; then echo "inkgen: running (pid $(cat "$pidfile")) on $(hostname -s):$port"; else echo "inkgen: not running on $(hostname -s)"; fi
    if out=$(health); then echo "health: $out"; else echo "health: no answer on :$port"; [ -f "$pidfile" ] && exit 1; fi ;;
  logs) tail -n "$lines" "$log" 2>/dev/null || echo "inkgen: no log at $log" ;;
  *) echo "inkgen_ctl: start|stop|status|logs" >&2; exit 2 ;;
esac
