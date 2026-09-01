#!/usr/bin/env bash
# Vite dev server for web/inkmap. Installs node_modules on first run and can
# point the app at a generator without touching any file:
#   scripts/inkmap_dev.sh [--api http://<inkgen-host>:8600] [--host 0.0.0.0] [--port 4180]
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root="$repo/web/inkmap"
host=127.0.0.1; port=4180; api=""
while [ $# -gt 0 ]; do
  case "$1" in
    --api) api="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "inkmap_dev: unknown argument $1" >&2; exit 2 ;;
  esac
done
command -v npm >/dev/null || { echo "inkmap_dev: need node >= 22 and npm" >&2; exit 3; }
[ -d "$root/node_modules" ] || (cd "$root" && npm ci --no-audit --no-fund)
[ -n "$api" ] && export VITE_INKMAP_API="$api"
echo "inkmap_dev: http://$host:$port/  generator=${api:-<config.json: http://127.0.0.1:8600>}"
cd "$root" && exec npx vite --host "$host" --port "$port"
