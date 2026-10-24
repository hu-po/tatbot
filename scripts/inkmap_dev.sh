#!/usr/bin/env bash
# Vite dev server for web/inkmap. Installs node_modules on first run.
#   scripts/inkmap_dev.sh [--host 0.0.0.0] [--port 4180]
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=scripts/lib/cli_hint.sh
source "$repo/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot inkmap dev"
root="$repo/web/inkmap"
host=127.0.0.1; port=4180
while [ $# -gt 0 ]; do
  case "$1" in
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    -h|--help) sed -n '2,3p' "$0"; exit 0 ;;
    *) echo "inkmap_dev: unknown argument $1" >&2; exit 2 ;;
  esac
done
command -v npm >/dev/null || { echo "inkmap_dev: need node >= 22 and npm" >&2; exit 3; }
[ -d "$root/node_modules" ] || (cd "$root" && npm ci --no-audit --no-fund)
echo "inkmap_dev: http://$host:$port/"
cd "$root" && exec npx vite --host "$host" --port "$port"
