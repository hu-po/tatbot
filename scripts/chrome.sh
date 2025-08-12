#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/chrome.sh [node_or_host] [port]
# Examples:
#   scripts/chrome.sh                 # localhost:8080 (legacy behavior)
#   scripts/chrome.sh ook             # 192.168.1.90:8080 (from src/conf/mcp/ook.yaml)
#   scripts/chrome.sh oop 8080        # 192.168.1.51:8080
#   scripts/chrome.sh 192.168.1.90    # 192.168.1.90:8080

NODE_OR_HOST="${1:-localhost}"
PORT="${2:-8080}"

if [[ "${NODE_OR_HOST}" == "-h" || "${NODE_OR_HOST}" == "--help" ]]; then
  echo "Usage: $0 [node_or_host] [port]"
  echo "Nodes: ook, oop, eek, hog, rpi1, rpi2, ojo (resolved via src/conf/mcp/<node>.yaml)"
  exit 0
fi

resolve_host() {
  local target="$1"
  case "${target}" in
    localhost|127.0.0.1)
      echo "127.0.0.1"; return 0;;
    ook|oop|eek|hog|rpi1|rpi2|ojo)
      local yaml="src/conf/mcp/${target}.yaml"
      if [[ -f "${yaml}" ]]; then
        # Extract host: "IP" from YAML
        awk -F '"' '/^host:/ {print $2; exit}' "${yaml}"
        return 0
      fi
      echo "${target}"; return 0;;
    *)
      # Assume provided value is already a hostname or IP
      echo "${target}"; return 0;;
  esac
}

HOST="$(resolve_host "${NODE_OR_HOST}")"
URL="http://${HOST}:${PORT}"

echo "🌐 Starting chrome browser for viz -> ${URL}"
pkill -9 -f chromium || true
rm -rf /tmp/viz.log || true
export DISPLAY=:0
setsid chromium-browser --kiosk "${URL}" --disable-gpu >> /tmp/viz.log 2>&1 &
disown
echo "✅ Launched kiosk at ${URL} (DISPLAY=${DISPLAY})"