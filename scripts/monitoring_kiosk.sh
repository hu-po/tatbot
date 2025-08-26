#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/monitoring_kiosk.sh [target_node] [refresh_seconds]
# Examples:
#   scripts/monitoring_kiosk.sh                    # eek:3000 with 5s refresh (default)
#   scripts/monitoring_kiosk.sh eek               # eek:3000 with 5s refresh
#   scripts/monitoring_kiosk.sh eek 10            # eek:3000 with 10s refresh
#   scripts/monitoring_kiosk.sh 192.168.1.97      # specific IP with 5s refresh

TARGET_NODE="${1:-eek}"
REFRESH_SECONDS="${2:-5}"

if [[ "${TARGET_NODE}" == "-h" || "${TARGET_NODE}" == "--help" ]]; then
  echo "Usage: $0 [target_node] [refresh_seconds]"
  echo "Opens Grafana Fleet Overview dashboard in kiosk mode"
  echo ""
  echo "Arguments:"
  echo "  target_node      - Node running Grafana (default: eek)"
  echo "  refresh_seconds  - Dashboard refresh interval (default: 5)"
  echo ""
  echo "Examples:"
  echo "  $0                    # eek:3000 with 5s refresh"
  echo "  $0 eek 10            # eek:3000 with 10s refresh"
  echo "  $0 192.168.1.97      # specific IP with 5s refresh"
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

HOST="$(resolve_host "${TARGET_NODE}")"
GRAFANA_URL="http://${HOST}:3000/d/fleet-overview/fleet-overview?kiosk=tv&refresh=${REFRESH_SECONDS}s"

echo "🖥️  Starting monitoring kiosk -> ${GRAFANA_URL}"
pkill -9 -f "chromium.*fleet-overview" || true
rm -rf /tmp/monitoring_kiosk.log || true
export DISPLAY=:0
setsid chromium-browser --kiosk "${GRAFANA_URL}" --disable-gpu --disable-extensions --no-first-run --no-default-browser-check >> /tmp/monitoring_kiosk.log 2>&1 &
disown
echo "✅ Launched monitoring kiosk (refresh every ${REFRESH_SECONDS}s)"
echo "📊 Dashboard: Fleet Overview"
echo "🔗 URL: ${GRAFANA_URL}"