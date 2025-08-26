#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/monitor/kiosk.sh [target_node] [refresh_seconds]
# Examples:
#   scripts/monitor/kiosk.sh                    # eek:3000 with 5s refresh (default)
#   scripts/monitor/kiosk.sh eek               # eek:3000 with 5s refresh
#   scripts/monitor/kiosk.sh eek 10            # eek:3000 with 10s refresh
#   scripts/monitor/kiosk.sh 192.168.1.97      # specific IP with 5s refresh

TARGET_NODE="${1:-eek}"
REFRESH_SECONDS="${2:-5}"

if [[ "${TARGET_NODE}" == "-h" || "${TARGET_NODE}" == "--help" ]]; then
  echo "Usage: $0 [target_node] [refresh_seconds]"
  echo "Opens Grafana Tatbot Compute dashboard in kiosk mode"
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

# Pre-flight checks
check_grafana() {
    local host="$1"
    local url="http://${host}:3000/api/health"
    
    echo "🔍 Verifying Grafana is accessible at ${host}:3000..."
    
    if curl -s --max-time 5 "$url" | grep -q "ok"; then
        echo "✅ Grafana health check passed"
        return 0
    else
        echo "❌ Grafana health check failed"
        echo "💡 Try: curl http://${host}:3000/api/health"
        return 1
    fi
}

check_dashboard() {
    local host="$1"
    local api_url="http://${host}:3000/api/dashboards/uid/tatbot-compute"
    
    echo "🔍 Verifying Tatbot Compute dashboard exists..."
    
    if curl -s --max-time 5 "$api_url" | grep -q '"slug":"tatbot-compute"'; then
        echo "✅ Tatbot Compute dashboard accessible"
        return 0
    else
        echo "❌ Tatbot Compute dashboard not found"
        echo "💡 Check dashboard is provisioned: http://${host}:3000/dashboards"
        return 1
    fi
}

HOST="$(resolve_host "${TARGET_NODE}")"
GRAFANA_URL="http://${HOST}:3000/d/tatbot-compute/tatbot-compute?kiosk=tv&refresh=${REFRESH_SECONDS}s"

# Run pre-flight checks
echo "🚀 Pre-flight checks for monitoring kiosk..."
if ! check_grafana "$HOST"; then
    echo "🚨 Grafana not ready. Please run monitoring server script on eek:"
    echo "   cd ~/tatbot && ./scripts/monitor/server.sh --restart"
    exit 1
fi

if ! check_dashboard "$HOST"; then
    echo "🚨 Dashboard not ready. Please check Grafana configuration."
    exit 1
fi

echo "🖥️  Starting monitoring kiosk -> ${GRAFANA_URL}"

# Kill all existing Chrome/Chromium processes cleanly
echo "🧹 Cleaning up existing browser processes..."
pkill -f "chromium-browser.*kiosk" || true
pkill -f "chromium-browser.*tatbot-compute" || true
pkill -f "chromium.*kiosk" || true
pkill -f "chrome.*kiosk" || true
sleep 2

# Clean up old logs and temp files
rm -rf /tmp/monitoring_kiosk.log || true
rm -rf /tmp/.org.chromium.Chromium.* || true

# Ensure X11 display is available
export DISPLAY=:0

# Start fresh Chromium in kiosk mode
echo "🚀 Launching fresh browser in kiosk mode..."
setsid chromium-browser --kiosk "${GRAFANA_URL}" \
    --disable-gpu \
    --disable-extensions \
    --disable-plugins \
    --disable-background-timer-throttling \
    --disable-backgrounding-occluded-windows \
    --disable-renderer-backgrounding \
    --disable-features=TranslateUI \
    --disable-ipc-flooding-protection \
    --no-first-run \
    --no-default-browser-check \
    --no-sandbox \
    --disable-dev-shm-usage \
    --disable-background-networking \
    --disable-default-apps \
    --disable-sync \
    --metrics-recording-only \
    --no-report-upload \
    --user-data-dir=/tmp/monitoring-kiosk-profile \
    >> /tmp/monitoring_kiosk.log 2>&1 &

disown
sleep 3

echo "✅ Launched monitoring kiosk (refresh every ${REFRESH_SECONDS}s)"
echo "📊 Dashboard: Tatbot Compute"
echo "🔗 URL: ${GRAFANA_URL}"
echo "📁 Logs: /tmp/monitoring_kiosk.log"
