#!/usr/bin/env bash
set -euo pipefail

# Fetches and pins selected dashboards to config/monitoring/grafana/dashboards
# Requires: curl, jq

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DASH_DIR="$ROOT_DIR/config/monitoring/grafana/dashboards"
mkdir -p "$DASH_DIR"

fetch() {
  local id="$1"; local out="$2";
  echo "Fetching dashboard ${id}..."
  
  # Get latest revision - handle both old and new API response formats
  local rev
  rev="$(curl -sL "https://grafana.com/api/dashboards/${id}/revisions" | jq -r 'if type=="array" then .[-1].revision else .revision end' 2>/dev/null || echo "1")"
  
  # Download the dashboard
  if curl -sL "https://grafana.com/api/dashboards/${id}/revisions/${rev}/download" -o "$out"; then
    echo "✅ Downloaded ${id} (revision ${rev}) to $(basename "$out")"
  else
    echo "❌ Failed to download dashboard ${id}"
    return 1
  fi
}

fetch 1860 "$DASH_DIR/node-exporter-full-1860.json"
fetch 12239 "$DASH_DIR/nvidia-dcgm-12239.json"
# Try both Jetson dashboards; ignore failures
fetch 14493 "$DASH_DIR/jetson-14493.json" || true
fetch 21727 "$DASH_DIR/jetson-21727.json" || true
fetch 23251 "$DASH_DIR/intel-gpu-23251.json"
echo "Dashboards fetched to $DASH_DIR"

