#!/usr/bin/env bash
set -euo pipefail

# Community dashboard downloader for monitoring stack
# Note: Grafana.com API has changed and many dashboards now require authentication

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DASH_DIR="$ROOT_DIR/config/monitoring/grafana/dashboards"
mkdir -p "$DASH_DIR"

echo "🔄 Attempting to fetch community dashboards from Grafana.com..."
echo "⚠️  Note: Grafana.com API has restricted access to many dashboards"
echo

# Function to try downloading a dashboard
try_fetch() {
  local id="$1"
  local name="$2" 
  local out="$DASH_DIR/${name}.json"
  
  echo "Trying dashboard ${id} (${name})..."
  
  # Try multiple API endpoints
  local endpoints=(
    "https://grafana.com/api/dashboards/${id}/revisions/latest/download"
    "https://grafana.com/api/dashboards/${id}/revisions/download"
    "https://grafana.com/api/dashboards/${id}"
  )
  
  for endpoint in "${endpoints[@]}"; do
    if curl -sL --max-time 10 "$endpoint" -o "/tmp/dash_${id}.json"; then
      # Check if we got valid JSON with dashboard content
      if jq -e '.dashboard // .panels // .title' "/tmp/dash_${id}.json" >/dev/null 2>&1; then
        mv "/tmp/dash_${id}.json" "$out"
        echo "✅ Downloaded ${name} (ID: ${id})"
        return 0
      fi
    fi
  done
  
  rm -f "/tmp/dash_${id}.json"
  echo "❌ Failed to download ${name} (ID: ${id}) - API access restricted"
  return 1
}

# Try to fetch popular monitoring dashboards
echo "Attempting downloads..."
try_fetch 1860 "node-exporter-full-1860" || true
try_fetch 12239 "nvidia-dcgm-12239" || true  
try_fetch 14493 "jetson-14493" || true
try_fetch 21727 "jetson-21727" || true
try_fetch 23251 "intel-gpu-23251" || true

echo
echo "📁 Dashboard directory: $DASH_DIR"
echo "📊 Available dashboards:"
ls -1 "$DASH_DIR"/*.json 2>/dev/null | while read -r file; do
  title=$(jq -r '.title // .dashboard.title // "Unknown"' "$file" 2>/dev/null || echo "Unknown")
  echo "   - $(basename "$file"): $title"
done

echo
echo "💡 Alternative approach:"
echo "   1. Visit https://grafana.com/grafana/dashboards/ in browser"
echo "   2. Search for: 'node exporter', 'nvidia dcgm', etc."
echo "   3. Download JSON manually to $DASH_DIR"
echo "   4. Restart Grafana: cd ~/tatbot/config/monitoring && docker compose restart grafana"
