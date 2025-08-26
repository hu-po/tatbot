#!/usr/bin/env bash
set -euo pipefail

# Clean monitoring stack data (Prometheus + Grafana volumes)
# Usage: scripts/monitor/clean.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "🧹 Cleaning monitoring stack volumes and containers..."
cd "$ROOT_DIR/config/monitoring"

# Use Makefile target which does compose down -v and prune
make -s clean || {
  echo "Falling back to docker compose directly..."
  docker compose -f compose/docker-compose.yml down -v || true
}

echo "✅ Monitoring stack cleaned"
echo "ℹ️  To restart services: make -C config/monitoring up"

