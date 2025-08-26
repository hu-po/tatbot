#!/bin/bash
set -euo pipefail

# Monitoring system cache cleanup script
# Removes all cached data, volumes, logs, and temporary files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

echo "🧹 Starting monitoring system cache cleanup..."
echo "📍 Running from: $(hostname)"
echo "📂 Repository: $ROOT_DIR"
echo

# Check if we should clean Docker data (only on eek)
CLEAN_DOCKER=false
if [[ "$(hostname)" == "eek" ]]; then
    CLEAN_DOCKER=true
    log_info "Running on eek - will clean Docker monitoring data"
else
    log_info "Running on $(hostname) - will clean local cache only"
fi

echo

# 1. Stop and clean Docker containers (eek only)
if [[ "$CLEAN_DOCKER" == true ]]; then
    log_info "=== Docker Container Cleanup ==="
    cd "${ROOT_DIR}/config/monitoring"
    
    # Stop all monitoring containers
    log_info "Stopping monitoring containers..."
    docker stop $(docker ps -q --filter "name=compose-prometheus" --filter "name=compose-grafana") 2>/dev/null || log_warning "No monitoring containers running"
    
    # Remove containers
    log_info "Removing monitoring containers..."
    docker rm $(docker ps -aq --filter "name=compose-prometheus" --filter "name=compose-grafana") 2>/dev/null || log_warning "No monitoring containers to remove"
    
    # Remove volumes with data
    log_info "Removing Docker volumes..."
    docker volume rm compose_prom_data compose_grafana_data monitoring_prom_data monitoring_grafana_data 2>/dev/null || log_warning "No volumes to remove"
    
    # Clean networks
    log_info "Cleaning Docker networks..."
    docker network prune -f >/dev/null 2>&1 || true
    
    log_success "Docker cleanup complete"
    echo
fi

# 2. Clean browser cache and profiles
log_info "=== Browser Cache Cleanup ==="

# Clean Chromium/Chrome temp files and profiles
log_info "Removing browser temp files..."
rm -rf /tmp/monitoring-kiosk-profile 2>/dev/null || true
rm -rf /tmp/.org.chromium.Chromium.* 2>/dev/null || true
rm -rf ~/.cache/chromium 2>/dev/null || true
rm -rf ~/.config/chromium/Default/Local\ Storage 2>/dev/null || true
rm -rf ~/.config/chromium/Default/Session\ Storage 2>/dev/null || true

# Kill any remaining browser processes
log_info "Killing browser processes..."
pkill -f "chromium-browser.*kiosk" 2>/dev/null || true
pkill -f "chromium.*tatbot-compute" 2>/dev/null || true
pkill -f "chrome.*kiosk" 2>/dev/null || true

log_success "Browser cache cleanup complete"
echo

# 3. Clean monitoring logs
log_info "=== Log File Cleanup ==="

log_info "Removing monitoring log files..."
rm -rf /tmp/monitoring_kiosk.log 2>/dev/null || true
rm -rf /tmp/monitoring_server.log 2>/dev/null || true
rm -rf /nfs/tatbot/mcp-logs/monitoring*.log 2>/dev/null || true

# Clean systemd journal entries (if running as root)
if [[ $EUID -eq 0 ]]; then
    log_info "Cleaning systemd journal (monitoring entries)..."
    journalctl --vacuum-time=1h >/dev/null 2>&1 || true
else
    log_warning "Run with sudo to clean systemd journal entries"
fi

log_success "Log cleanup complete"
echo

# 4. Clean temporary monitoring files
log_info "=== Temporary Files Cleanup ==="

log_info "Removing temporary monitoring files..."
rm -rf /tmp/monitoring_* 2>/dev/null || true
rm -rf /tmp/grafana_* 2>/dev/null || true
rm -rf /tmp/prometheus_* 2>/dev/null || true

# Clean any monitoring lock files
rm -rf /var/lock/monitoring_* 2>/dev/null || true

log_success "Temporary files cleanup complete"
echo

# 5. Clean application caches
log_info "=== Application Cache Cleanup ==="

# Clean any cached dashboard files
log_info "Removing cached dashboard files..."
find /tmp -name "*dashboard*" -type f -mtime +0 -delete 2>/dev/null || true
find /tmp -name "*grafana*" -type f -mtime +0 -delete 2>/dev/null || true

# Clean any monitoring-related downloads
rm -rf /tmp/node_exporter* 2>/dev/null || true
rm -rf /tmp/rpi_exporter* 2>/dev/null || true

log_success "Application cache cleanup complete"
echo

# 6. Summary and recommendations
log_info "=== Cleanup Summary ==="
echo "🧹 Cache cleanup completed on $(hostname)"

if [[ "$CLEAN_DOCKER" == true ]]; then
    echo "   ✅ Docker containers stopped and removed"
    echo "   ✅ Docker volumes deleted"
    echo "   ✅ Docker networks cleaned"
fi

echo "   ✅ Browser cache and profiles cleared"
echo "   ✅ Log files removed"
echo "   ✅ Temporary files cleaned"
echo "   ✅ Application caches cleared"

echo
log_info "💡 Next steps:"

if [[ "$CLEAN_DOCKER" == true ]]; then
    echo "   🔄 Restart monitoring: cd ~/tatbot && ./scripts/monitoring_server.sh"
else
    echo "   🔄 Restart kiosk: cd ~/tatbot && ./scripts/monitoring_kiosk.sh"
fi

echo "   📊 Check status: docker ps (on eek) or ps aux | grep chromium"
echo

log_success "🎉 Monitoring cache cleanup complete!"