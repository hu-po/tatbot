#!/bin/bash
set -euo pipefail

# Comprehensive monitoring stack verification script
# Run on eek to verify all nodes are providing metrics correctly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INVENTORY="${ROOT_DIR}/config/monitoring/inventory.yml"

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

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

check() {
    local description="$1"
    local command="$2"
    local expected_pattern="${3:-}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking: $description"
    
    if eval "$command" >/dev/null 2>&1; then
        if [[ -n "$expected_pattern" ]]; then
            local output
            output=$(eval "$command" 2>/dev/null || echo "")
            if echo "$output" | grep -q "$expected_pattern"; then
                log_success "$description"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
                return 0
            else
                log_error "$description - Expected pattern '$expected_pattern' not found"
                FAILED_CHECKS=$((FAILED_CHECKS + 1))
                return 1
            fi
        else
            log_success "$description"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
            return 0
        fi
    else
        log_error "$description"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

check_url() {
    local description="$1"
    local url="$2"
    local expected_pattern="${3:-}"
    local timeout="${4:-5}"
    
    check "$description" "curl -s --max-time $timeout '$url'" "$expected_pattern"
}

echo "🔍 Starting comprehensive monitoring verification..."
echo "📍 Running from: $(hostname)"
echo "📂 Repository: $ROOT_DIR"
echo

# 1. Check Docker services on eek
log_info "=== Docker Services (Local) ==="
check "Docker daemon running" "docker info"
check "Docker Compose available" "docker compose version"

# 2. Check Prometheus and Grafana containers
log_info "=== Prometheus & Grafana Containers ==="
check "Prometheus container running" "docker ps --filter 'name=prometheus' --format '{{.Status}}'" "Up"
check "Grafana container running" "docker ps --filter 'name=grafana' --format '{{.Status}}'" "Up"

# 3. Check Prometheus web interface
log_info "=== Prometheus Web Interface ==="
check_url "Prometheus health check" "http://localhost:9090/-/ready" "Prometheus is Ready"
check_url "Prometheus targets page" "http://localhost:9090/targets" "up"

# 4. Check Grafana web interface
log_info "=== Grafana Web Interface ==="
check_url "Grafana health check" "http://localhost:3000/api/health" "ok"
check_url "Fleet Overview dashboard" "http://localhost:3000/d/fleet-overview/fleet-overview" "Fleet Overview"

# 5. Parse inventory.yml and check all node exporters
log_info "=== Node Exporter Endpoints ==="
if [[ -f "$INVENTORY" ]]; then
    # Extract node addresses from inventory.yml
    while IFS= read -r line; do
        if [[ $line =~ addr:\ \"([^\"]+)\" ]]; then
            addr="${BASH_REMATCH[1]}"
            check_url "Node exporter: $addr" "http://$addr/metrics" "node_cpu_seconds_total"
        fi
    done < "$INVENTORY"
else
    log_error "Inventory file not found: $INVENTORY"
fi

# 6. Check GPU exporters
log_info "=== GPU Exporter Endpoints ==="
# NVIDIA DCGM exporters
check_url "NVIDIA GPU (ook)" "http://192.168.1.90:9400/metrics" "DCGM_FI_DEV_GPU_UTIL" 10
check_url "NVIDIA GPU (oop)" "http://192.168.1.51:9400/metrics" "DCGM_FI_DEV_GPU_UTIL" 10

# Intel GPU exporter
check_url "Intel GPU (hog)" "http://192.168.1.88:8080/metrics" "intel_gpu_top" 10

# Jetson exporter
check_url "Jetson GPU (ojo)" "http://192.168.1.96:9100/metrics" "jetson" 10

# 7. Check Raspberry Pi exporters
log_info "=== Raspberry Pi Exporters ==="
check_url "RPi exporter (rpi1)" "http://192.168.1.98:9110/metrics" "rpi_" 10
check_url "RPi exporter (rpi2)" "http://192.168.1.99:9110/metrics" "rpi_" 10

# 8. Check Prometheus is scraping all targets
log_info "=== Prometheus Target Status ==="
targets_response=$(curl -s http://localhost:9090/api/v1/targets 2>/dev/null || echo '{"status":"error"}')
if echo "$targets_response" | jq -e '.status == "success"' >/dev/null 2>&1; then
    active_targets=$(echo "$targets_response" | jq -r '.data.activeTargets[] | select(.health == "up") | .labels.instance' | wc -l)
    total_targets=$(echo "$targets_response" | jq -r '.data.activeTargets[] | .labels.instance' | wc -l)
    
    if [[ $active_targets -eq $total_targets && $total_targets -gt 0 ]]; then
        log_success "All $total_targets Prometheus targets UP"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        log_error "Only $active_targets/$total_targets Prometheus targets UP"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
else
    log_error "Failed to query Prometheus targets API"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
fi

# 9. Summary
echo
log_info "=== Verification Summary ==="
echo "📊 Total checks: $TOTAL_CHECKS"
echo "✅ Passed: $PASSED_CHECKS"
echo "❌ Failed: $FAILED_CHECKS"

if [[ $FAILED_CHECKS -eq 0 ]]; then
    echo
    log_success "🎉 All monitoring systems operational!"
    log_info "🖥️  Ready for kiosk display. Run on rpi1:"
    echo "   cd ~/tatbot && bash scripts/monitoring_kiosk.sh"
    echo
    log_info "🌐 Access URLs:"
    echo "   Grafana: http://eek:3000/"
    echo "   Prometheus: http://eek:9090/"
    echo "   Fleet Overview: http://eek:3000/d/fleet-overview/fleet-overview"
    echo
    exit 0
else
    echo
    log_error "🚨 $FAILED_CHECKS checks failed. Please fix issues before proceeding."
    log_info "💡 Common fixes:"
    echo "   - Ensure all nodes are powered on and accessible"
    echo "   - Check that exporters are running: sudo systemctl status node_exporter"
    echo "   - Verify Docker containers: docker ps"
    echo "   - Check Prometheus targets: curl http://localhost:9090/targets"
    echo
    exit 1
fi