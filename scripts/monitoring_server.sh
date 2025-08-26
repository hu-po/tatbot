#!/bin/bash
set -euo pipefail

# Comprehensive monitoring verification and startup script
# Single entry point for monitoring system on eek
# Usage: ./verify_monitoring.sh [--restart]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INVENTORY="${ROOT_DIR}/config/monitoring/inventory.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_debug() { echo -e "${PURPLE}🔍 $1${NC}"; }

# Check if restart flag is provided
RESTART_SERVER=false
if [[ "${1:-}" == "--restart" ]]; then
    RESTART_SERVER=true
fi

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

echo "🔍 Starting monitoring verification and setup..."
echo "📍 Running from: $(hostname)"
echo "📂 Repository: $ROOT_DIR"

# Verify we're on eek
if [[ "$(hostname)" != "eek" ]]; then
    log_error "This script must be run on eek (monitoring server host)"
    echo "💡 Current host: $(hostname)"
    exit 1
fi

echo

# 1. Restart monitoring services if requested
if [[ "$RESTART_SERVER" == true ]]; then
    log_info "=== Restarting Monitoring Services ==="
    log_info "Stopping existing containers..."
    cd "${ROOT_DIR}/config/monitoring"
    make -s down || log_warning "No containers were running"
    
    log_info "Starting Prometheus + Grafana..."
    make -s up
    
    log_info "Waiting for services to stabilize..."
    sleep 10
    echo
fi

# 2. Check Docker services on eek
log_info "=== Docker Services (Local) ==="
check "Docker daemon running" "docker info"
check "Docker Compose available" "docker compose version"

# 3. Check Prometheus and Grafana containers
log_info "=== Prometheus & Grafana Containers ==="
check "Prometheus container running" "docker ps --filter 'name=prometheus' --format '{{.Status}}'" "Up"
check "Grafana container running" "docker ps --filter 'name=grafana' --format '{{.Status}}'" "Up"

if docker ps --filter 'name=prometheus' --format '{{.Status}}' | grep -q "Restarting"; then
    log_warning "Prometheus is restarting - checking logs..."
    echo "Recent Prometheus logs:"
    docker logs --tail 10 "$(docker ps --filter 'name=prometheus' --format '{{.Names}}')" 2>&1 | sed 's/^/  /'
fi

# 4. Check Prometheus web interface
log_info "=== Prometheus Web Interface ==="
check_url "Prometheus health check" "http://localhost:9090/-/ready" "Prometheus.*Ready"
check_url "Prometheus targets page" "http://localhost:9090/targets" "up"

# 5. Check Grafana web interface
log_info "=== Grafana Web Interface ==="
check_url "Grafana health check" "http://localhost:3000/api/health" "ok"
check_url "Tatbot Compute dashboard" "http://localhost:3000/d/tatbot-compute/tatbot-compute" "Tatbot Compute"

# 6. Deep dive on each node
log_info "=== Deep Node Diagnostics ==="
declare -A node_ips=(["eek"]="192.168.1.97" ["ook"]="192.168.1.90" ["oop"]="192.168.1.51" ["hog"]="192.168.1.88" ["ojo"]="192.168.1.96" ["rpi1"]="192.168.1.98" ["rpi2"]="192.168.1.99")

for node in "${!node_ips[@]}"; do
    ip="${node_ips[$node]}"
    echo
    log_info "🖥️  NODE: $node ($ip)"
    
    # Basic connectivity
    if ping -c 1 -W 2 "$ip" >/dev/null 2>&1; then
        log_success "$node: Network reachable"
    else
        log_error "$node: Network unreachable"
        continue
    fi
    
    # SSH connectivity  
    if timeout 5 ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -q "$node" 'exit 0' 2>/dev/null; then
        log_success "$node: SSH accessible"
        
        # Get system info
        log_debug "$node: Getting system info..."
        uptime_info=$(timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'uptime' 2>/dev/null | tr -d '\n' || echo "unknown")
        echo "  Uptime: $uptime_info"
        
        # Check if node_exporter service is running (except ojo)
        if [[ "$node" != "ojo" ]]; then
            ssh_result=$(timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'systemctl is-active node_exporter 2>/dev/null || echo "inactive"' 2>/dev/null || echo "ssh-failed")
            if [[ "$ssh_result" == "active" ]]; then
                log_success "$node: node_exporter service active"
            else
                log_error "$node: node_exporter service $ssh_result"
                if [[ "$ssh_result" == "inactive" ]]; then
                    log_debug "$node: Checking node_exporter installation..."
                    if timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'ls -la /usr/local/bin/node_exporter' 2>/dev/null; then
                        echo "  node_exporter binary exists but service inactive"
                    else
                        echo "  node_exporter binary not found"
                    fi
                fi
            fi
        fi
        
        # Check specific services per node
        case "$node" in
            "ook"|"oop")
                # Check DCGM exporter
                dcgm_result=$(timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'docker ps --filter "name=dcgm" --format "{{.Status}}" 2>/dev/null | head -1' 2>/dev/null || echo "check-failed")
                if echo "$dcgm_result" | grep -q "Up"; then
                log_success "$node: DCGM exporter container running"
                else
                    log_error "$node: DCGM exporter not running ($dcgm_result)"
                    # Check if NVIDIA driver is working
                    nvidia_result=$(timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1' 2>/dev/null || echo "nvidia-smi failed")
                    if [[ "$nvidia_result" != "nvidia-smi failed" ]]; then
                        echo "  GPU detected: $nvidia_result"
                    else
                        echo "  nvidia-smi not working or GPU not detected"
                    fi
                fi
                ;;
            "hog")
                # Check Intel GPU exporter
                intel_result=$(timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'docker ps --filter "name=intel-gpu" --format "{{.Status}}" 2>/dev/null | head -1' 2>/dev/null || echo "check-failed")
                if echo "$intel_result" | grep -q "Up"; then
                    log_success "$node: Intel GPU exporter container running"
                else
                    log_error "$node: Intel GPU exporter not running ($intel_result)"
                    # Check if Intel GPU tools are available
                    intel_gpu_result=$(timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'which intel_gpu_top 2>/dev/null || echo "not found"' 2>/dev/null || echo "check-failed")
                    echo "  intel_gpu_top availability: $intel_gpu_result"
                fi
                ;;
            "ojo")
                # Check jetson-stats-node-exporter
                jetson_result=$(timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'systemctl is-active jetson-stats-node-exporter 2>/dev/null || echo "inactive"' 2>/dev/null || echo "ssh-failed")
                if [[ "$jetson_result" == "active" ]]; then
                    log_success "$node: jetson-stats-node-exporter service active"
                else
                    log_error "$node: jetson-stats-node-exporter service $jetson_result"
                    # Check if jetson-stats is installed
                    jtop_result=$(timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'python3 -c "import jetson_stats; print(\"installed\")" 2>/dev/null || echo "not installed"' 2>/dev/null || echo "check-failed")
                    echo "  jetson-stats package: $jtop_result"
                fi
                ;;
            "rpi1"|"rpi2")
                # Check rpi_exporter
                rpi_result=$(timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'systemctl is-active rpi_exporter 2>/dev/null || echo "inactive"' 2>/dev/null || echo "ssh-failed")
                if [[ "$rpi_result" == "active" ]]; then
                    log_success "$node: rpi_exporter service active"
                else
                    log_error "$node: rpi_exporter service $rpi_result"
                    if [[ "$rpi_result" == "inactive" ]]; then
                        log_debug "$node: Checking rpi_exporter installation..."
                        if timeout 10 ssh -o ConnectTimeout=3 -q "$node" 'ls -la /usr/local/bin/rpi_exporter' 2>/dev/null; then
                            echo "  rpi_exporter binary exists but service inactive"
                        else
                            echo "  rpi_exporter binary not found"
                        fi
                    fi
                fi
                ;;
        esac
    else
        log_error "$node: SSH not accessible"
    fi
    
    # Test HTTP endpoints
    case "$node" in
        "eek"|"ook"|"oop"|"hog"|"rpi1"|"rpi2")
            if check_url "$node: node_exporter metrics" "http://$ip:9100/metrics" "node_cpu_seconds_total" 5; then
                # Count metrics if successful
                metric_count=$(curl -s --max-time 3 "http://$ip:9100/metrics" 2>/dev/null | grep -c "^node_" || echo "0")
                echo "  Metrics available: $metric_count node_* metrics"
            fi
            ;;
        "ojo")
            if check_url "$node: jetson-stats metrics" "http://$ip:9100/metrics" "jetson" 5; then
                # Count jetson metrics if successful
                jetson_metric_count=$(curl -s --max-time 3 "http://$ip:9100/metrics" 2>/dev/null | grep -c "jetson_" || echo "0")
                echo "  Metrics available: $jetson_metric_count jetson_* metrics"
            fi
            ;;
    esac
    
    # GPU-specific endpoints
    case "$node" in
        "ook"|"oop")
            if check_url "$node: DCGM GPU metrics" "http://$ip:9400/metrics" "DCGM_FI_DEV_GPU_UTIL" 10; then
                gpu_metrics=$(curl -s --max-time 3 "http://$ip:9400/metrics" 2>/dev/null | grep -c "DCGM_" || echo "0")
                echo "  GPU metrics: $gpu_metrics DCGM_* metrics"
            fi
            ;;
        "hog")
            if check_url "$node: Intel GPU metrics" "http://$ip:8080/metrics" "intel_gpu" 10; then
                intel_metrics=$(curl -s --max-time 3 "http://$ip:8080/metrics" 2>/dev/null | grep -c "intel_gpu" || echo "0")
                echo "  GPU metrics: $intel_metrics intel_gpu* metrics"
            fi
            ;;
        "rpi1"|"rpi2")
            if check_url "$node: RPi SoC metrics" "http://$ip:9110/metrics" "rpi_" 5; then
                rpi_metrics=$(curl -s --max-time 3 "http://$ip:9110/metrics" 2>/dev/null | grep -c "rpi_" || echo "0")
                echo "  SoC metrics: $rpi_metrics rpi_* metrics"
            fi
            ;;
    esac
done

# 7. Detailed Prometheus target analysis
log_info "=== Prometheus Target Details ==="
if curl -s --max-time 5 http://localhost:9090/api/v1/targets >/dev/null 2>&1; then
    targets_response=$(curl -s http://localhost:9090/api/v1/targets)
    
    # Show detailed status for each target
    echo "$targets_response" | jq -r '.data.activeTargets[] | "\(.labels.job)/\(.labels.instance): \(.health) (\(.lastError // "no error"))"' 2>/dev/null | while read -r target_info; do
        if echo "$target_info" | grep -q ": up"; then
            log_success "Target: $target_info"
        else
            log_error "Target: $target_info"
        fi
    done
    
    # Check for common issues
    down_targets=$(echo "$targets_response" | jq -r '.data.activeTargets[] | select(.health != "up") | .labels.instance' 2>/dev/null)
    if [[ -n "$down_targets" ]]; then
        echo
        log_warning "Targets with issues:"
        while read -r target; do
            if [[ -n "$target" ]]; then
                echo "  🔍 Checking $target directly..."
                if curl -s --max-time 3 "http://$target/metrics" >/dev/null 2>&1; then
                    echo "    ✅ Direct access works - may be Prometheus config issue"
                else
                    echo "    ❌ Direct access fails - service/network issue"
                fi
            fi
        done <<< "$down_targets"
    fi
    
    # Overall Prometheus target summary
    total_targets=$(echo "$targets_response" | jq '.data.activeTargets | length' 2>/dev/null || echo "0")
    up_targets=$(echo "$targets_response" | jq '[.data.activeTargets[] | select(.health == "up")] | length' 2>/dev/null || echo "0")
    
    if [[ $up_targets -eq $total_targets && $total_targets -gt 0 ]]; then
        log_success "All $total_targets Prometheus targets UP"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        log_error "Only $up_targets/$total_targets Prometheus targets UP"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
else
    log_error "Cannot reach Prometheus API"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
fi

# 8. Dashboard validation
log_info "=== Dashboard Validation ==="
if curl -s --max-time 5 "http://localhost:3000/api/dashboards/uid/tatbot-compute" >/dev/null 2>&1; then
    dashboard_response=$(curl -s "http://localhost:3000/api/dashboards/uid/tatbot-compute")
    
    # Check if dashboard has panels
    panel_count=$(echo "$dashboard_response" | jq '.dashboard.panels | length' 2>/dev/null || echo "0")
    if [[ "$panel_count" -gt 0 ]]; then
        log_success "Tatbot Compute dashboard has $panel_count panels"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        
        # Check if Prometheus datasource is configured
        if curl -s --max-time 5 "http://localhost:3000/api/datasources" | grep -q "Prometheus"; then
            log_success "Prometheus datasource configured in Grafana"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
        else
            log_error "Prometheus datasource not found in Grafana"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
        fi
    else
        log_error "Tatbot Compute dashboard has no panels"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 2))
else
    log_error "Cannot access Tatbot Compute dashboard"
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
    echo "   Tatbot Compute: http://eek:3000/d/tatbot-compute/tatbot-compute"
    echo
    exit 0
else
    echo
    log_error "🚨 $FAILED_CHECKS checks failed. See detailed diagnostics above."
    log_info "💡 Common installation commands by node:"
    echo
    echo "Node Exporter (eek, ook, oop, hog, rpi1, rpi2):"
    echo "  cd /tmp && wget https://github.com/prometheus/node_exporter/releases/download/v1.9.1/node_exporter-1.9.1.linux-amd64.tar.gz"
    echo "  tar -xzf node_exporter-*.tar.gz && sudo install -o nodeexp -g nodeexp node_exporter*/node_exporter /usr/local/bin/"
    echo "  sudo install ~/tatbot/config/monitoring/exporters/\$(hostname)/node_exporter.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload && sudo systemctl enable --now node_exporter"
    echo
    echo "DCGM Exporter (ook, oop):"
    echo "  docker run -d --restart=always --gpus all --cap-add SYS_ADMIN --net host --name dcgm-exporter -e DCGM_EXPORTER_LISTEN=\":9400\" nvidia/dcgm-exporter:4.4.0-4.5.0-ubi9"
    echo
    echo "Intel GPU Exporter (hog):"
    echo "  docker run -d --restart=always --net host --name intel-gpu-exporter --privileged -v /sys:/sys:ro -v /dev/dri:/dev/dri restreamio/intel-prometheus:latest"
    echo
    echo "Jetson Exporter (ojo):"
    echo "  sudo pip3 install jetson-stats==4.3.2 jetson-stats-node-exporter==0.1.2"
    echo "  sudo install ~/tatbot/config/monitoring/exporters/ojo/jetson-stats-node-exporter.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload && sudo systemctl enable --now jetson-stats-node-exporter"
    echo
    echo "RPi Exporter (rpi1, rpi2):"
    echo "  cd /tmp && wget https://github.com/lukasmalkmus/rpi_exporter/releases/download/v0.4.0/rpi_exporter-0.4.0.linux-arm64.tar.gz"
    echo "  tar -xzf rpi_exporter-*.tar.gz && sudo install rpi_exporter /usr/local/bin/"
    echo "  sudo install ~/tatbot/config/monitoring/exporters/\$(hostname)/rpi_exporter.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload && sudo systemctl enable --now rpi_exporter"
    echo
    log_info "🔄 To restart monitoring services, run:"
    echo "   cd ~/tatbot && ./scripts/monitoring_server.sh --restart"
    echo
    exit 1
fi