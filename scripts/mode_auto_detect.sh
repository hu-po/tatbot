#!/usr/bin/env bash
set -euo pipefail

# Auto-detect home router and switch between edge/home modes
# This script runs continuously on rpi2 to provide automatic mode switching
# - If home router (192.168.1.1) is reachable → switch to home mode
# - If home router is not reachable → switch to edge mode with conditional DHCP

# Configuration
HOME_ROUTER_IP=${HOME_ROUTER_IP:-192.168.1.1}
CHECK_INTERVAL=${CHECK_INTERVAL:-20}  # seconds between checks
DNSMASQ_ACTIVE_LINK=/etc/dnsmasq.d/active.conf
LOG_PREFIX="[mode-auto-detect]"

# Function to log messages with timestamp
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX $1"
}

# Function to check if home router is reachable
check_home_router() {
    # Try to ping home router with 2 second timeout, single packet
    ping -c 1 -W 2 "$HOME_ROUTER_IP" >/dev/null 2>&1
}

# Function to get current mode from dnsmasq symlink
get_current_mode() {
    local link
    link=$(readlink -f "${DNSMASQ_ACTIVE_LINK}" 2>/dev/null || true)
    if [[ "$link" == *mode-home.conf ]]; then 
        echo "home"
    elif [[ "$link" == *mode-edge.conf ]]; then 
        echo "edge"
    else
        echo "unknown"
    fi
}

# Function to switch mode using the Python toggler
switch_mode() {
    local target_mode=$1
    log_msg "Switching to $target_mode mode..."
    
    # Check if we're in tatbot directory, if not cd to it
    if [[ ! -f ~/tatbot/src/tatbot/utils/mode_toggle.py ]]; then
        log_msg "ERROR: Cannot find mode_toggle.py"
        return 1
    fi
    
    # Use the Python mode toggler
    cd ~/tatbot
    if command -v uv >/dev/null 2>&1; then
        # Use uv if available (preferred)
        source scripts/setup_env.sh >/dev/null 2>&1 || true
        uv run -q python src/tatbot/utils/mode_toggle.py --mode "$target_mode" 2>&1 | while read -r line; do
            log_msg "$line"
        done
    else
        # Fallback to direct Python
        python3 src/tatbot/utils/mode_toggle.py --mode "$target_mode" 2>&1 | while read -r line; do
            log_msg "$line"
        done
    fi
    
    # Check if DHCP should be conditionally disabled in edge mode
    if [[ "$target_mode" == "edge" ]]; then
        # If home router is present, we need to disable DHCP even in edge mode
        if check_home_router; then
            log_msg "Home router detected in edge mode - disabling DHCP to prevent conflicts"
            # Stop DHCP by commenting out dhcp-range line
            sudo sed -i 's/^dhcp-range=/#dhcp-range=/' /etc/dnsmasq-profiles/mode-edge.conf
            sudo systemctl reload dnsmasq
        else
            # Re-enable DHCP if it was disabled
            sudo sed -i 's/^#dhcp-range=/dhcp-range=/' /etc/dnsmasq-profiles/mode-edge.conf
            sudo systemctl reload dnsmasq
        fi
    fi
}

# Main monitoring loop
log_msg "Starting auto-detection service (checking every ${CHECK_INTERVAL}s)"
log_msg "Home router IP: $HOME_ROUTER_IP"

# Track last known state to avoid unnecessary switches
last_router_state=""
last_mode=""

while true; do
    current_mode=$(get_current_mode)
    
    # Check if home router is reachable
    if check_home_router; then
        router_state="reachable"
        desired_mode="home"
    else
        router_state="unreachable"
        desired_mode="edge"
    fi
    
    # Only log and switch if state changed
    if [[ "$router_state" != "$last_router_state" ]]; then
        log_msg "Home router ($HOME_ROUTER_IP) is $router_state"
        last_router_state="$router_state"
    fi
    
    # Switch mode if needed
    if [[ "$current_mode" != "$desired_mode" ]]; then
        log_msg "Mode mismatch: current=$current_mode, desired=$desired_mode"
        switch_mode "$desired_mode"
        last_mode="$desired_mode"
    elif [[ "$current_mode" != "$last_mode" ]]; then
        # First iteration or mode was changed externally
        log_msg "Currently in $current_mode mode"
        last_mode="$current_mode"
    fi
    
    # Wait before next check
    sleep "$CHECK_INTERVAL"
done