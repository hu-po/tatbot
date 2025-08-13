#!/usr/bin/env bash
set -euo pipefail

# Simple auto-detection script for edge/home mode switching
# This script runs on rpi2 and switches dnsmasq configs based on home router availability
# NO Python dependencies - just bash and basic networking tools

# Configuration
HOME_ROUTER_IP=${HOME_ROUTER_IP:-192.168.1.1}
CHECK_INTERVAL=${CHECK_INTERVAL:-20}
DNSMASQ_PROFILES_DIR=/etc/dnsmasq-profiles
DNSMASQ_ACTIVE_LINK=/etc/dnsmasq.d/active.conf
LOG_PREFIX="[mode-auto-detect]"

# Function to log messages with timestamp
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX $1"
}

# Function to check if home router is reachable
check_home_router() {
    ping -c 1 -W 2 "$HOME_ROUTER_IP" >/dev/null 2>&1
}

# Function to get current mode from dnsmasq symlink
get_current_mode() {
    local link
    link=$(readlink -f "${DNSMASQ_ACTIVE_LINK}" 2>/dev/null || echo "")
    if [[ "$link" == *mode-home.conf ]]; then 
        echo "home"
    elif [[ "$link" == *mode-edge.conf ]]; then 
        echo "edge"
    else
        echo "unknown"
    fi
}

# Function to switch to home mode
switch_to_home() {
    log_msg "Switching to HOME mode (DNS forwarding to home router)"
    sudo ln -sf "${DNSMASQ_PROFILES_DIR}/mode-home.conf" "${DNSMASQ_ACTIVE_LINK}"
    sudo systemctl reload dnsmasq
}

# Function to switch to edge mode
switch_to_edge() {
    log_msg "Switching to EDGE mode (local DNS + conditional DHCP)"

    # Point active.conf symlink to the edge profile
    sudo ln -sf "${DNSMASQ_PROFILES_DIR}/mode-edge.conf" "${DNSMASQ_ACTIVE_LINK}"

    # Conditionally enable/disable DHCP directly in the edge profile
    if check_home_router; then
        log_msg "Home router detected - disabling DHCP in edge mode to prevent conflicts"
        sudo sed -i 's/^dhcp-range=/#dhcp-range=/' "${DNSMASQ_PROFILES_DIR}/mode-edge.conf"
    else
        log_msg "No home router detected - enabling DHCP in edge mode"
        sudo sed -i 's/^#dhcp-range=/dhcp-range=/' "${DNSMASQ_PROFILES_DIR}/mode-edge.conf"
    fi

    sudo systemctl reload dnsmasq
}

# Main monitoring loop
log_msg "Starting simplified auto-detection service (checking every ${CHECK_INTERVAL}s)"
log_msg "Home router IP: $HOME_ROUTER_IP"

# Track last known state to avoid unnecessary switches and logs
last_router_reachable=""
last_mode=""

while true; do
    current_mode=$(get_current_mode)
    
    # Check if home router is reachable
    if check_home_router; then
        router_reachable="yes"
        desired_mode="home"
    else
        router_reachable="no"
        desired_mode="edge"
    fi
    
    # Only log router state changes
    if [[ "$router_reachable" != "$last_router_reachable" ]]; then
        if [[ "$router_reachable" == "yes" ]]; then
            log_msg "Home router ($HOME_ROUTER_IP) is reachable"
        else
            log_msg "Home router ($HOME_ROUTER_IP) is unreachable"
        fi
        last_router_reachable="$router_reachable"
    fi
    
    # Switch mode if needed
    if [[ "$current_mode" != "$desired_mode" ]]; then
        log_msg "Mode mismatch: current=$current_mode, desired=$desired_mode"
        
        if [[ "$desired_mode" == "home" ]]; then
            switch_to_home
        else
            switch_to_edge
        fi
        
        last_mode="$desired_mode"
    elif [[ "$current_mode" != "$last_mode" ]]; then
        # First iteration or mode was changed externally
        log_msg "Currently in $current_mode mode"
        last_mode="$current_mode"
    fi
    
    # Wait before next check
    sleep "$CHECK_INTERVAL"
done