#!/usr/bin/env bash
set -euo pipefail

# Auto-detection script for edge/home mode switching with DHCP renewal triggers
# This script runs on rpi2 and switches dnsmasq configs based on home router availability
# When mode switches, it triggers DHCP renewal on all nodes

# Configuration
HOME_ROUTER_IP=${HOME_ROUTER_IP:-192.168.1.1}
CHECK_INTERVAL=${CHECK_INTERVAL:-20}
DNSMASQ_PROFILES_DIR=/etc/dnsmasq-profiles
DNSMASQ_ACTIVE_LINK=/etc/dnsmasq.d/active.conf
LOG_PREFIX="[mode-auto-detect]"

# List of nodes to trigger DHCP renewal on (exclude rpi2 itself)
TATBOT_NODES="eek hog ojo ook rpi1"

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

# Function to trigger DHCP renewal on a specific node
trigger_dhcp_renewal() {
    local node=$1
    log_msg "Triggering DHCP renewal on $node"
    
    # Try to SSH to the node and renew DHCP
    # Using timeout to avoid hanging if node is unreachable
    timeout 5 ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no "$node" \
        "sudo dhclient -r eth0 2>/dev/null; sudo dhclient eth0 2>/dev/null || \
         sudo dhclient -r eno1 2>/dev/null; sudo dhclient eno1 2>/dev/null || \
         sudo nmcli device reapply eth0 2>/dev/null || \
         sudo nmcli device reapply eno1 2>/dev/null || \
         sudo nmcli device reapply enp63s0 2>/dev/null || \
         sudo nmcli device reapply enp86s0 2>/dev/null || \
         sudo nmcli device reapply enp172s0 2>/dev/null || \
         true" 2>/dev/null || {
        log_msg "Failed to renew DHCP on $node (might be unreachable)"
    }
}

# Function to trigger DHCP renewal on all nodes
trigger_all_dhcp_renewals() {
    log_msg "Triggering DHCP renewal on all nodes..."
    
    # First, clear ARP cache to help with network transitions
    sudo ip neigh flush all 2>/dev/null || true
    
    # Trigger renewals in parallel for faster execution
    for node in $TATBOT_NODES; do
        trigger_dhcp_renewal "$node" &
    done
    
    # Wait for all background jobs to complete
    wait
    
    log_msg "DHCP renewal triggered on all nodes"
}

# Function to switch to home mode
switch_to_home() {
    log_msg "Switching to HOME mode (DNS forwarding to home router)"
    sudo ln -sf "${DNSMASQ_PROFILES_DIR}/mode-home.conf" "${DNSMASQ_ACTIVE_LINK}"
    sudo systemctl reload dnsmasq
    
    # Give dnsmasq a moment to reload
    sleep 2
    
    # Trigger DHCP renewals on all nodes
    trigger_all_dhcp_renewals
}

# Function to switch to edge mode
switch_to_edge() {
    log_msg "Switching to EDGE mode (local DNS + DHCP)"
    
    # Point active.conf symlink to the edge profile
    sudo ln -sf "${DNSMASQ_PROFILES_DIR}/mode-edge.conf" "${DNSMASQ_ACTIVE_LINK}"
    
    # In edge mode, always enable DHCP since no home router
    sudo sed -i 's/^#dhcp-range=/dhcp-range=/' "${DNSMASQ_PROFILES_DIR}/mode-edge.conf" 2>/dev/null || true
    
    sudo systemctl reload dnsmasq
    
    # Give dnsmasq a moment to reload
    sleep 2
    
    # Trigger DHCP renewals on all nodes
    trigger_all_dhcp_renewals
}

# Main monitoring loop
log_msg "Starting auto-detection service with DHCP triggers (checking every ${CHECK_INTERVAL}s)"
log_msg "Home router IP: $HOME_ROUTER_IP"
log_msg "Nodes to manage: $TATBOT_NODES"

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