#!/bin/bash
# Auto-renew DHCP when gateway becomes unreachable

GATEWAY_HOME="192.168.1.1"
GATEWAY_EDGE="192.168.1.90"
DNS_SERVER="192.168.1.99"
CHECK_INTERVAL=30

while true; do
    # Get current gateway
    CURRENT_GW=$(ip route show default | awk '{print $3}' | head -1)
    
    # Check if we can reach the DNS server (rpi2)
    if ! ping -c 1 -W 2 "$DNS_SERVER" >/dev/null 2>&1; then
        echo "Cannot reach DNS server, network might be down"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # Check if current gateway matches what we expect
    if [[ "$CURRENT_GW" == "$GATEWAY_HOME" ]]; then
        # We think we're in home mode, check if home router is actually reachable
        if ! ping -c 1 -W 2 "$GATEWAY_HOME" >/dev/null 2>&1; then
            echo "Home gateway unreachable but still configured, renewing DHCP..."
            IFACE=$(ip route show default | awk '{print $5}' | head -1)
            sudo dhclient -r "$IFACE" && sudo dhclient "$IFACE"
        fi
    elif [[ "$CURRENT_GW" == "$GATEWAY_EDGE" ]]; then
        # We're in edge mode, check if ook is reachable
        if ! ping -c 1 -W 2 "$GATEWAY_EDGE" >/dev/null 2>&1; then
            echo "Edge gateway (ook) unreachable, renewing DHCP..."
            IFACE=$(ip route show default | awk '{print $5}' | head -1)
            sudo dhclient -r "$IFACE" && sudo dhclient "$IFACE"
        fi
    fi
    
    sleep $CHECK_INTERVAL
done