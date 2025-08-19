#!/bin/bash
# Setup NAT on ook for internet sharing in EDGE mode
# This allows other tatbot nodes to access internet through ook's WiFi

set -e

echo "========================================="
echo "NAT Setup Script for ook (EDGE Mode)"
echo "========================================="
echo ""

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script must be run with sudo"
    echo "Usage: sudo bash $0"
    exit 1
fi

# Detect interfaces
WIFI_IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^wl' | head -1)
LAN_IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^(en|eth)' | head -1)

if [ -z "$WIFI_IFACE" ]; then
    echo "ERROR: No WiFi interface found"
    exit 1
fi

if [ -z "$LAN_IFACE" ]; then
    echo "ERROR: No Ethernet interface found"
    exit 1
fi

echo "Detected interfaces:"
echo "  WiFi: $WIFI_IFACE"
echo "  LAN:  $LAN_IFACE"
echo ""

# Enable IP forwarding permanently
echo "Enabling IP forwarding..."
if ! grep -q "^net.ipv4.ip_forward=1" /etc/sysctl.conf; then
    echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
    echo "  Added to /etc/sysctl.conf"
else
    echo "  Already enabled in /etc/sysctl.conf"
fi
sysctl -w net.ipv4.ip_forward=1 > /dev/null
echo "  IP forwarding enabled"
echo ""

# Setup NAT rules
echo "Configuring NAT rules..."

# Clear existing NAT rules (but preserve other tables)
iptables -t nat -F POSTROUTING 2>/dev/null || true
iptables -F FORWARD 2>/dev/null || true

# Add NAT masquerading for outbound traffic
iptables -t nat -A POSTROUTING -o "$WIFI_IFACE" -j MASQUERADE
echo "  Added MASQUERADE rule for $WIFI_IFACE"

# Allow forwarding from LAN to WiFi
iptables -A FORWARD -i "$LAN_IFACE" -o "$WIFI_IFACE" -j ACCEPT
echo "  Allowed forwarding: $LAN_IFACE → $WIFI_IFACE"

# Allow established connections back
iptables -A FORWARD -i "$WIFI_IFACE" -o "$LAN_IFACE" -m state --state ESTABLISHED,RELATED -j ACCEPT
echo "  Allowed established connections: $WIFI_IFACE → $LAN_IFACE"
echo ""

# Save rules persistently
echo "Saving iptables rules..."
if command -v iptables-save >/dev/null 2>&1; then
    # Check if iptables-persistent is installed
    if [ ! -d /etc/iptables ]; then
        echo "Installing iptables-persistent to save rules across reboots..."
        apt-get update >/dev/null 2>&1
        DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent >/dev/null 2>&1
    fi
    
    # Save current rules
    mkdir -p /etc/iptables
    iptables-save > /etc/iptables/rules.v4
    echo "  Rules saved to /etc/iptables/rules.v4"
else
    echo "  WARNING: iptables-save not found. Rules will not persist after reboot."
fi
echo ""

# Verify configuration
echo "Verifying NAT configuration..."
echo ""
echo "NAT Table:"
iptables -t nat -L POSTROUTING -n -v | grep -E "(Chain|MASQUERADE)" || echo "No NAT rules found!"
echo ""
echo "Forward Rules:"
iptables -L FORWARD -n -v | grep -E "(Chain|ACCEPT.*$LAN_IFACE.*$WIFI_IFACE|ACCEPT.*$WIFI_IFACE.*$LAN_IFACE)" || echo "No forward rules found!"
echo ""

# Test connectivity
echo "Testing configuration..."
if ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1; then
    echo "✓ Internet connectivity verified"
else
    echo "⚠ Could not reach internet (8.8.8.8)"
    echo "  Make sure WiFi is connected"
fi

if ping -c 1 -W 2 192.168.1.99 >/dev/null 2>&1; then
    echo "✓ Can reach rpi2 (DNS server)"
else
    echo "⚠ Cannot reach rpi2 (192.168.1.99)"
fi
echo ""

echo "========================================="
echo "NAT setup complete!"
echo ""
echo "Other nodes in EDGE mode will now be able to:"
echo "  1. Use ook (192.168.1.90) as their gateway"
echo "  2. Access the internet through ook's WiFi"
echo ""
echo "Note: Nodes need to get new DHCP lease to use ook as gateway."
echo "This happens automatically when switching to EDGE mode."
echo "========================================="