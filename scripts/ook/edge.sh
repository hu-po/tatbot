#!/bin/bash
# Switch ook to tatbot edge mode (ethernet connection to tatbot network)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TATBOT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🎭 Switching to TATBOT EDGE mode (ethernet connection to tatbot network)..."

# Check if running on ook
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" != "ook" ]]; then
    echo "⚠️  Warning: This script is designed for ook node, but running on $HOSTNAME"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Disable wifi to avoid conflicts
echo "📡 Disabling wifi..."
nmcli radio wifi off || echo "  (wifi may not be available)"

# Check if tatbot-demo profile exists
if nmcli connection show tatbot-demo &>/dev/null; then
    echo "✅ Found existing tatbot-demo profile"
else
    echo "🔧 Creating tatbot-demo profile..."
    
    # Copy the profile configuration
    PROFILE_SOURCE="$TATBOT_ROOT/config/network/tatbot-demo.nmconnection"
    PROFILE_DEST="/etc/NetworkManager/system-connections/tatbot-demo.nmconnection"
    
    if [[ -f "$PROFILE_SOURCE" ]]; then
        sudo cp "$PROFILE_SOURCE" "$PROFILE_DEST"
        sudo chmod 600 "$PROFILE_DEST"
        sudo chown root:root "$PROFILE_DEST"
        echo "  ✅ Profile installed"
    else
        echo "  ❌ Profile template not found: $PROFILE_SOURCE"
        exit 1
    fi
    
    # Reload NetworkManager
    sudo nmcli connection reload
fi

# Get the ethernet interface name
ETH_INTERFACE=$(nmcli -t -f DEVICE,TYPE,STATE device status | awk -F: '$2=="ethernet" && $3=="connected" {print $1; exit}')
if [[ -z "$ETH_INTERFACE" ]]; then
    # Fallback: pick first ethernet device
    ETH_INTERFACE=$(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="ethernet" {print $1; exit}')
fi
if [[ -z "$ETH_INTERFACE" ]]; then
    echo "❌ No ethernet interface found"
    exit 1
fi

echo "🔌 Using ethernet interface: $ETH_INTERFACE"

# Update the profile with the correct interface
nmcli connection modify tatbot-demo connection.interface-name "$ETH_INTERFACE"

# Activate the tatbot-demo connection
echo "🚀 Activating tatbot-demo connection..."
nmcli connection up tatbot-demo

# Wait a moment for connection to establish
sleep 3

# Verify connection
echo "🔍 Verifying connection..."
if ping -c 1 -W 2 192.168.1.99 &>/dev/null; then
    echo "  ✅ Can reach rpi2 (DNS server)"
else
    echo "  ❌ Cannot reach rpi2 (192.168.1.99)"
    echo "     Check ethernet cable and rpi2 status"
fi

if nslookup eek.tatbot.local 192.168.1.99 &>/dev/null; then
    echo "  ✅ DNS resolution working (eek.tatbot.local)"
else
    echo "  ❌ DNS resolution failed"
    echo "     Check rpi2 dnsmasq service"
fi

echo ""
echo "🎉 EDGE MODE ACTIVE"
echo "   IP: 192.168.1.90"
echo "   DNS: rpi2 (192.168.1.99)"
echo "   Domain: tatbot.local"
echo ""
echo "To return to wifi mode: ./scripts/ook/wifi.sh"