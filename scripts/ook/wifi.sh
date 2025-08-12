#!/bin/bash
# Switch ook to wifi update mode (external internet access)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TATBOT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📶 Switching to WIFI UPDATE mode..."

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

# Disconnect from tatbot network
echo "🔌 Disconnecting from tatbot network..."
nmcli connection down tatbot-demo 2>/dev/null || echo "  (already disconnected)"

# Enable wifi
echo "📡 Enabling wifi..."
nmcli radio wifi on

# Wait for wifi to come up
echo "⏳ Waiting for wifi to initialize..."
sleep 3

# Scan for networks
echo "🔍 Scanning for available networks..."
nmcli device wifi rescan || true
sleep 2

# Show available networks
echo ""
echo "Available WiFi Networks:"
echo "========================"
nmcli device wifi list --rescan-on=auto | head -20
echo ""

# Interactive wifi connection
echo "🌐 Connect to WiFi network..."
echo "You can either:"
echo "  1. Use existing saved connection"
echo "  2. Connect to new network"
echo ""

# Check for saved wifi connections
SAVED_WIFI=$(nmcli connection show | grep wifi | head -5)
if [[ -n "$SAVED_WIFI" ]]; then
    echo "Saved WiFi connections:"
    echo "$SAVED_WIFI"
    echo ""
fi

read -p "Enter WiFi SSID (or press Enter to list saved connections): " WIFI_SSID

if [[ -z "$WIFI_SSID" ]]; then
    # Show saved connections and let user pick
    echo ""
    echo "Saved connections:"
    nmcli connection show | grep -E '(wifi|wireless)'
    echo ""
    read -p "Enter connection name to activate: " CONNECTION_NAME
    
    if [[ -n "$CONNECTION_NAME" ]]; then
        echo "🔗 Connecting to saved network: $CONNECTION_NAME"
        nmcli connection up "$CONNECTION_NAME"
    else
        echo "❌ No connection specified"
        exit 1
    fi
else
    # Connect to new network
    echo "🔗 Connecting to: $WIFI_SSID"
    read -s -p "Enter password (or press Enter for open network): " WIFI_PASSWORD
    echo ""
    
    if [[ -n "$WIFI_PASSWORD" ]]; then
        nmcli device wifi connect "$WIFI_SSID" password "$WIFI_PASSWORD"
    else
        nmcli device wifi connect "$WIFI_SSID"
    fi
fi

# Wait for connection
echo "⏳ Waiting for connection..."
sleep 5

# Verify internet connectivity
echo "🔍 Verifying internet connection..."
if ping -c 1 -W 3 8.8.8.8 &>/dev/null; then
    echo "  ✅ Internet connectivity established"
else
    echo "  ❌ No internet connectivity"
    echo "     Check wifi password and network access"
fi

if ping -c 1 -W 3 google.com &>/dev/null; then
    echo "  ✅ DNS resolution working"
else
    echo "  ❌ DNS resolution failed"
fi

# Show connection status
IP_ADDRESS=$(ip route get 8.8.8.8 | grep -oP 'src \K\S+' || echo "unknown")
GATEWAY=$(ip route show default | grep -oP 'via \K\S+' | head -1 || echo "unknown")

echo ""
echo "🌐 WIFI UPDATE MODE ACTIVE"
echo "   IP: $IP_ADDRESS"
echo "   Gateway: $GATEWAY"
echo "   DNS: Auto (DHCP) + 1.1.1.1, 8.8.8.8"
echo ""
echo "You can now:"
echo "  - Download updates: git pull, apt update, pip install, etc."
echo "  - Access internet resources"
echo "  - Install new software"
echo ""
echo "To return to demo mode:"
echo "  1. Reconnect ethernet cable to tatbot network"
echo "  2. Run: ./scripts/demo_mode.sh"