#!/bin/bash
# Validation script for centralized DNS control configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TATBOT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔍 Validating DNS control configurations..."

# Check if required files exist
echo "📄 Checking configuration files..."

CONFIG_FILES=(
    "$TATBOT_ROOT/config/dnsmasq/mode-home.conf"
    "$TATBOT_ROOT/config/dnsmasq/mode-edge.conf"
    "$TATBOT_ROOT/src/tatbot/utils/mode_toggle.py"
    "$TATBOT_ROOT/src/tatbot/utils/network_config.py"
    "$TATBOT_ROOT/ip_addresses_dump.md"
)

for file in "${CONFIG_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  ✅ $file"
    else
        echo "  ❌ Missing: $file"
        exit 1
    fi
done

# Validate dnsmasq configurations locally
echo ""
echo "🔧 Validating dnsmasq configurations..."

if command -v dnsmasq >/dev/null 2>&1; then
    for config in "$TATBOT_ROOT/config/dnsmasq/mode-"*.conf; do
        if dnsmasq --test --conf-file="$config" >/dev/null 2>&1; then
            echo "  ✅ $(basename "$config") - syntax valid"
        else
            echo "  ❌ $(basename "$config") - syntax error"
            dnsmasq --test --conf-file="$config"
            exit 1
        fi
    done
else
    echo "  ⚠️  dnsmasq not installed locally, skipping syntax validation"
fi

# Check device inventory
echo ""
echo "📊 Validating device inventory..."

DEVICE_COUNT=$(uv run python -c "
from tatbot.utils.network_config import NetworkConfig
config = NetworkConfig()
config.parse_ip_dump()
print(len(config.devices))
" 2>/dev/null)

if [[ "$DEVICE_COUNT" -gt 10 ]]; then
    echo "  ✅ Found $DEVICE_COUNT devices in inventory"
else
    echo "  ❌ Only found $DEVICE_COUNT devices, expected more than 10"
    exit 1
fi

# Check nodes.yaml has MAC addresses
echo ""
echo "🔌 Checking nodes.yaml has MAC addresses..."

MAC_COUNT=$(grep -c "mac:" "$TATBOT_ROOT/src/conf/nodes.yaml" || echo "0")
if [[ "$MAC_COUNT" -gt 5 ]]; then
    echo "  ✅ Found MAC addresses for $MAC_COUNT nodes"
else
    echo "  ❌ Missing MAC addresses in nodes.yaml"
    exit 1
fi

# Validate SSH key exists
echo ""
echo "🔑 Checking SSH key..."

SSH_KEY="$HOME/.ssh/tatbot-key"
if [[ -f "$SSH_KEY" ]]; then
    echo "  ✅ SSH key found: $SSH_KEY"
else
    echo "  ❌ SSH key not found: $SSH_KEY"
    echo "      Run: uv run python -m tatbot.utils.net --debug"
    exit 1
fi

# Test rpi2 connectivity (without making changes)
echo ""
echo "📡 Testing rpi2 connectivity..."

if ssh -o ConnectTimeout=5 -o BatchMode=yes -i "$SSH_KEY" rpi2@192.168.1.99 "echo 'Connection test successful'" 2>/dev/null; then
    echo "  ✅ Can connect to rpi2"
else
    echo "  ❌ Cannot connect to rpi2"
    echo "      Check network connectivity and SSH keys"
    exit 1
fi

# Check that dnsmasq configs contain expected entries
echo ""
echo "📋 Validating configuration content..."

HOME_CONFIG="$TATBOT_ROOT/config/dnsmasq/mode-home.conf"
EDGE_CONFIG="$TATBOT_ROOT/config/dnsmasq/mode-edge.conf"

# Check home mode config
if grep -q "server=192.168.1.1" "$HOME_CONFIG" && \
   grep -q "address=/ook.tatbot.local/" "$HOME_CONFIG"; then
    echo "  ✅ Home mode config contains expected entries"
else
    echo "  ❌ Home mode config missing expected entries"
    exit 1
fi

# Check edge mode config  
if grep -q "dhcp-host=" "$EDGE_CONFIG" && \
   grep -q "dhcp-range=" "$EDGE_CONFIG" && \
   grep -q "address=/ook.tatbot.local/" "$EDGE_CONFIG"; then
    echo "  ✅ Edge mode config contains expected entries"
else
    echo "  ❌ Edge mode config missing expected entries"
    exit 1
fi

# Count DHCP reservations
DHCP_COUNT=$(grep -c "dhcp-host=" "$EDGE_CONFIG")
if [[ "$DHCP_COUNT" -gt 10 ]]; then
    echo "  ✅ Found $DHCP_COUNT DHCP reservations"
else
    echo "  ❌ Only found $DHCP_COUNT DHCP reservations, expected more than 10"
    exit 1
fi

echo ""
echo "🎉 All validations passed!"
echo ""
echo "Next steps:"
echo "1. Review the generated configurations in config/dnsmasq/"
echo "2. When ready, run: ./scripts/setup_dns_control.sh"
echo "3. Follow the manual setup instructions in docs/nodes.md"
echo ""
echo "The system is ready for deployment."