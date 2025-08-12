#!/bin/bash
# Show current network status and connectivity on ook

echo "🌐 NETWORK STATUS"
echo "================"

# Basic system info
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

# Network interfaces
echo "📡 Network Interfaces:"
echo "----------------------"
ip addr show | grep -E '^[0-9]+:|inet ' | sed 's/^[[:space:]]*/  /'
echo ""

# Active connections
echo "🔗 Active NetworkManager Connections:"
echo "------------------------------------"
nmcli connection show --active
echo ""

# Default route
echo "🛣️  Default Route:"
echo "-----------------"
ip route show default
echo ""

# DNS configuration
echo "🔍 DNS Configuration:"
echo "--------------------"
echo "  /etc/resolv.conf:"
cat /etc/resolv.conf | sed 's/^/    /'
if command -v resolvectl >/dev/null 2>&1; then
  echo ""
  echo "  systemd-resolved status:"
  resolvectl status 2>/dev/null | sed 's/^/    /' | head -n 50
fi
echo ""

# Test connectivity to key hosts
echo "🧪 Connectivity Tests:"
echo "---------------------"

# Test tatbot network connectivity
echo "  Testing tatbot network..."
if ping -c 1 -W 2 192.168.1.99 &>/dev/null; then
    echo "    ✅ rpi2 (192.168.1.99) - reachable"
    
    # Test DNS resolution in tatbot network
    if nslookup eek.tatbot.lan 192.168.1.99 &>/dev/null 2>&1; then
        echo "    ✅ tatbot DNS resolution - working"
    else
        echo "    ❌ tatbot DNS resolution - failed"
    fi
else
    echo "    ❌ rpi2 (192.168.1.99) - unreachable"
fi

if ping -c 1 -W 2 192.168.1.97 &>/dev/null; then
    echo "    ✅ eek (192.168.1.97) - reachable"
else
    echo "    ❌ eek (192.168.1.97) - unreachable"
fi

# Test internet connectivity
echo "  Testing internet connectivity..."
if ping -c 1 -W 3 8.8.8.8 &>/dev/null; then
    echo "    ✅ Internet (8.8.8.8) - reachable"
    
    if ping -c 1 -W 3 google.com &>/dev/null; then
        echo "    ✅ Internet DNS resolution - working"
    else
        echo "    ❌ Internet DNS resolution - failed"
    fi
else
    echo "    ❌ Internet (8.8.8.8) - unreachable"
fi

echo ""

# Determine current mode
echo "🎭 Current Mode Detection:"
echo "-------------------------"
CURRENT_CONNECTION=$(nmcli connection show --active | grep -E 'tatbot-demo|wifi' | head -1 | awk '{print $1}')

if [[ "$CURRENT_CONNECTION" == "tatbot-demo" ]]; then
    echo "  🎪 EDGE MODE - Connected to tatbot network"
elif nmcli connection show --active | grep -q wifi; then
    WIFI_SSID=$(nmcli connection show --active | grep wifi | head -1 | awk '{print $1}')
    echo "  📶 UPDATE MODE - Connected to wifi ($WIFI_SSID)"
else
    echo "  ❓ UNKNOWN MODE - No recognized connection active"
fi

echo ""

# MCP services check if in demo mode
if [[ "$CURRENT_CONNECTION" == "tatbot-demo" ]] && ping -c 1 -W 2 192.168.1.99 &>/dev/null; then
    echo "🔧 MCP Services Check:"
    echo "---------------------"
    
    # Check if we can reach other nodes' MCP servers
    for node in eek:5173 hog:5173 rpi1:5190; do
        NODE_NAME=$(echo $node | cut -d: -f1)
        NODE_PORT=$(echo $node | cut -d: -f2)
        NODE_IP=$(nslookup $NODE_NAME.tatbot.lan 192.168.1.99 2>/dev/null | grep -A1 "Name:" | tail -1 | awk '{print $2}')
        
        if [[ -n "$NODE_IP" ]] && timeout 2 bash -c "</dev/tcp/$NODE_IP/$NODE_PORT" 2>/dev/null; then
            echo "    ✅ $NODE_NAME MCP ($NODE_IP:$NODE_PORT) - accessible"
        else
            echo "    ❌ $NODE_NAME MCP ($NODE_IP:$NODE_PORT) - not accessible"
        fi
    done
fi

echo ""
echo "Status check complete."