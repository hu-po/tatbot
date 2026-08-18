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
if ping -c 1 -W 2 192.0.2.99 &>/dev/null; then
    echo "    ✅ rpi2 (192.0.2.99) - reachable"

    # Test DNS resolution in tatbot network (prefer dig A record, fallback to getent)
    if command -v dig >/dev/null 2>&1; then
        if dig +short A eek.tatbot.example @192.0.2.99 | grep -qE '^[0-9.]+'; then
            echo "    ✅ tatbot DNS resolution - working"
        else
            echo "    ❌ tatbot DNS resolution - failed"
        fi
    else
        if getent hosts eek.tatbot.example >/dev/null 2>&1; then
            echo "    ✅ tatbot DNS resolution - working"
        else
            echo "    ❌ tatbot DNS resolution - failed"
        fi
    fi
else
    echo "    ❌ rpi2 (192.0.2.99) - unreachable"
fi

if ping -c 1 -W 2 192.0.2.97 &>/dev/null; then
    echo "    ✅ eek (192.0.2.97) - reachable"
else
    echo "    ❌ eek (192.0.2.97) - unreachable"
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

# Determine current network state
echo "🎭 Network State:"
echo "-----------------"

# Detect if DNS is pointing at rpi2
DNS_USES_RPI2=false
if command -v resolvectl >/dev/null 2>&1; then
    if resolvectl status 2>/dev/null | grep -q "DNS Servers: .*192.0.2.99"; then
        DNS_USES_RPI2=true
    fi
else
    if grep -q "^nameserver[[:space:]]\+192.0.2.99" /etc/resolv.conf 2>/dev/null; then
        DNS_USES_RPI2=true
    fi
fi

if [[ "$DNS_USES_RPI2" == true ]]; then
    echo "  🧭 Tatbot DNS: rpi2 (192.0.2.99)"
fi

if nmcli connection show --active | grep -q wifi; then
    WIFI_SSID=$(nmcli connection show --active | grep wifi | head -1 | awk '{print $1}')
    echo "  📶 Wi‑Fi: $WIFI_SSID"
fi

if ! nmcli connection show --active | grep -q wifi && [[ "$DNS_USES_RPI2" != true ]]; then
    echo "  ❓ No Wi‑Fi detected and tatbot DNS not in use"
fi

echo ""

# MCP services check when rpi2 is reachable
if ping -c 1 -W 2 192.0.2.99 &>/dev/null; then
    echo "🔧 MCP Services Check:"
    echo "---------------------"
    
    # Check if we can reach other nodes' MCP servers
    for node in eek:8000 hog:8000 ook:8000 ojo:8000 rpi1:8000; do
        NODE_NAME=$(echo $node | cut -d: -f1)
        NODE_PORT=$(echo $node | cut -d: -f2)
        NODE_IP=$(nslookup $NODE_NAME.tatbot.example 192.0.2.99 2>/dev/null | grep -A1 "Name:" | tail -1 | awk '{print $2}')
        
        if [[ -n "$NODE_IP" ]] && timeout 2 bash -c "</dev/tcp/$NODE_IP/$NODE_PORT" 2>/dev/null; then
            echo "    ✅ $NODE_NAME MCP ($NODE_IP:$NODE_PORT) - accessible"
        else
            echo "    ❌ $NODE_NAME MCP ($NODE_IP:$NODE_PORT) - not accessible"
        fi
    done
fi

echo ""
echo "Status check complete."