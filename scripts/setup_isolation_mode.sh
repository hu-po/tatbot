#!/bin/bash
# Configure rpi2 for true isolation mode with NAT routing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🌐 Setting up rpi2 for true isolation mode...${NC}"

# Enable IP forwarding
echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -w net.ipv4.ip_forward=1

echo -e "${GREEN}✅ IP forwarding enabled${NC}"

# Configure iptables for NAT
# Clear existing rules
sudo iptables -t nat -F
sudo iptables -t filter -F

# Allow forwarding from tatbot network
sudo iptables -A FORWARD -i eth0 -j ACCEPT
sudo iptables -A FORWARD -o eth0 -j ACCEPT

# NAT masquerading for outbound traffic
# Note: Change wlan0 to the interface that provides internet access
INTERNET_IFACE=$(ip route | grep default | awk '{print $5}' | head -1)
if [ -n "$INTERNET_IFACE" ]; then
    sudo iptables -t nat -A POSTROUTING -o $INTERNET_IFACE -j MASQUERADE
    echo -e "${GREEN}✅ NAT configured for interface: $INTERNET_IFACE${NC}"
else
    echo -e "${YELLOW}⚠️ No internet interface found - NAT not configured${NC}"
fi

# Save iptables rules (method varies by distribution)
if command -v iptables-save >/dev/null; then
    sudo iptables-save | sudo tee /etc/iptables/rules.v4 >/dev/null
    echo -e "${GREEN}✅ iptables rules saved${NC}"
fi

# Install iptables-persistent to restore rules on boot
if ! dpkg -l | grep -q iptables-persistent; then
    echo -e "${YELLOW}📦 Installing iptables-persistent...${NC}"
    sudo apt update && sudo apt install -y iptables-persistent
fi

echo -e "${GREEN}🎯 True isolation mode setup complete!${NC}"
echo -e "${YELLOW}📋 To test:${NC}"
echo "  1. Disconnect ethernet cable from home router"
echo "  2. Restart all nodes"
echo "  3. rpi2 will serve as gateway for tatbot network"
echo "  4. Nodes should get DHCP from rpi2 and use it as gateway"