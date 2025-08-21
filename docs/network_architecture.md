---
summary: Dual-mode networking design and operations
tags: [network]
updated: 2025-08-21
audience: [dev, operator]
---

# 🔗 Network Architecture

## 🔍 Overview

Tatbot uses a sophisticated dual-mode networking system that automatically adapts based on network conditions with seamless failover and automatic configuration management.

## 🎨 Principles

- **Automatic Mode Detection**: Zero manual intervention for mode switches
- **Seamless Failover**: Sub-20 second transitions between modes
- **Internet Resilience**: Always-available internet access when possible
- **DNS Continuity**: `.tatbot.lan` domains work in both modes
- **Self-Healing**: Automatic DHCP renewal and configuration repair

## 🌐 Network Modes

### Home Mode
**Triggered when**: Home LAN cable connected to `switch-lan`

- **DHCP**: Home router (192.168.1.1) provides IP addresses
- **Gateway**: 192.168.1.1 (home router)
- **DNS**: rpi2 (192.168.1.99) forwards queries to home router
- **Internet**: Full access via home network
- **Scope**: Access to home computers (e.g., oop)

### Edge Mode
**Triggered when**: Home LAN cable disconnected from `switch-lan`

- **DHCP**: rpi2 (192.168.1.99) provides IP addresses
- **Gateway**: ook (192.168.1.90) with WiFi NAT
- **DNS**: rpi2 (192.168.1.99) with upstream to 1.1.1.1/8.8.8.8
- **Internet**: Available via ook's WiFi connection
- **Scope**: Isolated tatbot network with optional internet

## 🖥️ Key Components

### rpi2: DNS/DHCP Controller
- **Role**: Central network coordination
- **Services**: dnsmasq with mode-aware configuration
- **Auto-Detection**: Monitors home router availability every 20 seconds
- **DHCP Orchestration**: Triggers automatic renewal on all nodes during mode switches

### ook: Edge Gateway
- **Role**: Internet gateway in Edge mode
- **NAT Setup**: WiFi → Ethernet forwarding using iptables
- **IP Forwarding**: Enabled with persistent rules
- **Failover**: Provides internet when home network unavailable

### Other Nodes (eek, hog, ojo, rpi1)
- **Configuration**: Use rpi2 as DNS server (192.168.1.99)
- **DHCP Client**: Accept leases from either rpi2 or home router
- **Auto-Renewal**: Receive new network configuration automatically

## 🛠️ Automatic Features

### Mode Detection
- **Trigger**: Home router (192.168.1.1) reachability test
- **Frequency**: Every 20 seconds via `tatbot-mode-auto.service`
- **Script**: `scripts/mode_auto_switcher_with_dhcp.sh`

### DHCP Renewal Orchestration
- **Automatic**: Triggered on every mode switch
- **Method**: SSH-based `dhclient` commands to all nodes
- **Parallel Execution**: All nodes renewed simultaneously
- **Fallback**: 5-minute lease timeout ensures eventual consistency

### DNS Resolution
- **Tatbot Domains**: Always resolves `.tatbot.lan` addresses
- **Internet Domains**: Forwarded appropriately per mode
- **Static Entries**: All tatbot devices have fixed `.tatbot.lan` names

## 🌐 IP Addressing

### Static Reservations
```text
ook:     192.168.1.90  # Gateway in Edge mode
eek:     192.168.1.97  # NFS server
hog:     192.168.1.88  # Robot control
ojo:     192.168.1.96  # AI inference
rpi1:    192.168.1.98  # Visualization
rpi2:    192.168.1.99  # DNS/DHCP server
camera1-5: 192.168.1.91-95
arms:    192.168.1.2-3
```

### DHCP Ranges
- **Edge Mode**: 192.168.1.2-254 (covers static reservations)
- **Home Mode**: Delegated to home router

## 📁 Configuration Files

### Config Files
- `config/network/dnsmasq/mode-edge.conf` - Edge mode DNS/DHCP
- `config/network/dnsmasq/mode-home.conf` - Home mode DNS forwarding  
- `config/network/systemd/tatbot-mode-auto.service` - Auto-detection service

### Scripts
- `scripts/mode_auto_switcher_with_dhcp.sh` - Main mode detection and switching
- `scripts/setup_nat_ook.sh` - NAT configuration for ook
- `scripts/network_status.sh` - Network diagnostics and status

## 🌐 Network Flow

### Edge Mode Internet Path
```text
Node → ook (192.168.1.90) → WiFi NAT → Internet
```

### Home Mode Internet Path  
```text
Node → Home Router (192.168.1.1) → Internet
```

### DNS Resolution Path (Both Modes)
```text
Node → rpi2 (192.168.1.99) → [Home Router | Upstream DNS]
```

## 📊 Monitoring

### Status Checking
```bash
# Check current mode
ssh rpi2 "readlink -f /etc/dnsmasq.d/active.conf"

# Monitor mode switching
ssh rpi2 "sudo journalctl -u tatbot-mode-auto.service -f"

# Network status from any node
./scripts/network_status.sh
```

### Troubleshooting
- **Mode Detection Issues**: Check `tatbot-mode-auto.service` logs
- **DHCP Problems**: Verify lease files and dnsmasq status  
- **Internet Issues**: Verify ook's WiFi and NAT configuration
- **DNS Problems**: Test resolution with `nslookup <host>.tatbot.lan 192.168.1.99`

## 🔒 Security Considerations

- **Network Isolation**: Edge mode isolates tatbot from home network
- **Minimal Attack Surface**: Only required ports and services exposed
- **Automatic Updates**: Network configuration stays current without manual intervention
- **Failsafe Design**: Degrades gracefully when components unavailable

## 🚀 Roadmap

- **VLAN Segmentation**: Further isolate device types
- **Certificate Management**: TLS for inter-node communication
- **Load Balancing**: Multiple internet gateways in edge mode
- **Monitoring Dashboard**: Real-time network status visualization
