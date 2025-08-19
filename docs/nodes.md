# 🖥️ Nodes

Tatbot consists of several computers, cameras, and robots connected via ethernet in a distributed computing architecture.

## 💻 Compute Nodes

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card}
:class-header: bg-light

{{ojo}} **ojo**
^^^
NVIDIA Jetson AGX Orin
- ARM Cortex-A78AE, 12-core @ 2.2 GHz
- 32GB Unified RAM
- 200 TOPS AI performance
- Agent model inference
:::

:::{grid-item-card}
:class-header: bg-light

{{ook}} **ook**
^^^
Acer Nitro V 15
- Intel i7-13620H, 16-core @ 3.6 GHz
- 16GB RAM + 6GB VRAM (RTX 4050)
- 194 TOPS AI performance
- GPU-accelerated batch IK
:::

:::{grid-item-card}
:class-header: bg-light

{{eek}} **eek**
^^^
System76 Meerkat PC
- Intel i5-1340P, 16-core @ 4.6 GHz  
- 15GB RAM
- NFS server & shared storage
:::

:::{grid-item-card}
:class-header: bg-light

{{hog}} **hog**
^^^
GEEKOM GT1 Mega
- Intel Core Ultra 9 185H, 16-core @ 5.1 GHz
- 32GB RAM + Intel Arc graphics
- IP: 192.168.1.88
- Robot arm control & RealSense cameras
:::

:::{grid-item-card}
:class-header: bg-light

{{rpi1}} **rpi1**
^^^
Raspberry Pi 5
- ARM Cortex-A76, 4-core @ 2.4 GHz
- 8GB RAM
- Visualization
- Opencode Agent Frontend
:::

:::{grid-item-card}
:class-header: bg-light

{{rpi2}} **rpi2**
^^^
Raspberry Pi 5
- ARM Cortex-A76, 4-core @ 2.4 GHz
- 8GB RAM
- DNS control server (centralized mode switching)
:::

::::

```{admonition} Modes: Edge and Home
:class: tip

**Home Mode:** Nodes are on the home network. {{rpi2}} forwards DNS to the home router for `tatbot.lan`. Gateway: 192.168.1.1 (home router).

**Edge Mode:** Local-first operation. {{rpi2}} provides DNS and DHCP for `tatbot.lan`. Gateway: 192.168.1.90 ({{ook}} via WiFi NAT). Internet access available through {{ook}}.

**Automatic Switching:** Mode changes are detected within 20 seconds and all nodes automatically get new DHCP leases with correct settings.
```

## 📷 Camera System
- `camera1` 📷: Amcrest IP PoE camera (5MP, 30fps)
- `camera2` 📷: Amcrest IP PoE camera (5MP, 30fps)
- `camera3` 📷: Amcrest IP PoE camera (5MP, 30fps)
- `camera4` 📷: Amcrest IP PoE camera (5MP, 30fps)
- `camera5` 📷: Amcrest IP PoE camera (5MP, 30fps)
- `realsense1` 📷: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `realsense2` 📷: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `switch-lan`: 8-port gigabit ethernet switch
- `switch-poe`: 8-port gigabit PoE switch
- `arm-l` 🦾: Trossen Arm Controller box (back) connected to WidowXAI arm
- `arm-r` 🦾: Trossen Arm Controller box (front) connected to WidowXAI arm
- `display` 🖥️: touchscreen monitor w/ speakers



see:

- `src/tatbot/data/node.py`
- `src/conf/nodes.yaml`
- `src/tatbot/utils/net.py`

## Network

tatbot uses shared ssh keys for easy communication

- `switch-lan`:
    - (1) short black ethernet cable to `hog`
    - (2) short black ethernet cable to `ojo`
    - (3) blue ethernet cable to `arm-l` controller box
    - (4) blue ethernet cable to `arm-r` controller box
    - (5) long black ethernet cable to `ook`
    - (6) short black ethernet cable to `eek`
    - (7) *home mode* long ethernet cable to `oop`
    - (8) short black ethernet cable to `switch-poe`
- `switch-poe`:
    - (uplink-1) -
    - (uplink-2) short black ethernet cable to `switch-lan`/
    - (1) short black ethernet cable to `camera1`
    - (2) long black ethernet cable to `camera2`
    - (3) long black ethernet cable to `camera3`
    - (4) long black ethernet cable to `camera4`
    - (5) long black ethernet cable to `camera5`
    - (6) -
    - (7) short black ethernet cable to `rpi1`
    - (8) short black ethernet cable to `rpi2`

to setup the network:

```bash
cd ~/tatbot
source scripts/setup_env.sh
# Optional helper to inspect network state with Python (not required for mode switching)
uv run python -m tatbot.utils.net --debug || true
```

## NFS Setup

currently the `eek` node serves as the NFS server for all other nodes:

```bash
sudo apt install nfs-kernel-server
sudo mkdir -p /nfs/tatbot
sudo chmod 777 /nfs/tatbot
sudo nano /etc/exports
# add this line:
> /nfs/tatbot 192.168.1.0/24(rw,sync,no_subtree_check)
sudo exportfs -ra
sudo systemctl restart nfs-server
sudo exportfs -v
# enable on startup
sudo systemctl enable nfs-server
```

rest of the nodes:

```bash
sudo apt install nfs-common
showmount -e 192.168.1.97
sudo mkdir -p /nfs/tatbot
sudo mount -t nfs 192.168.1.97:/nfs/tatbot /nfs/tatbot
# enable on startup
sudo nano /etc/fstab
# add this line:
> 192.168.1.97:/nfs/tatbot /nfs/tatbot nfs defaults,_netdev 0 0
sudo systemctl daemon-reload
sudo mount -a
```

sometimes on reboot the NFS mount is not mounted, make sure to:

```bash
sudo systemctl daemon-reload
sudo mount -a
```

## Network Modes

tatbot operates in two modes that **automatically switch** based on home network availability with **automatic DHCP renewal** on all nodes:

- **Edge Mode**: Local-first operation when home LAN cable is disconnected. `rpi2` provides DNS and DHCP for the `tatbot.lan` network. `ook` provides internet access via WiFi NAT (gateway: 192.168.1.90).

- **Home Mode**: Integration with home network when home LAN cable is connected. `rpi2` forwards DNS queries to the home router while maintaining `tatbot.lan` hostname resolution. All nodes get DHCP from home router (gateway: 192.168.1.1).

**Automatic DHCP Renewal**: When modes switch, all nodes automatically get new DHCP leases with correct gateway/DNS settings within 20 seconds.

### Setup Instructions

**1. Setup DNS Control Node (rpi2) with Auto-Detection**

```bash
# SSH into rpi2
ssh rpi2

# Install dnsmasq for DNS/DHCP services
sudo apt update && sudo apt install -y dnsmasq

# Create configuration directories
sudo mkdir -p /etc/dnsmasq.d          # Active config directory
sudo mkdir -p /etc/dnsmasq-profiles   # Mode profile storage

# Copy mode configuration files to rpi2
cp ~/tatbot/config/network/dnsmasq/mode-*.conf /tmp/
sudo mv /tmp/mode-*.conf /etc/dnsmasq-profiles/

# Create initial symlink (will be managed by auto-detect service)
# Edge mode is safer default - won't conflict if home router present
sudo ln -sf /etc/dnsmasq-profiles/mode-edge.conf /etc/dnsmasq.d/active.conf

# Configure dnsmasq systemd override to follow active symlink
sudo mkdir -p /etc/systemd/system/dnsmasq.service.d
sudo cp ~/tatbot/config/network/systemd/dnsmasq.service.d/override.conf /etc/systemd/system/dnsmasq.service.d/override.conf

# Autoselect mode at start based on home router reachability (avoid DHCP conflict on boot)
sudo cp ~/tatbot/config/network/systemd/dnsmasq.service.d/edge-home-autoselect.conf /etc/systemd/system/dnsmasq.service.d/edge-home-autoselect.conf

# Enable and start dnsmasq
sudo systemctl daemon-reload
sudo systemctl enable dnsmasq
sudo systemctl restart dnsmasq

# Verify dnsmasq is running with our config
sudo systemctl status dnsmasq  # Should show "active (running)"

# The auto-detection script with DHCP triggers is at scripts/mode_auto_switcher_with_dhcp.sh
# This script automatically triggers DHCP renewal on all nodes when switching modes

# Install the auto-detection service that switches modes and triggers DHCP renewals
# This service monitors home router (192.168.1.1) availability
sudo cp ~/tatbot/config/network/systemd/tatbot-mode-auto.service /etc/systemd/system/tatbot-mode-auto.service

# Enable and start the auto-detection service
sudo systemctl daemon-reload
sudo systemctl enable tatbot-mode-auto.service
sudo systemctl start tatbot-mode-auto.service

# Verify auto-detection is running
sudo systemctl status tatbot-mode-auto.service  # Should show "active (running)"
# Watch the logs to see mode detection in action
sudo journalctl -u tatbot-mode-auto.service -f  # Ctrl+C to exit
```

**2. Setup WiFi Internet Sharing on ook (Edge Mode Internet Access)**

```bash
# SSH into ook - this node has WiFi that can provide internet in edge mode
ssh ook

# Run the NAT setup script (handles all configuration automatically)
cd ~/tatbot
sudo bash scripts/setup_nat_ook.sh

# The script will:
# - Enable IP forwarding
# - Configure NAT rules for WiFi → Ethernet sharing
# - Save rules to persist across reboots
# - Verify the configuration
```

**3. Configure Other Nodes (eek, hog, ojo, rpi1)**

```bash
# Install DNS utilities for testing
sudo apt install -y dnsutils

# Check which network manager is in use
systemctl list-units --type=service --state=active | grep -E '(NetworkManager|dhcpcd)'

# This ensures tatbot.lan resolution works in both modes
# Configure to use rpi2 DNS and accept DHCP from either rpi2 or home router
nmcli connection show --active
sudo nmcli connection modify 'Wired connection 1' \
  ipv4.dns '192.168.1.99' \
  ipv4.ignore-auto-dns yes \
  ipv4.method auto

# Apply the changes
sudo nmcli connection reload
sudo nmcli device reapply enp63s0 # on ook
sudo nmcli device reapply enp86s0 # on eek
sudo nmcli device reapply enp172s0 # on hog
sudo nmcli device reapply eno1 # on ojo
sudo nmcli device reapply eth0 # on rpi1, rpi2

# Test DNS resolution
nslookup ook.tatbot.lan 192.168.1.99  # Should resolve to 192.168.1.90
```

**4. Configure IP Cameras and Arm Controllers**

For IP Cameras:
1. Access device web interface (e.g., http://192.168.1.91 for camera1)
2. Navigate to Network Settings
3. Set Primary DNS: `192.168.1.99`
4. Keep DHCP enabled (will use home router or rpi2 automatically)
5. Save and reboot device

For Arm Control Boxes:
1. Set DNS in `config/trossen/arm-l.yaml` and `config/trossen/arm-r.yaml` to `192.168.1.99`
2. Push configs to arm controller boxes with `src/tatbot/bot/trossen_config.py`
3. Reboot arm controller boxes

### Operation and Monitoring

**Check Current Status**

```bash
# From any tatbot node, check network status
./scripts/network_status.sh

# Check which mode is currently active on rpi2
ssh rpi2 "readlink -f /etc/dnsmasq.d/active.conf"
# Output: /etc/dnsmasq-profiles/mode-edge.conf (or mode-home.conf)

# Monitor auto-detection service logs on rpi2
ssh rpi2 "sudo journalctl -u tatbot-mode-auto.service -n 20"
# Shows recent mode switches and home router detection status

# Check current mode (simple check)
ssh rpi2 "sudo systemctl status tatbot-mode-auto.service | tail -5"
```

**Manual Mode Override (if needed)**

```bash
# The auto-detection service normally handles this automatically
# To manually override, you can directly change the symlink on rpi2:

# Force switch to Home mode
ssh rpi2 "sudo ln -sf /etc/dnsmasq-profiles/mode-home.conf /etc/dnsmasq.d/active.conf && sudo systemctl reload dnsmasq"

# Force switch to Edge mode  
ssh rpi2 "sudo ln -sf /etc/dnsmasq-profiles/mode-edge.conf /etc/dnsmasq.d/active.conf && sudo systemctl reload dnsmasq"

# Note: Auto-detection will switch back within 20 seconds based on cable status
# To permanently override, stop the auto-detection service first:
ssh rpi2 "sudo systemctl stop tatbot-mode-auto.service"
```

**Troubleshooting**

```bash
# If modes aren't switching automatically:
ssh rpi2 "sudo systemctl status tatbot-mode-auto.service"
ssh rpi2 "ping -c 1 192.168.1.1"  # Test if home router is reachable

# If DHCP conflicts occur:
# Check which DHCP servers are active
sudo nmap --script broadcast-dhcp-discover

# If DNS isn't working:
nslookup ook.tatbot.lan 192.168.1.99  # Should always work
dig @192.168.1.99 ook.tatbot.lan      # More detailed DNS query

# If systemd-resolved is interfering with DNS resolution:
# Check if systemd-resolved is preventing proper domain routing
resolvectl status  # Shows DNS servers per interface
# If tatbot.lan queries are failing, disable systemd-resolved:
sudo systemctl stop systemd-resolved
sudo systemctl disable systemd-resolved
sudo rm /etc/resolv.conf
echo "nameserver 192.168.1.99" | sudo tee /etc/resolv.conf
echo "nameserver 192.168.1.1" | sudo tee -a /etc/resolv.conf

# If nodes have both WiFi and Ethernet active:
# Disable WiFi to prevent routing conflicts in edge mode
sudo nmcli radio wifi off
# Or set ethernet priority higher than WiFi
sudo nmcli connection modify 'Wired connection 1' connection.autoconnect-priority 100

# Reset everything to clean state:
ssh rpi2 "sudo systemctl restart dnsmasq tatbot-mode-auto.service"
```

#### Scenarios

**Power on tatbot nodes with no external internet**
- DESIRED BEHAVIOR: All nodes can see and access each other, no external internet access. Nodes can run mcp servers and use each other as clients.

**In Edge Mode and WiFi becomes available**
- DESIRED BEHAVIOR: `ook` can still see and access all other nodes, but it can now also access the outside internet via WiFi. From the perspective of the other nodes the system is still effectively in edge mode.

**In Edge Mode and we attach home LAN cable to switch-lan**
- DESIRED BEHAVIOR: All nodes switch to the home network, they can now access the outside internet and talk to home computers such as `oop`.
  
  The auto-detection service handles this automatically within 20 seconds, including triggering DHCP renewal on all nodes.

**Power on tatbot nodes with attached home LAN cable to switch-lan**
- DESIRED BEHAVIOR: All nodes connect to the home network, they can access the outisde internet and talk to home computers such as `oop`.

**In Home Mode and we detach home LAN cable from switch-lan**
- DESIRED BEHAVIOR: All nodes switch to the edge mode, they can access the internet through ook's WiFi (if available) but cannot talk to home computers such as `oop`. They can still see and access each other. MCP servers remain unaffected and continue running.
  
  The auto-detection service handles this automatically within 20 seconds, including triggering DHCP renewal on all nodes to use ook as gateway.