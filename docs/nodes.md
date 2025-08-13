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

**Home:** Nodes are on the home network. {{rpi2}} forwards DNS to the home router for `tatbot.lan`.

**Edge:** Local-first operation. {{rpi2}} is authoritative DNS for `tatbot.lan` (and can provide DHCP if enabled). Internet is optional.
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
- [`paramiko`](https://github.com/paramiko/paramiko)

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
uv run python -m tatbot.utils.net --debug
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

## Network Modes

tatbot operates in two simple modes:

- **Edge Mode (Default)**: Local-first operation. `rpi2` provides DNS and DHCP for the `tatbot.lan` network. Nodes boot to this mode by default and fall back to it when WiFi is unavailable. Internet access is optional for `ook` via NAT using Wifi.

- **Home Mode**: Integration with home network. `rpi2` forwards DNS queries to the home router while maintaining `tatbot.lan` hostname resolution. Used when WiFi is available and home integration is desired.

**Setup DNS Control Node (rpi2)**

```bash
ssh rpi2
sudo apt update && sudo apt install -y dnsmasq
sudo mkdir -p /etc/dnsmasq.d
# copy configs to rpi2
cp ~/tatbot/config/dnsmasq/mode-*.conf /tmp/
# Create profiles directory and move configs there
sudo mkdir -p /etc/dnsmasq-profiles
sudo mv /tmp/mode-*.conf /etc/dnsmasq-profiles/
# Create active config symlink (default to EDGE mode on boot)
sudo ln -sf /etc/dnsmasq-profiles/mode-edge.conf /etc/dnsmasq.d/active.conf
# Override systemd service to use our configuration method
sudo mkdir -p /etc/systemd/system/dnsmasq.service.d
cat <<EOF | sudo tee /etc/systemd/system/dnsmasq.service.d/override.conf
[Service]
ExecStart=
ExecStart=/usr/sbin/dnsmasq -x /run/dnsmasq/dnsmasq.pid -u dnsmasq --conf-file=/etc/dnsmasq.d/active.conf --local-service
EOF
# Reload systemd and restart dnsmasq
sudo systemctl daemon-reload
sudo systemctl enable dnsmasq && sudo systemctl restart dnsmasq
# Verify it's using our configuration (should show interface binding and our forwarding rules)
sudo systemctl status dnsmasq
# Enable NAT on rpi2 for internet access in Edge mode via wifi
# Enable IP forwarding and basic NAT (replace OUT_IFACE if needed)
echo 'net.ipv4.ip_forward=1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -w net.ipv4.ip_forward=1
OUT_IFACE=$(ip route | awk '/default/ {print $5; exit}')
sudo iptables -t nat -F
sudo iptables -t filter -F
sudo iptables -A FORWARD -i eth0 -j ACCEPT
sudo iptables -A FORWARD -o eth0 -j ACCEPT
sudo iptables -t nat -A POSTROUTING -o "$OUT_IFACE" -j MASQUERADE
if command -v iptables-save >/dev/null; then
  sudo mkdir -p /etc/iptables
  sudo iptables-save | sudo tee /etc/iptables/rules.v4 >/dev/null
fi
```

**Setup Other Nodes (eek, hog, ook, ojo, rpi1)**

```bash
# install dependencies
sudo apt install dnsutils
# First, check which network management system is active:
systemctl list-units --type=service --state=active | grep -E '(NetworkManager|dhcpcd)'
# Check active connections
nmcli connection show --active
# Configure Wi‑Fi to use rpi2 for DNS and keep default route preferred
# Replace SSID_CONN_NAME with your Wi‑Fi connection name
sudo nmcli connection modify 'SSID_CONN_NAME' ipv4.dns '192.168.1.99' ipv4.ignore-auto-dns yes ipv4.never-default no ipv4.route-metric 600
sudo nmcli connection down 'SSID_CONN_NAME' && sudo nmcli connection up 'SSID_CONN_NAME'
# Configure Ethernet to use rpi2 for DNS but avoid adding a competing default route
sudo nmcli connection modify 'Wired connection 1' ipv4.dns '192.168.1.99' ipv4.ignore-auto-dns yes ipv4.never-default yes
sudo nmcli device reapply $(nmcli -t -f DEVICE,TYPE device status | awk -F: '$2=="ethernet"{print $1; exit}') || true
```

For IP cameras and arm controllers, configure DNS via web interface to `192.168.1.99`.

**Toggle Mode and Check Status**

```bash
# Check current status
./scripts/network_status.sh
# Determine current mode
cd ~/tatbot && source scripts/setup_env.sh
uv run python src/tatbot/utils/mode_toggle.py --mode status
# Switch to Home mode (DNS forwarder to home router)
uv run python src/tatbot/utils/mode_toggle.py --mode home
# Switch to Edge mode (local DNS + DHCP, default mode)
uv run python src/tatbot/utils/mode_toggle.py --mode edge
```

#### Scenarios

**Power on tatbot nodes with no external internet**
- DESIRED BEHAVIOR: All nodes can see and access each other, no external internet access. Nodes can run mcp servers and use each other as clients.
- All nodes powered off, turning on one-by-one, no internet
- rpi2 boots → starts dnsmasq in edge mode (default symlink)
- rpi2 provides DHCP (192.168.1.100-200) and DNS for tatbot.lan
- Each node boots → gets IP from rpi2's DHCP
- Nodes can resolve each other (ook.tatbot.lan → 192.168.1.90)

**In Edge Mode and WiFi becomes available**
- DESIRED BEHAVIOR: `ook` can still see and access all other nodes, but it can now also access the outside internet. From the perspective of the other nodes the system is still effectively in edge mode.
- On `ook`, NAT is enabled to allow internet access via WiFi

**In Edge Mode and we attach home LAN cable to switch-lan**
- DESIRED BEHAVIOR: All nodes switch to the home network, they can now access the outisde internet and talk to home computers such as `oop`.

**Power on tatbot nodes with attached home LAN cable to switch-lan**
- DESIRED BEHAVIOR: All nodes connect to the home network, they can access the outisde internet and talk to home computers such as `oop`.

**In Home Mode and we detach home LAN cable from switch-lan**
- DESIRED BEHAVIOR: All nodes switch to the edge mode, they can no longer access the outisde internet and talk to home computers such as `oop`, but they can still see and access each other. Ideally MCP servers are unaffected and still running.