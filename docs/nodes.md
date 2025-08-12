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

## Modes

- **Home**: Nodes on home network. `rpi2` runs dnsmasq as a DNS forwarder (no DHCP). Internet via home router.
- **Edge**: Local-only by default. `rpi2` runs dnsmasq as authoritative DNS for `tatbot.lan`. DHCP may be enabled in `mode-edge.conf`. Internet can be provided either by the home router or optionally via NAT on `rpi2`.

#### Setup Instructions

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
# Create active config symlink (default to EDGE on boot)
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

# Optional: enable NAT on rpi2 if no home router is present and you want internet in Edge
# (not a separate mode, just an option)
~/tatbot/scripts/mode_setup_rpi2.sh
```

**Configure All Nodes to Use rpi2 as DNS (one-time)**

```bash
# install dependencies
sudo apt install dnsutils
# First, check which network management system is active:
systemctl list-units --type=service --state=active | grep -E '(NetworkManager|dhcpcd)'
# Check active connections
nmcli connection show --active
# Apply sane defaults for DNS and routes
~/tatbot/scripts/mode_setup_node.sh
```

For IP cameras and arm controllers, configure DNS via web interface to `192.168.1.99`.

#### Usage

```bash
# Check current status
cd ~/tatbot && source scripts/setup_env.sh
uv run python src/tatbot/utils/mode_toggle.py --mode status
# Switch to Home (DNS forwarder)
uv run python src/tatbot/utils/mode_toggle.py --mode home
# Switch to Edge (authoritative DNS; DHCP if enabled in config)
uv run python src/tatbot/utils/mode_toggle.py --mode edge
```

#### Configuration

**Home Mode** (`config/dnsmasq/mode-home.conf`):
- DNS forwarder to home router (192.168.1.1)
- Static hostname resolution for tatbot devices
- No DHCP service
- ⚠️ **Important**: Ensure your home router has DHCP reservations matching the static A records

**Edge Mode** (`config/dnsmasq/mode-edge.conf`):
- Authoritative DNS for `tatbot.lan`
- DHCP static reservations for devices (if enabled)
- Upstream DNS (1.1.1.1, 8.8.8.8) for external lookups
- Optional NAT on `rpi2` via `scripts/mode_setup_rpi2.sh` when home router is absent (not a separate mode)

#### Troubleshooting

**Validate dnsmasq configuration:**
```bash
ssh rpi2 "sudo dnsmasq --test --conf-file=/etc/dnsmasq.d/active.conf"
```

**Check dnsmasq status:**
```bash
ssh rpi2 "sudo systemctl status dnsmasq"
```

**View dnsmasq logs:**
```bash
ssh rpi2 "sudo journalctl -u dnsmasq -f"
```

**Test DNS resolution:**
```bash
# From any node, test tatbot.lan resolution
nslookup ook.tatbot.lan 192.168.1.99
nslookup camera1.tatbot.lan 192.168.1.99
```

**Check Current Status**:
```bash
./scripts/network_status.sh
```