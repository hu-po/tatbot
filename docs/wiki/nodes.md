# Nodes

tatbot consists of several computers, cameras, and robots connected via ethernet:

- `ojo` 🦎: NVIDIA Jetson AGX Orin (ARM Cortex-A78AE, 12-core @ 2.2 GHz) (32GB Unified RAM) (200 TOPS)
- `ook` 🦧: Acer Nitro V 15 w/ NVIDIA RTX 4050 (Intel i7-13620H, 16-core @ 3.6 GHz) (16GB RAM) (6GB VRAM) (194 TOPS)
- `trossen-ai` 🦾: System76 Meerkat PC (Intel i5-1340P, 16-core @ 4.6 GHz) (15GB RAM)
- `rpi1` 🍓: Raspberry Pi 5 (ARM Cortex-A76, 4-core @ 2.4 GHz) (8GB RAM)
- `rpi2` 🍇: Raspberry Pi 5 (ARM Cortex-A76, 4-core @ 2.4 GHz) (8GB RAM)
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

during development *dev mode*, the following pc is also available:

- `oop` 🦊: Ubuntu PC w/ NVIDIA RTX 3090 (AMD Ryzen 9 5900X, 24-core @ 4.95 GHz) (66GB RAM) (24GB VRAM) (TOPS)

see:

- `tatbot/tatbot/data/node.py`
- `tatbot/config/nodes.yaml`
- `tatbot/tatbot/utils/net.py`
- [`paramiko`](https://github.com/paramiko/paramiko)

## Network

tatbot uses shared ssh keys for easy communication

- `switch-lan`:
    - (1) short black ethernet cable to `trossen-ai`
    - (2) short black ethernet cable to `ojo`
    - (3) blue ethernet cable to `arm-l` controller box
    - (4) blue ethernet cable to `arm-r` controller box
    - (5) long black ethernet cable to `ook`
    - (6) 
    - (7) *dev mode* long ethernet cable to `oop`
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
source scripts/setup-env.sh
uv run -m tatbot.utils.net --debug
```

# NFS Setup

currently the `rpi2` node serves as the NFS server for all other nodes:

```bash
sudo apt install nfs-kernel-server
sudo mkdir -p /home/rpi2/tatbot/nfs
sudo chmod 777 /home/rpi2/tatbot/nfs
sudo nano /etc/exports
# add this line:
> /home/rpi2/tatbot/nfs 192.168.1.0/24(rw,sync,no_subtree_check)
sudo exportfs -ra
sudo systemctl restart nfs-server
sudo exportfs -v
# enable on startup
sudo systemctl enable nfs-server
```

rest of the nodes `rpi1`, `trossen-ai`, `ook`, `ojo`, and `oop`:

```bash
sudo apt install nfs-common
showmount -e 192.168.1.99
mkdir -p ~/tatbot/nfs
sudo mount -t nfs 192.168.1.99:/home/rpi2/tatbot/nfs ~/tatbot/nfs
# enable on startup
sudo nano /etc/fstab
# add this line:
> 192.168.1.99:/home/rpi2/tatbot/nfs /home/<USERNAME>/tatbot/nfs nfs defaults,nolock,vers=3,_netdev 0 0
sudo systemctl daemon-reload
sudo mount -a
```