CanaKit Raspberry Pi 5 Basic Kit (8GB RAM | NO SD Card)

Argon NEO 5 Case for Raspberry Pi 5 | Aluminum case with Built-in Fan (Black-Black)

SanDisk 128GB Extreme microSDXC UHS-I Memory Card with Adapter - Up to 190MB/s, C10, U3, V30, 4K, 5K, A2, Micro SD Card - SDSQXAA-128G-GN6MA

Each rpi is connected to power and ethernet via a  single ethernet cable that provides power and ethernet.

A splitter is used to split the ethernet into the ethernet port, and the power into the usb-c port.

Industrial Gigabit PoE Splitter for Raspberry Pi 5, Onboard MPS Control Chip, 802.3af/at-Compliant 37V ~ 57V Input, 5V 5A Output, Type-C Power Output Port

installation instructions (ssh into fresh install):

```bash
sudo apt-get update
echo "🐳 installing docker"
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
```

add github ssh keys
https://github.com/settings/ssh/new

```bash
ls -al ~/.ssh
mkdir -p ~/.ssh
ssh-keygen -t ed25519 -C "hu.po.xyz@gmail.com"
cat ~/.ssh/id_ed25519.pub
```

```bash
mkdir ~/dev
cd ~/dev && git clone git@github.com:hu-po/tatbot-dev.git
```

disable wifi and bluetooth

```bash
sudo nano /boot/firmware/config.txt
```
add these lines

```txt
# Disable wifi and bluetooth
dtoverlay=disable-wifi
dtoverlay=disable-bt
```