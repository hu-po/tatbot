Upgrading the software on Trossen Arm Controller boxes:

https://docs.trossenrobotics.com/trossen_arm/main/getting_started/software_setup.html#software-upgrade

specifications on robot

https://docs.trossenrobotics.com/trossen_arm/main/specifications.html

| Specification         | Value               |
|----------------------|--------------------|
| Degrees of Freedom  | 6                  |
| Payload Capacity    | 1.5 kg             |
| Weight             | 4 kg               |
| Reach              | 0.769 m            |
| Nominal Voltage    | 24 V               |
| Peak Current       | 15 A               |
| Communication      | UDP over Ethernet  |

| Joint  | Min Position [rad (deg)] | Max Position [rad (deg)] | Velocity [rad/s (deg/s)] | Effort [N*m] |
|--------|--------------------------|--------------------------|--------------------------|--------------|
| Joint 1 | -3.054 (-175)            | 3.054 (175)             | 3.14 (180)               | 27           |
| Joint 2 | 0 (0)                    | 3.14 (180)              | 3.14 (180)               | 27           |
| Joint 3 | 0 (0)                    | 4.712 (270)             | 3.14 (180)               | 27           |
| Joint 4 | -1.57 (-90)              | 1.57 (90)               | 3.14 (180)               | 7            |
| Joint 5 | -1.57 (-90)              | 1.57 (90)               | 3.14 (180)               | 7            |
| Joint 6 | -3.14 (-180)             | 3.14 (180)              | 3.14 (180)               | 7            |

logged in to trossen-ai-pc, changed password from default password `trossen-ai`

```bash
sudo passwd
```

set up ssh

```bash
sudo apt install openssh-server
sudo systemctl enable ssh
```

add github ssh keys
https://github.com/settings/ssh/new

```bash
ssh-keygen -t ed25519 -C "hu.po.xyz@gmail.com"
cat ~/.ssh/id_ed25519.pub
```

```bash
mkdir ~/dev
cd ~/dev && git clone git@github.com:hu-po/tatbot-dev.git
git config --global user.email "hu.po.xyz@gmail.com"
git config --global user.name "hu-po"
./scripts/specs.sh
```

install some basic tools

```bash
sudo apt install arp-scan
sudo apt-get install vim
```

make sure to use the right conda environment

```bash
cd /home/trossen-ai/libtrossen_arm/demos/python
conda activate trossen_ai_data_collection_ui_env
```

install realsense

```bash
sudo apt  install curl
sudo apt-get install apt-transport-https
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
sudo apt-get install librealsense2-dev
sudo apt-get install librealsense2-dbg
sudo reboot
modinfo uvcvideo | grep "version:" # should have `realsense`
realsense-viewer
```

install docker

```bash
sudo apt-get update
echo "🐳 installing docker"
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
```