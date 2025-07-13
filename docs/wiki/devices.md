# Devices

tatbot consists of several computers, cameras, and robotos connected via ethernet:

- `ojo` 🦎: NVIDIA Jetson AGX Orin (ARM Cortex-A78AE, 12-core @ 2.2 GHz) (32GB Unified RAM) (200 TOPS)
- `trossen-ai` 🦾: System76 Meerkat PC (Intel i5-1340P, 16-core @ 4.6 GHz) (15GB RAM)
- `ook` 🦧: Acer Nitro V 15 w/ NVIDIA RTX 4050 (Intel i7-13620H, 16-core @ 3.6 GHz) (16GB RAM) (6GB VRAM) (194 TOPS)
- `rpi1` 🍓: Raspberry Pi 5 (ARM Cortex-A76, 4-core @ 2.4 GHz) (8GB RAM)
- `rpi2` 🍇: Raspberry Pi 5 (ARM Cortex-A76, 4-core @ 2.4 GHz) (8GB RAM)
- `camera1` 📷: Amcrest PoE cameras (5MP)
- `camera2` 📷: Amcrest PoE cameras (5MP)
- `camera3` 📷: Amcrest PoE cameras (5MP)
- `camera4` 📷: Amcrest PoE cameras (5MP)
- `camera5` 📷: Amcrest PoE cameras (5MP)
- `realsense1` 📷: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `realsense2` 📷: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `switch-lan`: 8-port gigabit ethernet switch
- `switch-poe`: 8-port gigabit PoE switch
- `arm-l` 🦾: Trossen Arm Controller box (back) connected to WidowXAI arm
- `arm-r` 🦾: Trossen Arm Controller box (front) connected to WidowXAI arm

during development *dev mode*, the following pc is also available:

- `oop` 🦊: Ubuntu PC w/ NVIDIA RTX 3090 (AMD Ryzen 9 5900X, 24-core @ 4.95 GHz) (66GB RAM) (24GB VRAM) (TOPS) 