# Tech

technical description of tatbot

## Compute

tatbot consists of multiple seperate compute nodes connected via ethernet:

- `ojo`: NVIDIA Jetson AGX Orin (12-core ARM Cortex-A78AE @ 2.2 GHz) (32GB Unified RAM) (200 TOPS)
- `trossen-ai`: System76 Meerkat PC (13th Gen Intel i5-1340P, 16-core @ 4.6GHz) (15GB RAM)
- `rpi1`: Raspberry Pi 5 (4-core ARM Cortex-A76 @ 2.4 GHz) (8GB RAM)
- `rpi2`: Raspberry Pi 5 (4-core ARM Cortex-A76 @ 2.4 GHz) (8GB RAM)
- `camera1`: Amcrest PoE cameras (5MP)
- `camera2`: Amcrest PoE cameras (5MP)
- `camera3`: Amcrest PoE cameras (5MP)
- `camera4`: Amcrest PoE cameras (5MP)
- `camera5`: Amcrest PoE cameras (5MP)
- `cam_wrist`: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `cam_main`: Intel Realsense D405 (1280x720 RGBD, 90fps)
- `switch-main`: 5-port gigabit ethernet switch
- `switch-poe`: 8-port gigabit PoE switch
- `arm-leader`: Trossen Arm Controller box connected to leader arm
- `arm-follower`: Trossen Arm Controller box connected to follower arm
- `oop`: (only used for development)

## Networking

- `switch-main`: 
    (1) short black ethernet cable to `switch-poe`
    (2) short black ethernet cable to `trossen-ai`
    (3) short black ethernet cable to `ojo`
    (4) blue ethernet cable to `arm-follower` controller box
    (5) blue ethernet cable to `arm-leader` controller box
- `switch-poe`: 
    (1) long black ethernet cable to `camera1`
    (2) long black ethernet cable to `camera2`
    (3) long black ethernet cable to `camera3`
    (4) long black ethernet cable to `camera4`
    (5) long black ethernet cable to `camera5`
    (6) -
    (7) short black ethernet cable to `rpi1`
    (8) short black ethernet cable to `rpi2`
    (uplink-1) -
    (uplink-2) short black ethernet cable to `switch-main`

## Devices

the following devices are connected to each compute node:

- `oop`
    - `cam_wrist` via usb3 port
- `trossen-ai`
    - `cam_main` via usb3 port
    - wireless keyboard via usb2 port
    - touchscreen display via usbc port

## Startup

1. flip power strip in back to on
2. press power button on `trossen-ai`, it will glow blue
3. flip rocker switches to "on" on `arm-follower` and `arm-leader` control boxes

keys, tokens, passwords are stored in the `.env` file.

```bash
source config/.env
```

code is separated into projects, each intended to be run seperately, organized as folders in `/src`

python dependencies are managed with environments using [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

```bash
cd src/<foo-project>
deactivate && \
rm -rf .venv && \
rm uv.lock && \
uv venv && \
source .venv/bin/activate && \
uv pip install -r pyproject.toml && \
uv run python demo.py
```