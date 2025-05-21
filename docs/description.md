# Software

tatbot consists of many compute nodes connected via ethernet:

- `oop`: Ubuntu PC used for development
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

to be added:
- android phone with anker app, tattoo app?
- jetson board for rerun and fast3r

## Backends

the `BACKEND` environment variable is used when running docker containers with specific workloads:

- `oop` uses `x86-3090` backend
- `ojo` uses `arm-agx` backend
- `trossen-ai` uses `x86-meerkat` backend
- `rpi1` uses `arm-rpi` backend
- `rpi2` uses `arm-rpi` backend

## Networking

- `switch-main`
    (1) short black ethernet cable to `switch-poe`
    (2) short black ethernet cable to `trossen-ai`
    (3) short black ethernet cable to `ojo`
    (4) blue ethernet cable to `arm-follower` controller box
    (5) blue ethernet cable to `arm-leader` controller box
- `switch-poe`
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

during development, both `switch-main` and `switch-poe` are connected to a LAN with a router and DNS.
the home network also includes `oop` and most development is done via ssh from `oop`.

## Devices

the following devices are connected to each compute node:

- `oop`
    - `cam_wrist` via usb3 port
- `trossen-ai`
    - `cam_main` via usb3 port
    - wireless keyboard via usb2 port
    - touchscreen display via usbc port

## Startup and Shutdown

startup sequence:
1. flip power strip in back to on
2. press power button on `trossen-ai`, it will glow blue
3. flip rocker switches to "on" on `arm-follower` and `arm-leader` control boxes

shutdown sequence:
1. run the `scripts/oop/shutdown.py` script
2. flip rocker switches to "off" on `arm-follower` and `arm-leader` control boxes
3. flip power strip in back to off

## Environment Setup

keys, tokens, passwords are stored in the `.env` file. copy it to the compute nodes:

```bash
./scripts/oop/copy-to-compute.py --file-paths config/.env
source config/.env
```

## Development Setup

configure and set the backend (e.g. `x86-3090`, `arm-rpi`)

```bash
source scripts/backends/x86-3090.sh
```

## Workflow

the following assets are required:

- at least one tattoo design in `assets/designs`
- a 3d mesh model of the tattoo area in `assets/3d`
	- a urdf of the robot arm in `assets/trossen_arm_description`

run the stencil simulation to generate ik poses

```bash
./scripts/stencil.sh
```

visualize the final stencil placement (requires `usdview`)

```bash
usdview $TATBOT_ROOT/output/stencil.usd
```

run the ik solver with a specific morph (e.g. `gpt-e409cb`)

```bash
./scripts/ik/morph_render.sh gpt-e409cb
```

visualize the ik result (requires `usdview`)

```bash
usdview $TATBOT_ROOT/output/ik_gpt-e409cb.usd
```

put the arms back in the sleep position:

```bash
python ~/dev/tatbot-dev/scripts/trossen-ai/arms-sleep.py
```

configure the arms (config files are in `~/dev/tatbot-dev/config/trossen`):

```bash
python ~/dev/tatbot-dev/scripts/trossen-ai/arms-config.py
python ~/dev/tatbot-dev/scripts/trossen-ai/arms-config.py --push
```

reset the realsense cameras:

```bash
sudo bash ~/dev/tatbot-dev/scripts/trossen-ai/reset-realsenses.sh
```

## Testing

the following tests should work on all backends

```bash
./scripts/test/ik.sh
./scripts/test/ai.sh # this will test model apis (do not run this every time as it consumes credits)
```