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
./scripts/oop/copy-to-compute.py --file-paths cfg/.env
source cfg/.env
```

## Workload

during operation the following work is performed by each node:

- `oop`
    - docker container `tatbot-oop`
        - finetuned policy
- `trossen-ai`
    - docker container `tatbot-trossen-ai`
        - ros master node
        - ros moveit node
        - ros robot description node
        - ros pointcloud2 node
        - ros parameter server
        - ros tf2 node
- `rpi1`
    - docker container `tatbot-rpi1`
        - ros apriltag detection
- `rpi2`
    - docker container `tatbot-rpi2`
        - DNS server

to be added:
    - frontend (rviz? rerun?)
    - some kind of video recording or live streaming server
    - some kind of mcp server
    - inventory management via sqlite db

## Notes

notes contain loose text information related to different aspects of the system:

- [ojo](notes/ojo.md)
- [rpis](notes/rpis.md)
- [switches](notes/switches.md)
- [cameras](notes/cameras.md)
- [apriltags](notes/apriltags.md)
- [ros](notes/ros.md)
- [trossen](notes/trossen.md)
- [realsense](notes/realsense.md)
- [power](notes/power.md)

## Specs

system specs for different compute nodes can be found at [docs/specs](docs/specs).

to create a new spec file (this will create a `docs/specs/rpi1.md` file when run on `rpi1`):

```bash
./scripts/specs.sh
```

specs for each node:

- [ojo](specs/ojo.md)
- [rpi1](specs/rpi1.md)
- [rpi2](specs/rpi2.md)
- [trossen-ai](specs/trossen-ai.md)
- [oop](specs/oop.md) (only used for development, not present in production)
