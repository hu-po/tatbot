# Networking

tatbot uses shared ssh keys for nodes to talk, send files, and run remote commands: see `_net.py` and `config/nodes.yaml`.

- `switch-lan`:
    - (1) short black ethernet cable to `switch-poe`
    - (2) short black ethernet cable to `trossen-ai`
    - (3) short black ethernet cable to `ojo`
    - (4) long black ethernet cable to `ook`
    - (5) *dev mode* long ethernet cable to `oop`
    - (6) blue ethernet cable to `arm-r` controller box
    - (7) blue ethernet cable to `arm-l` controller box
    - (8)
- `switch-poe`:
    - (1) long black ethernet cable to `camera1`
    - (2) long black ethernet cable to `camera2`
    - (3) long black ethernet cable to `camera3`
    - (4) long black ethernet cable to `camera4`
    - (5) long black ethernet cable to `camera5`
    - (6) -
    - (7) short black ethernet cable to `rpi1`
    - (8) short black ethernet cable to `rpi2`
    - (uplink-1) -
    - (uplink-2) short black ethernet cable to `switch-lan`

to setup the network:

```bash
uv run tatbot.net.net
```

NFS setup: [wiki](nfs.md) 