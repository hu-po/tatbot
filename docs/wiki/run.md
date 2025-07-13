# Run

Setup environment, optionally clean output directory

```bash
source scripts/env.sh
./scripts/clean.sh
```

tatbot is designed as a multi-node system, with the following roles:

`oop` 🦊 and `ook` 🦧 are the main nodes, they:
- run the mcp server to interact with tatbot
- run the visualization server to view plans

```bash
# optionally install dev dependencies
uv pip install .[dev,viz,mcp] && \
uv run -m tatbot.viz.strokes --debug --scene-name "calib"
```

`trossen-ai` 🦾 sends commands to robot arms, receives realsense camera images, and records lerobot datasets:

```bash
uv pip install .[bot] && \
# configure trossen arms
uv run -m tatbot.bot.trossen --debug
# run lerobot dataset recording from plan
uv run -m tatbot.bot.record --debug --scene-name "calib"
```

`ojo` 🦎 runs the policy servers for the VLA model and for the 3d reconstruction model

```bash
uv pip install .[vla] && \
# TODO
```

`rpi1` 🍓 runs apriltag tracking and camera calibration:

```bash
uv pip install .[tag] && \
uv run tatbot.tag.scan --bot_scan_dir ~/tatbot/outputs/ --debug
```

`rpi2` 🍇 hosts the network drive, which contains the output directory for all nodes

```bash
uv pip install .[viz] && \
uv run -m tatbot.viz.plan --plan_dir ~/tatbot/outputs/plans/yawning_cat
``` 