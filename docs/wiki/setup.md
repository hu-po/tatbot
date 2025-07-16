# Setup

python dependencies managed using [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

streamlined environment setup:

```bash
git clone --depth=1 https://github.com/hu-po/tatbot.git && cd tatbot
source scripts/env.sh
uv pip install .[foo,bar] # see optional dependencies below
```

dependencies are seperated into groups, see `pyproject.toml`

- `bot` - robot dependencies (lerobot, trossen, realsense, etc)
- `dev` - development dependencies (ruff, isort)
- `gen` - generation of strokes and batch ik
- `map` - skin reconstruction and design mapping
- `viz` - visualization

camera passwords and model api keys are stored in `.env`, see `.env.example`

```bash
# Basic install
git clone --depth=1 https://github.com/hu-po/tatbot.git && \
cd ~/tatbot && \
git pull && \
# Optional: Clean old uv environment
deactivate && \
rm -rf .venv && \
rm uv.lock && \
# Setup new uv environment
uv venv --prompt="tatbot" && \
source .venv/bin/activate && \
uv pip install .
# source env variables (i.e. keys, tokens, camera passwords)
source .env
```

to turn on the robot:

1. flip power strip in the back to on.
2. press power button on `trossen-ai`, it will glow blue.
3. flip rocker switches to "on" on `arm-r` and `arm-l` control boxes underneath workspace.
4. flip rocker switch on the back of the light to turn it on.

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