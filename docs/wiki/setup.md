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
5. turn on the tattoo pen batteries