# Setup

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Python dependency and virtual environment management.

## Quick Install
This command clones the repository and sets up the basic environment.
```bash
git clone --depth=1 https://github.com/hu-po/tatbot.git && cd tatbot
python3 -m venv .venv
source .venv/bin/activate
uv pip install .
```

## Optional Dependencies
Dependencies are separated into optional groups, defined in `pyproject.toml`. Install the groups you need for your task.
- `bot`: Robot-specific dependencies (`lerobot`, `trossen-arm`, etc.)
- `cam`: Camera-specific dependencies (`pyrealsense2`, `pupil-apriltags`, etc.)
- `dev`: Development tools (`ruff`, `pytest`)
- `gen`: Stroke generation and inverse kinematics
- `gpu`: For GPU-accelerated tasks
- `img`: Image processing libraries (`opencv`)
- `viz`: Visualization tools

Install one or more groups like this:
```bash
uv pip install .[bot,viz,cam]
```

## Full Environment Setup
For a clean, from-scratch setup:
```bash
git clone --depth=1 https://github.com/hu-po/tatbot.git && cd ~/tatbot
git pull

# Create and activate a new virtual environment
python3 -m venv .venv --prompt="tatbot"
source .venv/bin/activate

# Install base and all optional dependencies
uv pip install .[bot,cam,dev,gen,gpu,img,viz]

# Source environment variables (e.g., API keys, camera passwords)
# Ensure you have a .env file (see .env.example)
set -a; source .env; set +a
```

## Starting the System
1. **Power On**: Flip the main power strip on.
2. **`trossen-ai` PC**: Press the power button; it will glow blue.
3. **Robot Arms**: Flip the rocker switches on the `arm-r` and `arm-l` control boxes to "ON".
4. **Lighting**: Turn on the light bar via its rocker switch.
5. **Pens**: Turn on the tattoo pen batteries.
6. **MCP Servers**: SSH into each required node (`ook`, `oop`, `trossen-ai`, etc.) and run the appropriate MCP server command.
   ```bash
   # On ook
   cd ~/tatbot && ./scripts/run_mcp.sh ook

   # On trossen-ai
   cd ~/tatbot && ./scripts/run_mcp.sh trossen-ai
   ```
