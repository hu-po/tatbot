# ⚡ Setup

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Python dependency and virtual environment management.

```{admonition} Prerequisites
:class: important

**Required:**
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Git
```

## 🚀 Quick Install

::::{tab-set}

:::{tab-item} Basic Installation
This command clones the repository and sets up the basic environment.

```bash
git clone --depth=1 https://github.com/hu-po/tatbot.git && cd tatbot
uv venv
source .venv/bin/activate
uv pip install -e .
```
:::

:::{tab-item} One-liner Setup
For experienced users who want everything at once:

```bash
git clone --depth=1 https://github.com/hu-po/tatbot.git && cd tatbot && source scripts/setup_env.sh
```
:::

::::

## 📦 Optional Dependencies

Dependencies are separated into optional groups, defined in `pyproject.toml`. Install the groups you need for your task.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card}
:class-header: bg-light

🤖 **Core Functionality Groups**
^^^
- `bot`: Robot-specific dependencies (`lerobot`, `trossen-arm`, etc.)
- `cam`: Camera-specific dependencies (`pyrealsense2`, `pupil-apriltags`, etc.)  
- `gen`: Stroke generation and inverse kinematics
- `gpu`: For GPU-accelerated tasks (`jax[cuda12]`)
- `img`: Image processing libraries (`opencv`)
- `viz`: Visualization tools (`viser`)
:::

:::{grid-item-card}
:class-header: bg-light

🛠️ **Development and Quality Groups**
^^^
- `dev`: Basic development tools (`ruff`, `isort`, `pytest`)
- `quality`: Advanced development tools (`mypy`, `pre-commit`, type stubs)
- `docs`: Documentation generation (`sphinx`, themes)
:::

::::

**Installation Examples:**

::::{tab-set}

:::{tab-item} Robot Control
```bash
# For robot operations
uv pip install .[bot,viz,cam]
```
:::

:::{tab-item} Development
```bash
# Development environment
uv pip install .[dev,docs]
```
:::

:::{tab-item} Full Install
```bash
# Everything (recommended for main development)
uv pip install .[bot,cam,dev,gen,gpu,img,viz,docs]
```
:::

::::

## Full Environment Setup
For a clean, from-scratch setup:
```bash
git clone --depth=1 https://github.com/hu-po/tatbot.git && cd ~/tatbot
source scripts/setup_env.sh

# Install all dependencies (choose based on your needs)
uv pip install .[bot,cam,dev,gen,gpu,img,viz,docs]

# Source environment variables (e.g., API keys, camera passwords)
# Ensure you have a .env file (see .env.example)
set -a; source /nfs/tatbot/.env; set +a
```

## Starting the System
1. **Power On**: Flip the main power strip on.
2. **`hog` and `eek` PCs**: Press the power button.
3. **Robot Arms**: Flip the rocker switches on the `arm-r` and `arm-l` control boxes to "ON".
4. **Lighting**: Turn on the light bar via its rocker switch.
5. **Pens**: Turn on the tattoo pen batteries.
6. **MCP Servers**: SSH into each required node (`ook`, `oop`, `eek`, etc.) and run the appropriate MCP server command.
   ```bash
   # On ook
   cd ~/tatbot && ./scripts/run_mcp.sh ook
   ```

