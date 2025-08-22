---
summary: Setup, install, and developer workflows
tags: [setup, dev]
updated: 2025-08-21
audience: [dev]
---

# 💻 Development

## Quick Setup

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Python dependency and virtual environment management.

```{admonition} Quick Reference
:class: tip
- Setup: `source scripts/setup_env.sh`
- Lint code: `bash scripts/lint_code.sh`
- Lint docs: `bash scripts/lint_docs.sh`
```

```{admonition} Prerequisites
:class: important

**Required:**
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Git
```

### Quick Install

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

### Optional Dependencies

Dependencies are separated into optional groups, defined in `pyproject.toml`. Install the groups you need for your task.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card}
:class-header: bg-light

⚙️ **Core Functionality Groups**
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

📝 **Development and Docs**
^^^
- `dev`: Development tools (`ruff`, `isort`, `pytest`, `mypy`, `pre-commit`)
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

### Full Setup
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

### Startup
1. **Power On**: Flip the main power strip on.
2. **`hog` and `eek` PCs**: Press the power button.
3. **Robot Arms**: Flip the rocker switches on the `arm-r` and `arm-l` control boxes to "ON".
4. **Lighting**: Turn on the light bar via its rocker switch.
5. **Pens**: Turn on the tattoo pen batteries.
6. **MCP Servers**: SSH into each required node (`ook`, `oop`, `eek`, etc.) and run the appropriate MCP server command.
   ```bash
   # On ook
   cd ~/tatbot && ./scripts/mcp_run.sh ook
   ```

## Workflow

### Code Quality
This project uses `ruff` for both linting and formatting, plus `isort` for import sorting.

To run all code quality checks, use the lint script:
```bash
./scripts/lint_code.sh
```

### Docs
Generate and validate documentation minimally during linting. The main lint script now builds docs with warnings-as-errors:
```bash
./scripts/lint_code.sh
```
This runs `sphinx-build -W docs docs/_build` after code checks.

### Commands

oneliner to get diff for browser-based models:
```bash
rm -rf diff.txt && git diff main...HEAD > /tmp/diff.txt && xclip -selection clipboard < /tmp/diff.txt
```

when merge conflicts arise in forked repos (e.g., `lerobot`), follow this process:
```bash
cd ~/lerobot # or other forked repo
git pull
git fetch upstream
git merge upstream/main
git push origin main
```

### General Tips
- Always work within the `uv` virtual environment (`source .venv/bin/activate`)
- Use `uv pip install` and `uv run` for consistency

 

## Architecture

This section provides a comprehensive overview of the `tatbot` source code structure, organized by module. Each module serves a specific purpose in the robotic tattoo system, from hardware control to data generation and visualization.

### Main Scripts (`src/tatbot/`)

This module contains the main entry points and configuration schema files. These files are central to the application's configuration loading and validation process, bridging the gap between raw YAML files and the strongly-typed Pydantic data models used throughout the system.

**Core Abstractions:**
- **Hydra-Powered Configuration**: Uses [Hydra](https://hydra.cc/) for managing complex configuration
- **Pydantic Schema Validation**: Provides an additional layer of validation and type safety

**Key Files:**
- `main.py`: Primary entry point for loading and validating configuration
- `config_schema.py`: Defines the top-level Pydantic model for the entire application configuration

### Bot Module (`src/tatbot/bot`)

Responsible for controlling the robot's hardware, specifically the Trossen arms. Handles configuration, homing, and URDF-based kinematic calculations.

**Core Abstractions:**
- `TrossenConfig`: Data class holding configuration parameters for the Trossen arms
- `yourdfpy.URDF` and `pyroki.Robot`: Load and represent the robot's model from URDF

**Key Files:**
- `trossen_config.py`: Manages Trossen arm configuration using YAML files
- `trossen_homing.py`: Interactive process for calibrating and homing a Trossen arm
- `urdf.py`: Handles URDF loading and forward kinematics

### Camera Module (`src/tatbot/cam`)

Handles all camera-related operations, including intrinsic and extrinsic calibration, AprilTag tracking, and capturing point cloud data.

**Core Abstractions:**
- `Intrinsics`: Internal camera parameters (focal length, principal point)
- `Pose`: 3D position and orientation of objects
- `TagTracker`: Detects AprilTags and calculates their 3D poses
- `DepthCamera`: Interfaces with RealSense depth cameras

**Key Files:**
- `intrinsics_rs.py`: Manages RealSense camera intrinsics
- `extrinsics.py`: Calculates camera extrinsic parameters
- `tracker.py`: Detects and tracks AprilTags
- `depth.py`: Interfaces with RealSense cameras for depth data

### Data Module (`src/tatbot/data`)

Defines the core data structures used throughout the application using Pydantic for typed data models.

**Core Abstractions:**
- `BaseCfg`: Foundation of all data models with YAML serialization
- `Scene`: Master data structure encapsulating entire robotic task state
- `Stroke`: Represents a single continuous robot motion
- `StrokeList`: Manages bimanual stroke pairs
- `StrokeBatch`: JAX-optimized structure for batch processing

**Key Data Models:**
- `Pose`: 3D position and orientation
- `Arms`: Bimanual robot arm configuration
- `Cams`: Camera configuration
- `URDF`: Robot URDF file and link names
- `Skin`: Surface properties for tattooing
- `Inks`: Ink and ink cap properties
- `Tags`: AprilTag detection configuration
- `Node`: Computing node in distributed system

### Generation Module (`src/tatbot/gen`)

Generates data required for robot tasks, converting high-level representations into concrete stroke and batch objects.

**Pipeline Philosophy:**
`gcode.py`/`align.py` → `map.py` → `batch.py` → `ik.py`

**Key Files:**
- `gcode.py`: Parses G-code files into StrokeList
- `align.py`: Generates predefined alignment sequences
- `inkdip.py`: Creates ink-dipping trajectories
- `map.py`: Projects 2D strokes onto 3D mesh surfaces
- `ik.py`: Performs inverse kinematics calculations
- `batch.py`: Converts StrokeList to StrokeBatch

### MCP Module (`src/tatbot/mcp`)

Implements the Multi-agent Command Platform (MCP) server as the primary API endpoint.

**Core Abstractions:**
- `FastMCP`: Underlying server implementation
- Tools as Functions: Server functionality exposed as decorated functions
- Hydra for Configuration: Flexible deployments via configuration
- Pydantic Models: Strongly typed and validated communication

**Key Files:**
- `server.py`: Main MCP server entry point
- `models.py`: Pydantic models for requests/responses

### Tools Module (`src/tatbot/tools`)

Provides a unified, decorator-based approach to defining operations and utilities that can be executed via MCP across multiple nodes.

**Core Abstractions:**
- **Unified Module**: All tools are in `src/tatbot/tools/` with a clean architecture
- **Decorator-Based**: Clean `@tool` decorator system eliminates complex inheritance hierarchies
- **Type-Safe**: Full Pydantic validation for inputs and outputs
- **Node-Aware**: Tools specify which nodes they're available on and requirements
- **Async Generators**: Maintains progress reporting via async generators
- **Auto-Discovery**: Tools register themselves automatically on import

**Key Tool Categories:**
- **System Tools**: `list_nodes`, `ping_nodes`, `list_scenes`, `list_recordings`
- **Robot Tools**: `align`, `reset`, `sense`, `stroke`
- **GPU Tools**: `convert_strokelist_to_batch` (GPU-accelerated IK)
- **Visualization Tools**: `start_stroke_viz`, `start_teleop_viz`, `start_map_viz`

**Key Files:**
- `base.py`: Base types and ToolContext
- `registry.py`: Tool registration and management
- `robot/`: Robot operation tools
- `system/`: System/utility tools
- `gpu/`: GPU-specific tools
- `viz/`: Visualization tools

### Utilities Module (`src/tatbot/utils`)

Helper functions and classes used across the application.

**Key Files:**
- `log.py`: Standardized logging setup
- `net.py`: Network management and SSH connections
- `mode_toggle.py`: Network configuration switching
- `plymesh.py`: 3D point cloud and mesh utilities
- `colors.py`: Color definitions and conversions
- `validation.py` & `jnp_types.py`: Data validation and type conversion

### Visualization Module (`src/tatbot/viz`)

Interactive 3D visualization tools built on the `viser` library.

**Core Abstractions:**
- `viser.ViserServer`: WebSocket server for real-time 3D scenes
- `BaseViz`: Abstract base class for visualization applications

**Key Visualizations:**
- `stroke.py`: Visualize StrokeBatch execution
- `teleop.py`: Interactive robot teleoperation with IK
- `map.py`: Visualize 2D to 3D stroke mapping

### How to Use the Modules

1. **Configuration**: Start by understanding the Hydra configuration system in `main.py`
2. **Data Flow**: Follow the pipeline from G-code → StrokeList → StrokeBatch → Execution
3. **Tools**: Use the MCP server to run high-level operations via the tools system
4. **Visualization**: Use the viz tools for debugging and calibration
5. **Development**: Extend existing modules following established patterns
