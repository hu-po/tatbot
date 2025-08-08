# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tatbot is a tattoo robot system that uses dual Trossen WidowXAI arms, Intel RealSense D405 depth cameras, Amcrest IP cameras, and VLA models for autonomous tattooing. The system is distributed across multiple nodes with MCP (Model Context Protocol) servers for remote control and coordination.

## Common Development Commands

### Environment Setup
```bash
# Initial setup with uv
source scripts/setup_env.sh
uv pip install -e .

# Install node-specific dependencies
uv pip install .[bot,cam,dev,gen,gpu,img,viz]

# Load environment variables (API keys, passwords)
set -a; source .env; set +a
```

### Code Quality
```bash
# Run linting and formatting
./scripts/lint.sh
```

### MCP Server Operations
```bash
# Start MCP server for a specific node
./scripts/run_mcp.sh <node_name>  # ook, oop, ojo, trossen-ai, rpi1, rpi2

# Kill existing MCP processes
./scripts/kill.sh

# Restart mcp server on trossen-ai
ssh trossen-ai "bash scripts/run_mcp.sh trossen-ai"

# Monitor MCP logs
tail -f ~/tatbot/nfs/mcp-logs/<node_name>.log
```

### Running Operations
```bash
# Run main application with Hydra configuration
uv run python -m tatbot.main

# Run with specific scene
uv run python -m tatbot.main scenes=tatbotlogo

# Run visualization tools
uv run python -m tatbot.viz.stroke --scene=tatbotlogo
uv run python -m tatbot.viz.teleop --scene=default
```

## Architecture

### Core Components

1. **MCP Servers** (`src/tatbot/mcp/`)
   - Distributed control system using Model Context Protocol
   - Each node runs an MCP server with specific capabilities
   - Tools are dynamically registered based on node configuration
   - Uses Hydra for configuration management

2. **Configuration System** (`src/conf/`)
   - Hydra-based configuration with defaults and overrides
   - Scene configurations combine arms, cameras, inks, poses
   - All configs validate through Pydantic schemas

3. **Robot Control** (`src/tatbot/bot/`)
   - Trossen arm control via `trossen_arm` library
   - URDF-based kinematics
   - Dual-arm coordination (left/right)

4. **Vision System** (`src/tatbot/cam/`)
   - Intel RealSense depth cameras
   - AprilTag tracking for localization
   - Camera calibration and extrinsics

5. **Generation Pipeline** (`src/tatbot/gen/`)
   - G-code parsing and stroke generation (`gcode.py`)
   - Inverse kinematics solving using JAX (`ik.py`)
   - 2D to 3D surface mapping using geodesic paths (`map.py`)
   - Stroke batch processing for GPU acceleration (`batch.py`)
   - Ink dipping trajectory generation (`inkdip.py`)

6. **Operations** (`src/tatbot/ops/`)
   - Recording and playback workflows
   - Stroke execution
   - System reset procedures

### Node Topology

- **ook** (🦧): Acer Nitro V 15 with NVIDIA RTX 4050, performs GPU-accelerated batch ik
- **oop** (🦊): Ubuntu PC with NVIDIA RTX 3090 (only available in "home" mode)
- **ojo** (🦎): NVIDIA Jetson AGX Orin, runs agent models via Ollama
- **trossen-ai** (🦾): System76 Meerkat PC, robot arm control and RealSense cameras
- **rpi1/rpi2** (🍓🍇): Raspberry Pi 5 nodes, rpi2 serves as NFS server
- **camera1-5** (📷): Amcrest IP PoE cameras for scene coverage
- **realsense1-2** (📷): Intel RealSense D405 depth cameras mounted on arms

### Data Flow

1. **Design Generation**: Image → DrawingBotV3 → G-code files
2. **Stroke Processing**: G-code → `make_gcode_strokes` → `StrokeList`
3. **Surface Mapping**: __optional__ `StrokeList` → `map_strokes_to_mesh` → 3D mapped strokes → `StrokeList`
4. **IK Solving**: `StrokeList` → `batch_ik` → `StrokeBatch` with joint angles
5. **Execution**: `StrokeBatch` → `stroke_tool` → Robot arms via LeRobot interface

## Key Design Patterns

- **Hydra Configuration**: All components configured via YAML with schema validation
- **MCP Distribution**: Tools exposed as MCP endpoints for remote execution
- **Pydantic Validation**: Strong typing and validation throughout
- **Virtual Environment**: Uses `uv` for fast, deterministic dependency management
- **Modular Extras**: Optional dependencies grouped by functionality
- **Configuration Constants**: Centralized constant classes replace magic numbers (AppConstants, ServerConstants, MCPConstants, CalibrationConstants)
- **Specific Exception Handling**: Custom exception types (ConfigurationError, NetworkError, CalibrationError, etc.) replace generic Exception catching
- **Descriptive Naming**: Variables use meaningful names instead of cryptic abbreviations

## Critical Files

- `src/tatbot/main.py`: Entry point with Hydra initialization
- `src/tatbot/mcp/server.py`: MCP server implementation
- `src/tatbot/gen/batch.py`: Stroke batch processing
- `src/conf/config.yaml`: Root Hydra configuration
- `pyproject.toml`: Project dependencies and metadata

## Important Notes

- Always use `uv` for Python package management
- MCP server changes require manual restart in Cursor UI (Ctrl+Shift+P > "View: OpenMCP Settings")
- Distributed system - ensure network connectivity between nodes
- Camera passwords and API keys stored in `.env` file
- NFS mount required: `~/tatbot/nfs` shared across all nodes (served by rpi2)
- Designs stored in `tatbot/nfs/designs/` directory
- DrawingBotV3 configs in `config/dbv3/` for pen settings and G-code generation
- Both RealSense cameras connected to trossen-ai via USB3
- `tatbot/nfs/recordings/` directory for recordings of robot tools