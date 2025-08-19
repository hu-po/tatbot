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
uv pip install .[bot,cam,gen,gpu,img,viz]

# Development dependencies
uv pip install .[dev,docs]

# Load environment variables (API keys, passwords)
set -a; source .env; set +a
```

### Documentation (build, view, live-reload)
```bash
# Build docs
uv pip install -e .[docs]
uv run sphinx-build docs docs/_build

# View docs
xdg-open docs/_build/index.html

# Live-reload docs during edits (serves at http://127.0.0.1:8000)
uv pip install -e .[dev]  # ensures sphinx-autobuild
uv run sphinx-autobuild docs docs/_build
```

### Code Quality, Type Checking, Linting
```bash
# Run linting and formatting
./scripts/lint.sh
```

### System Monitoring (TUI)
```bash
# Start TUI monitor on rpi1 (real-time system dashboard)
ssh rpi1
source scripts/setup_env.sh
uv run tatbot-monitor
```

### MCP Server Operations
```bash
# Start MCP server for a specific node
./scripts/mcp_run.sh <node_name>  # ook, oop, ojo, eek, rpi1, rpi2

# Kill existing MCP processes
./scripts/kill.sh

# Restart MCP servers on specific nodes (must SSH to each node)
ssh eek "bash ~/tatbot/scripts/mcp_run.sh eek"
ssh ook "bash ~/tatbot/scripts/mcp_run.sh ook"

# Monitor MCP logs
tail -f /nfs/tatbot/mcp-logs/<node_name>.log
```

### Running Operations
```bash
# run mcp server on the eek node
cd ~/tatbot && ./scripts/mcp_run.sh eek


# Run visualization tools
 uv run python -m tatbot.viz.stroke --scene=tatbotlogo
uv run python -m tatbot.viz.teleop --enable-robot --enable-depth
```

## Meta Configuration System

The system supports **meta configs** for easy parameter overrides across scenes. All MCP tools now accept a `meta` parameter to apply predefined configuration templates.

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

6. **Tools System** (`src/tatbot/tools/`)
   - Unified decorator-based tool registration
   - Robot operations: align, reset, sense, stroke
   - System utilities: list_nodes, ping_nodes, list_scenes, list_recordings  
   - GPU acceleration: convert_strokelist_to_batch
   - Cross-node operation execution via MCP


### Data Flow

1. **Design Generation**: Image → DrawingBotV3 → G-code files (stored in `/nfs/tatbot/designs/`)
2. **Stroke Processing**: G-code → `make_gcode_strokes` → `StrokeList` 
3. **Surface Mapping**: (optional) `StrokeList` → `map_strokes_to_mesh` → 3D mapped strokes → `StrokeList`
4. **IK Solving**: `StrokeList` → `strokebatch_from_strokes` → `StrokeBatch` with joint angles
   - Uses GPU acceleration via `convert_strokelist_to_batch` MCP tool 
   - Automatic cross-node GPU routing when local GPU unavailable
5. **Execution**: `StrokeBatch` → MCP `stroke` tool → Robot arms via Trossen control

## Key Design Patterns

- **Hydra Configuration**: All components configured via YAML with schema validation
- **MCP Distribution**: Tools exposed as MCP endpoints for remote execution
- **Pydantic Validation**: Strong typing and validation throughout
- **Virtual Environment**: Uses `uv` for fast, deterministic dependency management
- **Modular Extras**: Optional dependencies grouped by functionality (bot, cam, dev, gen, gpu, img, viz, docs)
- **Configuration Constants**: Centralized constant classes replace magic numbers (AppConstants, ServerConstants, MCPConstants, CalibrationConstants)
- **Specific Exception Handling**: Custom exception types (ConfigurationError, NetworkError, CalibrationError, etc.) replace generic Exception catching
- **Descriptive Naming**: Variables use meaningful names instead of cryptic abbreviations

## Critical Files

- `src/tatbot/main.py`: Entry point with Hydra initialization
- `src/tatbot/mcp/server.py`: MCP server implementation with dynamic tool registration
- `src/tatbot/tools/`: Unified tools system with decorator-based registration
  - `src/tatbot/tools/registry.py`: Tool registration and discovery
  - `src/tatbot/tools/robot/`: Robot control tools (align, reset, sense, stroke)
  - `src/tatbot/tools/gpu/`: GPU acceleration tools (convert_strokelist_to_batch) 
  - `src/tatbot/tools/system/`: System utilities (list_nodes, ping_nodes)
- `src/tatbot/gen/batch.py`: Stroke batch processing with JAX/GPU acceleration
- `src/conf/config.yaml`: Root Hydra configuration
- `src/conf/mcp/`: Node-specific MCP server configurations
- `pyproject.toml`: Project dependencies and metadata with optional extras

## Available MCP Tools

The following tools are available via MCP servers on different nodes:

### Robot Control Tools (eek, hog)
- **`align_tool`**: Generate and execute alignment strokes for calibration (supports `meta` parameter)
- **`reset_tool`**: Reset robot to safe/ready position (supports `meta` parameter)
- **`sense_tool`**: Capture environmental data from cameras and sensors (supports `meta` parameter)  
- **`stroke_tool`**: Execute artistic strokes with ink on canvas (supports `meta` parameter)

### GPU Processing Tools (ook, oop when available)
- **`convert_strokelist_to_batch`**: GPU-accelerated stroke trajectory conversion using JAX
- **`reset_tool`**: Emergency robot reset capability

### System Management Tools (oop, rpi1)  
- **`list_nodes`**: List all available tatbot nodes
- **`ping_nodes`**: Test connectivity to tatbot nodes

### Visualization Tools (oop, ook)
- **`start_stroke_viz`**: Start interactive stroke execution visualization server (supports `meta` parameter)
- **`start_teleop_viz`**: Start teleoperation server for interactive robot control (supports `meta` parameter)
- **`start_map_viz`**: Start surface mapping visualization for 2D-to-3D debugging (supports `meta` parameter)
- **`stop_viz_server`**: Stop a running visualization server
- **`list_viz_servers`**: List all running visualization servers

### Cross-Node Operations
- Robot operations automatically detect GPU availability and route conversion tasks
- Uses NFS shared storage for seamless file access across nodes
- JSON-RPC 2.0 over HTTP for reliable cross-node communication

## Important Notes

- Always use `uv` for Python package management
- MCP server changes require restarting servers on each node via SSH (see commands above)
- Cursor IDE MCP integration may require restart: Ctrl+Shift+P > "View: OpenMCP Settings"  
- Distributed system - ensure network connectivity between nodes
- Camera passwords and API keys stored in `.env` file
- NFS mount required: `/nfs/tatbot` shared across all nodes (served by eek)
- Designs stored in `/nfs/tatbot/designs/` directory with DrawingBotV3 project files
- DrawingBotV3 configs in `config/dbv3/` for pen settings and G-code generation
- Both RealSense cameras connected to eek via USB3
- `/nfs/tatbot/recordings/` directory contains timestamped robot operation recordings