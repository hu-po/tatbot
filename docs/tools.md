---
summary: Tools architecture and how to define and use tools
tags: [tools, mcp]
updated: 2025-08-21
audience: [dev]
---

# Tatbot Tools Architecture

The Tatbot tools system provides a unified, decorator-based approach to defining operations and utilities that can be executed via MCP (Model Context Protocol) across multiple nodes.

## 🔍 Overview

- **Unified Module**: All tools are now in `src/tatbot/tools/` with a clean architecture
- **Decorator-Based**: Clean `@tool` decorator system eliminates complex inheritance hierarchies
- **Type-Safe**: Full Pydantic validation for inputs and outputs
- **Node-Aware**: Tools specify which nodes they're available on and requirements
- **Async Generators**: Maintains beloved progress reporting via async generators
- **Auto-Discovery**: Tools register themselves automatically on import

## 🏠 Tool Structure

```text
src/tatbot/tools/
├── __init__.py              # Registry, decorators, auto-discovery
├── base.py                  # Base types and ToolContext
├── registry.py              # Tool registration and management
├── robot/                   # Robot operation tools
│   ├── align.py            # Alignment operations
│   ├── stroke.py           # Stroke execution
│   ├── sense.py            # Environment sensing
│   └── reset.py            # Robot reset operations
├── system/                  # System/utility tools
│   ├── ping_nodes.py       # Network connectivity testing
│   ├── list_scenes.py      # Scene configuration discovery
│   ├── list_nodes.py       # Node discovery
│   └── models.py           # Pydantic models for system tools
└── gpu/                     # GPU-specific tools
    ├── convert_strokes.py   # GPU-accelerated stroke conversion
    └── models.py           # Pydantic models for GPU tools
```

## 🔨 Creating Tools

Tools are simple async functions decorated with `@tool`:

```python
from tatbot.tools.base import ToolInput, ToolOutput, ToolContext
from tatbot.tools.registry import tool

class MyToolInput(ToolInput):
    parameter: str
    debug: bool = False

class MyToolOutput(ToolOutput):
    result: str

@tool(
    name="my_tool",
    nodes=["eek", "ook"],  # Available on these nodes
    description="Example tool that does something useful",
    input_model=MyToolInput,
    output_model=MyToolOutput,
    requires=["gpu"]  # Optional requirements
)
async def my_tool(input_data: MyToolInput, ctx: ToolContext):
    """Tool implementation with async generator pattern."""
    
    # Report progress
    yield {"progress": 0.1, "message": "Starting processing..."}
    
    # Do work
    result = f"Processed: {input_data.parameter}"
    
    yield {"progress": 0.8, "message": "Almost done..."}
    
    # Return final result
    return MyToolOutput(
        success=True,
        message="Tool completed successfully",
        result=result
    )
```

## 🏷️ Tool Categories

### System Tools

- **`list_nodes`** (rpi1, ook, oop): List all configured tatbot nodes
- **`ping_nodes`** (rpi1): Test connectivity to tatbot nodes  
- **`list_scenes`** (rpi2): Discover available scene configurations
- **`list_recordings`** (rpi2): List available recordings from the recordings directory

### GPU Tools (GPU Nodes Only)

- **`convert_strokelist_to_batch`** (ook, oop): GPU-accelerated inverse kinematics for stroke conversion

### Robot Tools

- **`align`** (hog, oop): Generate and execute alignment strokes for calibration
- **`reset`** (hog, ook, oop): Reset robot to safe/ready position
- **`sense`** (hog): Capture environmental data (cameras, sensors)
- **`stroke`** (hog): Execute artistic strokes on paper/canvas

### Visualization Tools

- **`start_stroke_viz`** (ook, oop): Start stroke visualization server
- **`start_teleop_viz`** (ook, oop): Start teleoperation visualization server
- **`start_map_viz`** (ook, oop): Start surface mapping visualization server
- **`stop_viz_server`** (ook, oop): Stop running visualization servers
- **`list_viz_servers`** (ook, oop): List running visualization servers
- **`status_viz_server`** (ook, oop): Get status of visualization servers

## 🌐 Node Availability

Tools specify node availability in their decorator:

```python
@tool(nodes=["*"])              # All nodes
@tool(nodes=["eek"])     # Specific node only
@tool(nodes=["ook", "oop"])     # Multiple specific nodes
```

## ✅ Requirements

Tools can specify requirements (like GPU support):

```python
@tool(requires=["gpu"])         # GPU required
@tool(requires=["camera"])      # Camera hardware required
@tool(requires=[])              # No special requirements (default)
```

Requirements are checked against the `extras` field in node MCP configuration files.

## 📈 Progress Reporting

Tools use async generators to report progress:

```python
async def my_tool(input_data, ctx):
    # Yield progress updates
    yield {"progress": 0.0, "message": "Starting..."}
    yield {"progress": 0.5, "message": "Halfway done..."}
    yield {"progress": 1.0, "message": "Complete!"}
    
    # Return final result
    return MyToolOutput(success=True, message="Done")
```

## ⚠️ Error Handling

The registry system automatically handles:

- Input validation (via Pydantic models)
- Node availability checking
- Requirements verification
- Exception catching and conversion to tool outputs
- Logging and progress reporting

## 🔧 ToolContext

The `ToolContext` provides a unified interface:

```python
async def my_tool(input_data, ctx: ToolContext):
    # Report progress
    await ctx.report_progress(0.5, "Working...")
    
    # Send info message
    await ctx.info("Important information")
    
    # Access node name
    print(f"Running on node: {ctx.node_name}")
```

## 🔗 MCP Integration

Tools integrate seamlessly with the existing MCP server:

1. Tools register themselves automatically on import
2. MCP server queries available tools per node via `get_tools_for_node()`
3. Tool execution is handled by the registry wrapper
4. Progress reports flow through MCP protocol to clients

## 🏢 Architecture

The tools system provides a clean, modern approach to robotic operations:

```python
# tools/robot/align.py
@tool(name="align", nodes=["hog", "oop"])
async def align_tool(input_data: AlignInput, ctx: ToolContext):
    # Clean, self-contained implementation
    yield {"progress": 0.1, "message": "Starting alignment..."}
    # Implementation here
    return AlignOutput(success=True)
```

## ⚙️ Configuration

Node configurations in `src/conf/mcp/` control tool availability:

```yaml
# src/conf/mcp/ook.yaml
host: "0.0.0.0"
port: 8000
extras: ["gpu"]  # Enables GPU tools
tools:
  - reset
  - list_nodes
  - convert_strokelist_to_batch
  - start_stroke_viz
  - start_teleop_viz
  - start_map_viz
  - stop_viz_server
  - list_viz_servers
  - status_viz_server
```

## 🎩 Meta Configs

All robot tools and the GPU conversion tool support optional meta configuration overlays via Hydra:

- Add meta files under `src/conf/meta/`.
- Select a meta at call time with `meta=<name>`.
- Only specified subfields are overridden; other config remains unchanged.

Example meta file `src/conf/meta/tatbotlogo.yaml`:

```yaml
# @package _global_
arms:
  offset_range: [-0.01, 0.01]
  offset_num: 32
scenes:
  stroke_length: 128
  design_name: "tatbotlogo"
  inks_config_name: "red_blue"
  pen_names_l: [true_blue]
  pen_names_r: [bright_red]
skins:
  description: "100mm x 70mm square on sponge fake skin"
  image_width_m: 0.1
  image_height_m: 0.07
```

Usage across tools (JSON payloads):

```json
{ "scene": "flower", "meta": "tatbotlogo" }
```

Works for:
- `stroke`
- `align`
- `sense`
- `reset`
- `convert_strokelist_to_batch` (GPU)

Hydra defaults already include the optional group. See `src/conf/config.yaml` for the `defaults:` list with `- optional meta: _skip_`.
