# Tatbot Tools Architecture

The Tatbot tools system provides a unified, decorator-based approach to defining operations and utilities that can be executed via MCP (Model Context Protocol) across multiple nodes.

## Overview

- **Unified Module**: All tools are now in `src/tatbot/tools/` replacing the previous split between `mcp/handlers` and `ops` modules
- **Decorator-Based**: Clean `@tool` decorator system eliminates complex inheritance hierarchies
- **Type-Safe**: Full Pydantic validation for inputs and outputs
- **Node-Aware**: Tools specify which nodes they're available on and requirements
- **Async Generators**: Maintains beloved progress reporting via async generators
- **Auto-Discovery**: Tools register themselves automatically on import

## Tool Structure

```
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

## Creating a Tool

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
    nodes=["trossen-ai", "ook"],  # Available on these nodes
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

## Tool Categories

### System Tools (Available on All Nodes)

- **`ping_nodes`**: Test connectivity to tatbot nodes
- **`list_scenes`**: Discover available scene configurations
- **`list_nodes`**: List all configured tatbot nodes

### GPU Tools (GPU Nodes Only)

- **`convert_strokelist_to_batch`**: GPU-accelerated inverse kinematics for stroke conversion

### Robot Tools (Robot Nodes)

- **`align`**: Generate and execute alignment strokes for calibration
- **`stroke`**: Execute artistic strokes on paper/canvas
- **`sense`**: Capture environmental data (cameras, sensors)  
- **`reset`**: Reset robot to safe/ready position

## Node Availability

Tools specify node availability in their decorator:

```python
@tool(nodes=["*"])              # All nodes
@tool(nodes=["trossen-ai"])     # Specific node only
@tool(nodes=["ook", "oop"])     # Multiple specific nodes
```

## Requirements

Tools can specify requirements (like GPU support):

```python
@tool(requires=["gpu"])         # GPU required
@tool(requires=["camera"])      # Camera hardware required
@tool(requires=[])              # No special requirements (default)
```

Requirements are checked against the `extras` field in node MCP configuration files.

## Progress Reporting

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

## Error Handling

The registry system automatically handles:

- Input validation (via Pydantic models)
- Node availability checking
- Requirements verification
- Exception catching and conversion to tool outputs
- Logging and progress reporting

## ToolContext

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

## MCP Integration

Tools integrate seamlessly with the existing MCP server:

1. Tools register themselves automatically on import
2. MCP server queries available tools per node via `get_tools_for_node()`
3. Tool execution is handled by the registry wrapper
4. Progress reports flow through MCP protocol to clients

## Migration from Old System

### Before (Complex Inheritance)
```python
# ops/record_align.py
class AlignOp(RecordOp):
    op_name = "align"
    
    async def _run(self):
        # Implementation buried in inheritance
        
# mcp/handlers.py  
@mcp_handler
async def run_op(input_data, ctx):
    # Generic handler with factory pattern
    op_class, op_config = get_op(op_name, node_name)
```

### After (Clean Decorator)
```python
# tools/robot/align.py
@tool(name="align", nodes=["trossen-ai", "ook", "oop"])
async def align_tool(input_data: AlignInput, ctx: ToolContext):
    # Clean, self-contained implementation
    yield {"progress": 0.1, "message": "Starting alignment..."}
    # Implementation here
    return AlignOutput(success=True)
```

## Configuration

Node configurations in `src/conf/mcp/` control tool availability:

```yaml
# conf/mcp/ook.yaml
host: "0.0.0.0"
port: 8000
extras: ["gpu"]  # Enables GPU tools
tools:
  - align
  - convert_strokelist_to_batch
```

## Benefits

- ✅ **No More Split Architecture**: Everything in one logical place
- ✅ **No Complex Inheritance**: Simple functions with decorators
- ✅ **Type Safety**: Full Pydantic validation throughout
- ✅ **Better Discoverability**: Tools organized by purpose
- ✅ **Async Generators Preserved**: Progress reporting you love
- ✅ **Multi-Node Support**: Maintained and improved
- ✅ **Auto-Discovery**: No manual registration needed
- ✅ **Clean Migration Path**: Incremental adoption possible

This architecture maintains all the features you liked (async progress reporting, multi-node support) while dramatically simplifying the codebase and eliminating confusing abstractions.