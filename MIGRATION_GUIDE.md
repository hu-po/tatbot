# Tatbot Tools Migration Guide

This guide covers migrating from the old split `mcp/handlers` + `ops` system to the new unified `tatbot.tools` architecture.

## ✅ Completed (All Phases)

### Phase 1: Foundation
- [x] Created unified `src/tatbot/tools/` module structure
- [x] Implemented `@tool` decorator and registry system
- [x] Created `ToolContext` wrapper for clean API
- [x] Migrated system tools: `ping_nodes`, `list_scenes`, `list_nodes`
- [x] Migrated GPU tool: `convert_strokelist_to_batch`
- [x] Updated documentation (`docs/tools.md`, updated `docs/mcp.md`)
- [x] Created integration test framework

### Phase 2: Robot Tools Migration
- [x] Migrated `reset` tool (simplest robot operation)
- [x] Migrated `align` tool with full dataset recording
- [x] Migrated `sense` tool with camera and sensor capture
- [x] Migrated `stroke` tool with joystick support and resumption
- [x] Updated MCP server integration to use new registry
- [x] Created comprehensive input/output models

### Phase 3: Integration & Testing
- [x] Updated tools registry to import all migrated tools
- [x] Modified MCP server to use `get_tools_for_node()` 
- [x] Verified node-specific tool availability
- [x] Tested backwards compatibility
- [x] Validated complete migration with comprehensive test suite

## 🎯 Migration Results

### Step 1: Migrate Robot Operations

Migrate the remaining robot tools from `ops/` to `tools/robot/`:

#### A. Align Tool (`ops/record_align.py` → `tools/robot/align.py`)

```python
@tool(
    name="align",
    nodes=["trossen-ai", "ook", "oop"],
    description="Generate alignment strokes for robot calibration", 
    input_model=AlignInput,
    output_model=AlignOutput
)
async def align_tool(input_data: AlignInput, ctx: ToolContext):
    # Extract logic from AlignOp._run() method
    yield {"progress": 0.2, "message": "Generating alignment strokes..."}
    # ... implementation
    return AlignOutput(success=True, stroke_count=len(strokes.strokes))
```

#### B. Stroke Tool (`ops/record_stroke.py` → `tools/robot/stroke.py`)

```python
@tool(
    name="stroke", 
    nodes=["trossen-ai", "ook", "oop"],
    description="Execute artistic strokes on paper/canvas",
    input_model=StrokeInput,
    output_model=StrokeOutput
)
async def stroke_tool(input_data: StrokeInput, ctx: ToolContext):
    # Extract logic from StrokeOp._run() method
    # ... implementation
```

#### C. Sense Tool (`ops/record_sense.py` → `tools/robot/sense.py`)

```python
@tool(
    name="sense",
    nodes=["trossen-ai"],  # Only available on robot with sensors
    description="Capture environmental data from cameras and sensors",
    input_model=SenseInput,
    output_model=SenseOutput
)
async def sense_tool(input_data: SenseInput, ctx: ToolContext):
    # Extract logic from SenseOp._run() method
    # ... implementation
```

#### D. Reset Tool (`ops/reset.py` → `tools/robot/reset.py`)

```python
@tool(
    name="reset",
    nodes=["trossen-ai", "ook", "oop"], 
    description="Reset robot to safe/ready position",
    input_model=ResetInput,
    output_model=ResetOutput
)
async def reset_tool(input_data: ResetInput, ctx: ToolContext):
    # Extract logic from ResetOp.run() method
    # ... implementation
```

### Step 2: Update MCP Server Integration

Modify `src/tatbot/mcp/server.py` to use the new tools registry:

```python
# OLD: from tatbot.mcp import handlers
# NEW: from tatbot.tools import get_tools_for_node

def _register_tools(mcp: FastMCP, tool_names: Optional[List[str]], node_name: str) -> None:
    """Register tools dynamically based on configuration."""
    # OLD: available_tools = handlers.get_available_tools()
    # NEW: available_tools = get_tools_for_node(node_name)
    
    tools_to_register = tool_names or list(available_tools.keys())
    
    log.info(f"Registering tools for {node_name}: {tools_to_register}")
    
    for tool_name in tools_to_register:
        if tool_name in available_tools:
            tool_fn = available_tools[tool_name]
            mcp.tool()(tool_fn)
            log.info(f"✅ Registered tool: {tool_name}")
        else:
            log.warning(f"⚠️ Tool {tool_name} not found for node {node_name}")
```

### Step 3: Update Tool Registration in `tools/__init__.py`

```python
def register_all_tools() -> None:
    """Auto-register all tools by importing tool modules.""" 
    try:
        # System tools
        from tatbot.tools.system import list_nodes, list_scenes, ping_nodes  # noqa: F401
        # GPU tools
        from tatbot.tools.gpu import convert_strokes  # noqa: F401
        # Robot tools (add as migrated)
        from tatbot.tools.robot import align, stroke, sense, reset  # noqa: F401
        
        log.info(f"Auto-registered {len(_REGISTRY)} tools")
    except ImportError as e:
        log.debug(f"Some tool modules not yet available: {e}")
```

## 🧪 Testing Each Migration

For each migrated tool, test:

1. **Tool Registration**: Verify tool appears in registry
2. **Node Availability**: Verify tool only appears on correct nodes  
3. **Input/Output**: Test with sample data
4. **Progress Reporting**: Verify async generator pattern works
5. **Error Handling**: Test error cases

Example test:
```python
async def test_align_tool():
    tools = get_tools_for_node("trossen-ai")
    align_tool = tools["align"]
    
    input_data = '{"scene_name": "default"}'
    mock_context = create_mock_context("trossen-ai")
    
    result = await align_tool(input_data, mock_context)
    assert result["success"] is True
```

## 🗑️ Cleanup (Phase 3)

Once all tools are migrated and tested:

1. **Remove Old Files**:
   - `src/tatbot/mcp/handlers.py`  
   - `src/tatbot/ops/` directory
   
2. **Update Imports**: Search codebase for old imports and update

3. **Update Config**: Remove `list_ops` tool (no longer needed)

4. **Update Tests**: Update any existing tests to use new system

## 📋 Benefits After Migration

- ✅ **Single Source of Truth**: All tools in `tatbot.tools`
- ✅ **No Complex Inheritance**: Simple functions with decorators
- ✅ **Better Type Safety**: Full Pydantic validation
- ✅ **Cleaner Codebase**: Eliminates confusing abstractions
- ✅ **Preserved Features**: Async generators and multi-node support maintained
- ✅ **Better Documentation**: Self-documenting via decorators

## 🚀 Implementation Timeline

- **Week 1**: Migrate `align` and `reset` tools (simpler ones first)
- **Week 2**: Migrate `stroke` and `sense` tools (more complex)
- **Week 3**: Update MCP server integration and test thoroughly
- **Week 4**: Cleanup old code and update remaining documentation

This migration preserves all functionality while dramatically simplifying the architecture!