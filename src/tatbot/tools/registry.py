"""Tool registry and decorator system for unified tools architecture."""

import ast
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
from mcp.server.fastmcp import Context
from pydantic import ValidationError

from tatbot.tools.base import (
    ToolContext,
    ToolDefinition,
    ToolFunction,
    ToolInput,
    ToolOutput,
)
from tatbot.utils.exceptions import ConfigurationError, SerializationError
from tatbot.utils.log import get_logger

log = get_logger("tools.registry", "📋")

# Global tool registry
_REGISTRY: Dict[str, ToolDefinition] = {}


def tool(
    name: Optional[str] = None,
    nodes: List[str] = ["*"],
    description: str = "",
    input_model: type[ToolInput] = ToolInput,
    output_model: type[ToolOutput] = ToolOutput,
    requires: List[str] = None,
) -> Callable:
    """
    Decorator to register a tool function.
    
    Args:
        name: Tool name (defaults to function name)
        nodes: List of node names where tool is available (["*"] for all nodes)
        description: Tool description for documentation
        input_model: Pydantic model for input validation
        output_model: Pydantic model for output validation  
        requires: List of requirements (e.g., ["gpu"] for GPU-only tools)
    
    Example:
        @tool(
            name="align",
            nodes=["trossen-ai", "ook", "oop"],
            description="Generate alignment strokes for robot calibration",
            input_model=AlignInput,
            output_model=AlignOutput
        )
        async def align_tool(input_data: AlignInput, ctx: ToolContext):
            yield {"progress": 0.1, "message": "Starting..."}
            # Tool logic here
            return AlignOutput(success=True, message="Complete")
    """
    if requires is None:
        requires = []
        
    def decorator(func: ToolFunction) -> ToolFunction:
        tool_name = name or func.__name__
        
        # Validate function signature - must have input_data and ctx parameters
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        if len(params) != 2 or 'input_data' not in params or 'ctx' not in params:
            raise ValueError(
                f"Tool function {tool_name} must have signature: "
                "(input_data: {input_model.__name__}, ctx: ToolContext)"
            )
        
        # Create tool definition first (needed in wrapper)
        definition = ToolDefinition(
            name=tool_name,
            func=func,  # Will be updated with wrapper later
            nodes=nodes,
            description=description or func.__doc__ or "",
            input_model=input_model,
            output_model=output_model,
            requires=requires,
        )
        
        # Create wrapper that completely hides ctx from MCP schema
        async def mcp_exposed_wrapper(ctx: Context, **kwargs):
            """
            MCP-exposed wrapper with hidden Context parameter.
            
            FastMCP will inject Context automatically but won't expose it in the 
            client-visible schema. Clients only see the **kwargs parameters.
            """
            
            # Filter out any framework-related fields that clients might send
            framework_fields = ['ctx', 'context', 'mcp_context', 'tool_context']
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                              if k not in framework_fields and not isinstance(v, Context)}
            
            # Log and remove any framework fields if present
            for field in framework_fields:
                if field in kwargs:
                    removed_value = kwargs[field]
                    log.warning(f"🔧 {tool_name}: Client sent '{field}' - ignoring: {removed_value}")
            
            # Extract node name from FastMCP context
            if ctx and hasattr(ctx, 'fastmcp') and ctx.fastmcp:
                server_name = ctx.fastmcp.name
                node_name = server_name.split(".", 1)[1] if "." in server_name else server_name
            else:
                # Fallback to hostname if context is not available
                import socket
                node_name = socket.gethostname()
            
            # Create unified ToolContext
            tool_ctx = ToolContext(node_name=node_name, mcp_context=ctx)
            
            try:
                # Parse input data from kwargs
                parsed_input = _parse_input_data(filtered_kwargs or {}, input_model, tool_name)
                
                # Enable debug logging if requested
                if hasattr(parsed_input, 'debug') and parsed_input.debug:
                    log.setLevel(logging.DEBUG)
                
                await tool_ctx.info(f"Running tool: {tool_name} on {node_name}")
                
                # Check node availability
                if not definition.is_available_on_node(node_name):
                    raise ConfigurationError(
                        f"Tool {tool_name} is not available on node {node_name}. "
                        f"Available on: {definition.nodes}"
                    )
                
                # Check requirements (load node config)
                node_config = _load_node_config(node_name)
                if not definition.check_requirements(node_config):
                    raise ConfigurationError(
                        f"Tool {tool_name} requires {definition.requires} but node {node_name} "
                        f"only has {node_config.get('extras', [])}"
                    )
                
                # Execute the original tool function
                result = None
                async for progress_data in func(parsed_input, tool_ctx):
                    if isinstance(progress_data, dict):
                        # Handle progress reports
                        progress = progress_data.get('progress', 0.0)
                        message = str(progress_data.get('message', 'Processing...'))
                        await tool_ctx.report_progress(progress, message)
                    else:
                        # Final result
                        result = progress_data
                        break
                
                # Ensure we have a result
                if result is None:
                    result = output_model(success=True, message=f"Completed {tool_name}")
                
                # Validate output
                if not isinstance(result, output_model):
                    log.warning(f"Tool {tool_name} returned {type(result)}, expected {output_model}")
                    result = output_model(success=True, message=str(result))
                
                log.info(f"✅ Tool {tool_name} completed successfully")
                return json.loads(result.model_dump_json())
                
            except ValidationError as e:
                error_msg = f"❌ Input validation failed for {tool_name}: {e}"
                log.error(error_msg)
                error_result = output_model(success=False, message=error_msg)
                return json.loads(error_result.model_dump_json())
                
            except ConfigurationError as e:
                error_msg = f"❌ Configuration error in {tool_name}: {e}"
                log.error(error_msg)
                await tool_ctx.error(error_msg)
                error_result = output_model(success=False, message=error_msg)
                return json.loads(error_result.model_dump_json())
                
            except Exception as e:
                error_msg = f"❌ Unexpected error in {tool_name}: {e}"
                log.error(error_msg)
                await tool_ctx.error(error_msg)
                error_result = output_model(success=False, message=error_msg)
                return json.loads(error_result.model_dump_json())
        
        # Update tool definition with the MCP-exposed wrapper function
        definition.func = mcp_exposed_wrapper
        
        # Register the tool
        if tool_name in _REGISTRY:
            log.warning(f"Tool {tool_name} is being re-registered")
        
        _REGISTRY[tool_name] = definition
        log.debug(f"Registered tool: {tool_name} (nodes: {nodes})")
        
        return mcp_exposed_wrapper
    
    return decorator


def _parse_input_data(input_data: Union[str, dict, Any], model_class: type, tool_name: str) -> Any:
    """Parse input_data into the specified model class."""
    if isinstance(input_data, str):
        try:
            data_dict = json.loads(input_data) if input_data.strip() else {}
        except json.JSONDecodeError as e:
            # Fallback: accept Python-literal style dict strings (from some clients)
            try:
                literal = ast.literal_eval(input_data)
                if isinstance(literal, dict):
                    data_dict = literal
                else:
                    raise ValueError("Parsed literal is not a dict")
            except Exception as e2:
                log.error(f"Failed to parse input for {tool_name} as JSON or python-literal dict: {e}; {e2}")
                raise SerializationError(f"Invalid input for {tool_name}: {e}")
    elif isinstance(input_data, dict):
        data_dict = input_data.copy()
    elif isinstance(input_data, model_class):
        return input_data
    else:
        log.warning(f"Unexpected input type {type(input_data)} for {tool_name}, using empty dict")
        data_dict = {}
    
    # Filter out 'ctx' field that some clients erroneously include in input_data
    if 'ctx' in data_dict:
        ctx_value = data_dict.pop('ctx')
        log.warning(
            f"🔧 {tool_name}: Removed 'ctx' from input_data. "
            f"Context should be automatically provided by FastMCP, not sent by client. "
            f"Removed value: {ctx_value}"
        )
    
    # Also filter out other framework-related fields that shouldn't be in user input
    framework_fields = ['context', 'mcp_context', 'tool_context']
    for field in framework_fields:
        if field in data_dict:
            removed_value = data_dict.pop(field)
            log.warning(f"🔧 {tool_name}: Removed framework field '{field}' from input: {removed_value}")
    
    try:
        return model_class(**data_dict)
    except (ValueError, TypeError) as e:
        log.error(f"Failed to create {model_class.__name__} for {tool_name}: {e}")
        raise ConfigurationError(f"Invalid input for {tool_name}: {e}")


def _load_node_config(node_name: str) -> dict:
    """Load node configuration from MCP config files."""
    try:
        config_dir = Path(__file__).parent.parent.parent / "conf" / "mcp"
        node_config_file = config_dir / f"{node_name}.yaml"
        
        if node_config_file.exists():
            with open(node_config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            log.warning(f"No config file found for node {node_name}")
            return {}
    except Exception as e:
        log.error(f"Error loading config for node {node_name}: {e}")
        return {}


def get_tools_for_node(node_name: str) -> Dict[str, Callable]:
    """Get all tools available for a specific node."""
    available_tools = {}
    node_config = _load_node_config(node_name)
    
    for tool_name, definition in _REGISTRY.items():
        if (definition.is_available_on_node(node_name) and 
            definition.check_requirements(node_config)):
            available_tools[tool_name] = definition.func
    
    log.info(f"Found {len(available_tools)} tools for node {node_name}")
    return available_tools


def get_all_tools() -> Dict[str, ToolDefinition]:
    """Get all registered tools."""
    return _REGISTRY.copy()


def register_all_tools() -> None:
    """Auto-register all tools by importing tool modules."""
    initial_count = len(_REGISTRY)
    
    # Import GPU tools
    try:
        from tatbot.tools.gpu import convert_strokes  # noqa: F401
        log.debug("Imported GPU tools")
    except ImportError as e:
        log.debug(f"GPU tools not available: {e}")
    
    # Import robot tools
    try:
        from tatbot.tools.robot import align, reset, sense, stroke  # noqa: F401
        log.debug("Imported robot tools")
    except ImportError as e:
        log.debug(f"Robot tools not available: {e}")
    
    # Import system tools
    try:
        from tatbot.tools.system import (  # noqa: F401
            list_nodes,
            list_recordings,
            list_scenes,
            ping_nodes,
        )
        log.debug("Imported system tools")
    except ImportError as e:
        log.debug(f"System tools not available: {e}")
    
    registered_count = len(_REGISTRY) - initial_count
    log.info(f"Auto-registered {registered_count} new tools (total: {len(_REGISTRY)})")
