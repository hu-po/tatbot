"""MCP tool handlers with registration decorator."""

import asyncio
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Callable, Dict

from mcp.server.fastmcp import Context

# Import models locally inside functions to avoid circular imports
from tatbot.utils.log import get_logger

log = get_logger("mcp.handlers", "🔌🛠️")

# Registry for MCP handlers
_REGISTRY: Dict[str, Callable] = {}


def mcp_handler(fn: Callable) -> Callable:
    """Decorator to register MCP tool handlers."""
    _REGISTRY[fn.__name__] = fn
    return fn


def get_available_tools() -> Dict[str, Callable]:
    """Get all registered MCP tools."""
    return _REGISTRY.copy()


def _parse_input_data(input_data, model_class):
    """Parse input_data string or dict into the specified model class."""
    if isinstance(input_data, str):
        try:
            # Parse JSON string into dict
            data_dict = json.loads(input_data) if input_data.strip() else {}
            # Create model instance
            return model_class(**data_dict)
        except (json.JSONDecodeError, ValueError) as e:
            log.error(f"Failed to parse input_data: {e}")
            # Return default model instance
            return model_class()
    elif isinstance(input_data, dict):
        try:
            # Create model instance directly from dict
            return model_class(**input_data)
        except (ValueError, TypeError) as e:
            log.error(f"Failed to create model from dict: {e}")
            return model_class()
    elif isinstance(input_data, model_class):
        return input_data
    else:
        log.error(f"Unexpected input_data type: {type(input_data)}")
        return model_class()


@mcp_handler
async def run_op(input_data, ctx: Context):
    """Runs an operation, yields intermediate results, see available ops in tatbot.ops module."""
    # Import locally to avoid circular imports
    from tatbot.mcp.models import RunOpInput, RunOpResult
    from tatbot.ops import get_op

    # Parse input data
    parsed_input = _parse_input_data(input_data, RunOpInput)
    
    # Extract node name from the server name (format: "tatbot.{node_name}")
    server_name = ctx.fastmcp.name
    node_name = server_name.split(".", 1)[1] if "." in server_name else server_name
    
    await ctx.info(f"Running robot op: {parsed_input.op_name} on {node_name}")
    
    op = None  # Initialize op for cleanup
    try:
        
        op_class, op_config = get_op(parsed_input.op_name, node_name)
        config = op_config(scene=parsed_input.scene_name, debug=parsed_input.debug)
        op = op_class(config)
        
        await ctx.report_progress(
            progress=0.01, total=1.0, message=f"Created op: {config}"
        )
        
        async for result in op.run():
            log.info(f"Intermediate result: {result}")
            await ctx.report_progress(
                progress=result['progress'], 
                total=1.0, 
                message=result['message']
            )
        
        message = f"✅ Completed {parsed_input.op_name}"
        log.info(message)
        
        return RunOpResult(
            message=message,
            success=True,
            op_name=parsed_input.op_name,
            scene_name=parsed_input.scene_name
        )
        
    except (KeyboardInterrupt, asyncio.CancelledError):
        message = "🛑⌨️ Keyboard/E-stop interrupt detected"
        log.error(message)
        return RunOpResult(
            message=message,
            success=False,
            op_name=parsed_input.op_name,
            scene_name=parsed_input.scene_name
        )
    except Exception as e:
        message = f"❌ Exception when running op: {str(e)}"
        log.error(message)
        return RunOpResult(
            message=message,
            success=False,
            op_name=parsed_input.op_name,
            scene_name=parsed_input.scene_name
        )
    finally:
        # Ensure cleanup is called even if op.run() never yielded
        if op and hasattr(op, 'cleanup'):
            try:
                op.cleanup()
            except Exception as cleanup_error:
                log.error(f"Error during op cleanup: {cleanup_error}")


@mcp_handler
async def ping_nodes(input_data, ctx: Context):
    """Ping nodes and report connectivity status."""
    # Import locally to avoid circular imports
    from tatbot.mcp.models import PingNodesInput, PingNodesResponse
    from tatbot.utils.net import NetworkManager

    # Parse input data
    parsed_input = _parse_input_data(input_data, PingNodesInput)
    
    log.info(f"🔌 Pinging nodes: {parsed_input.nodes or 'all'}")
    
    try:
        log.info("Creating NetworkManager...")
        net = NetworkManager()
        log.info(f"NetworkManager created, loaded {len(net.nodes)} nodes")
        
        target_nodes, error = net.get_target_nodes(parsed_input.nodes)
        log.info(f"get_target_nodes returned: {len(target_nodes)} nodes, error: {error}")
        
        if error:
            return PingNodesResponse(
                status=error,
                details=[],
                all_success=False
            )
        
        if not target_nodes:
            return PingNodesResponse(
                status="No nodes to ping.",
                details=[],
                all_success=True
            )

        messages = []
        all_success = True

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_node = {
                executor.submit(net._test_node_connection, node): node 
                for node in target_nodes
            }
            for future in concurrent.futures.as_completed(future_to_node):
                _, success, message = future.result()
                messages.append(message)
                if not success:
                    all_success = False

        header = (
            "✅ All specified nodes are responding"
            if all_success
            else "❌ Some specified nodes are not responding"
        )
        if not parsed_input.nodes:
            header = (
                "✅ All nodes are responding" 
                if all_success 
                else "❌ Some nodes are not responding"
            )

        return PingNodesResponse(
            status=header,
            details=sorted(messages),
            all_success=all_success
        )
        
    except Exception as e:
        log.error(f"Error pinging nodes: {e}")
        return PingNodesResponse(
            status=f"❌ Error pinging nodes: {str(e)}",
            details=[],
            all_success=False
        )


@mcp_handler
async def list_scenes(input_data, ctx: Context):
    """List available scenes from the config directory."""
    # Import locally to avoid circular imports
    from tatbot.mcp.models import ListScenesResponse
    
    try:
        scenes_dir = Path("~/tatbot/src/conf/scenes").expanduser().resolve()
        if not scenes_dir.exists():
            return ListScenesResponse(scenes=[], count=0)
        
        scenes = [
            f.replace(".yaml", "") 
            for f in os.listdir(str(scenes_dir)) 
            if f.endswith(".yaml")
        ]
        scenes.sort()
        
        log.info(f"Found {len(scenes)} scenes")
        return ListScenesResponse(scenes=scenes, count=len(scenes))
        
    except Exception as e:
        log.error(f"Error listing scenes: {e}")
        return ListScenesResponse(scenes=[], count=0)


@mcp_handler
async def list_nodes(input_data, ctx: Context):
    """List available network nodes."""
    # Import locally to avoid circular imports
    from tatbot.mcp.models import ListNodesResponse
    from tatbot.utils.net import NetworkManager
    
    try:
        net = NetworkManager()
        
        node_names = [node.name for node in net.nodes]
        log.info(f"Found {len(node_names)} nodes")
        
        return ListNodesResponse(nodes=node_names, count=len(node_names))
        
    except Exception as e:
        log.error(f"Error listing nodes: {e}")
        return ListNodesResponse(nodes=[], count=0)


@mcp_handler
async def list_ops(input_data, ctx: Context):
    """List available operations, optionally filtered by node."""
    # Import locally to avoid circular imports
    from tatbot.mcp.models import ListOpsInput, ListOpsResponse
    from tatbot.ops import NODE_AVAILABLE_OPS

    # Parse input data
    parsed_input = _parse_input_data(input_data, ListOpsInput)
    
    try:
        if parsed_input.node_name:
            # List ops for specific node
            if parsed_input.node_name not in NODE_AVAILABLE_OPS:
                log.warning(f"Node {parsed_input.node_name} not found in available ops")
                return ListOpsResponse(
                    ops=[], 
                    count=0, 
                    node_name=parsed_input.node_name
                )
            
            ops = NODE_AVAILABLE_OPS[parsed_input.node_name]
            log.info(f"Found {len(ops)} ops for node {parsed_input.node_name}")
            
            return ListOpsResponse(
                ops=sorted(ops), 
                count=len(ops), 
                node_name=parsed_input.node_name
            )
        else:
            # List all unique ops across all nodes
            all_ops = set()
            for node_ops in NODE_AVAILABLE_OPS.values():
                all_ops.update(node_ops)
            
            ops = sorted(list(all_ops))
            log.info(f"Found {len(ops)} unique ops across all nodes")
            
            return ListOpsResponse(ops=ops, count=len(ops))
        
    except Exception as e:
        log.error(f"Error listing ops: {e}")
        return ListOpsResponse(ops=[], count=0, node_name=parsed_input.node_name)



# Export available tools for discoverability
__all__ = list(_REGISTRY.keys())