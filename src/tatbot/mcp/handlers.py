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
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse JSON input_data: {e}")
            log.error(f"Input was: {repr(input_data)}")
            # Return default model instance
            return model_class()
    elif isinstance(input_data, dict):
        data_dict = input_data.copy()
    elif isinstance(input_data, model_class):
        return input_data
    else:
        log.error(f"Unexpected input_data type: {type(input_data)}")
        return model_class()
    
    # Handle common parameter aliases for better UX
    if model_class.__name__ == "RunOpInput":
        # Allow 'scene' as alias for 'scene_name'
        if 'scene' in data_dict and 'scene_name' not in data_dict:
            data_dict['scene_name'] = data_dict.pop('scene')
    
    try:
        # Create model instance
        return model_class(**data_dict)
    except (ValueError, TypeError) as e:
        log.error(f"Failed to create {model_class.__name__} from data: {e}")
        log.error(f"Data was: {data_dict}")
        log.error(f"Expected fields: {list(model_class.model_fields.keys())}")
        # Return default model instance
        return model_class()


@mcp_handler
async def run_op(input_data, ctx: Context):
    """Runs an operation, yields intermediate results, see available ops in tatbot.ops module.
    
    Parameters (JSON format):
    - op_name (str, required): Operation to run. Available: "stroke", "align", "reset", "sense"
    - scene_name (str, optional): Scene to use. Default: "default". Available: "tatbotlogo", "flower", "test", etc.
    - debug (bool, optional): Enable debug mode. Default: false
    
    Example usage:
    {"op_name": "stroke", "scene_name": "tatbotlogo"}
    {"op_name": "reset"}
    """
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
            progress=0.01, total=1.0, message=f"Created op for scene: {parsed_input.scene_name}"
        )
        
        async for result in op.run():
            # Safely log and report intermediate results without numpy arrays
            try:
                from tatbot.mcp.models import NumpyEncoder
                serialized_result = json.loads(json.dumps(result, cls=NumpyEncoder))
                log.info(f"Intermediate result: progress={serialized_result.get('progress')}, message={serialized_result.get('message')}")
                await ctx.report_progress(
                    progress=serialized_result.get('progress', 0.0), 
                    total=1.0, 
                    message=str(serialized_result.get('message', 'Processing...'))
                )
            except Exception as serialize_error:
                log.warning(f"Failed to serialize intermediate result: {serialize_error}")
                # Extract only basic info for logging and reporting
                progress = result.get('progress', 0.0) if isinstance(result, dict) else 0.0
                message = str(result.get('message', 'Processing...')) if isinstance(result, dict) else str(result)
                log.info(f"Intermediate result: progress={progress}, message={message}")
                await ctx.report_progress(
                    progress=progress, 
                    total=1.0, 
                    message=message
                )
        
        message = f"✅ Completed {parsed_input.op_name}"
        log.info(message)
        
        result = RunOpResult(
            message=message,
            success=True,
            op_name=parsed_input.op_name,
            scene_name=parsed_input.scene_name
        )
        # Use custom JSON encoder to handle any numpy arrays
        return json.loads(result.model_dump_json())
        
    except (KeyboardInterrupt, asyncio.CancelledError):
        message = "🛑⌨️ Keyboard/E-stop interrupt detected"
        log.error(message)
        result = RunOpResult(
            message=message,
            success=False,
            op_name=parsed_input.op_name,
            scene_name=parsed_input.scene_name
        )
        # Use custom JSON encoder to handle any numpy arrays
        return json.loads(result.model_dump_json())
    except Exception as e:
        message = f"❌ Exception when running op: {str(e)}"
        log.error(message)
        result = RunOpResult(
            message=message,
            success=False,
            op_name=parsed_input.op_name,
            scene_name=parsed_input.scene_name
        )
        # Use custom JSON encoder to handle any numpy arrays
        return json.loads(result.model_dump_json())
    finally:
        # Ensure cleanup is called even if op.run() never yielded
        if op and hasattr(op, 'cleanup'):
            try:
                op.cleanup()
            except Exception as cleanup_error:
                log.error(f"Error during op cleanup: {cleanup_error}")


@mcp_handler
async def ping_nodes(input_data, ctx: Context):
    """Ping nodes and report connectivity status.
    
    Parameters (JSON format):
    - nodes (list, optional): List of node names to ping. If not provided, pings all nodes.
    
    Example usage:
    {"nodes": ["ook", "eek"]}
    {} (pings all nodes)
    """
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

        result = PingNodesResponse(
            status=header,
            details=sorted(messages),
            all_success=all_success
        )
        return json.loads(result.model_dump_json())
        
    except Exception as e:
        log.error(f"Error pinging nodes: {e}")
        result = PingNodesResponse(
            status=f"❌ Error pinging nodes: {str(e)}",
            details=[],
            all_success=False
        )
        return json.loads(result.model_dump_json())


@mcp_handler
async def list_scenes(input_data, ctx: Context):
    """List available scenes from the config directory.
    
    No parameters required. Returns list of available scene names.
    
    Example usage:
    {}
    """
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
        result = ListScenesResponse(scenes=scenes, count=len(scenes))
        return json.loads(result.model_dump_json())
        
    except Exception as e:
        log.error(f"Error listing scenes: {e}")
        result = ListScenesResponse(scenes=[], count=0)
        return json.loads(result.model_dump_json())


@mcp_handler
async def list_nodes(input_data, ctx: Context):
    """List available network nodes.
    
    No parameters required. Returns list of available node names.
    
    Example usage:
    {}
    """
    # Import locally to avoid circular imports
    from tatbot.mcp.models import ListNodesResponse
    from tatbot.utils.net import NetworkManager
    
    try:
        net = NetworkManager()
        
        node_names = [node.name for node in net.nodes]
        log.info(f"Found {len(node_names)} nodes")
        
        result = ListNodesResponse(nodes=node_names, count=len(node_names))
        return json.loads(result.model_dump_json())
        
    except Exception as e:
        log.error(f"Error listing nodes: {e}")
        result = ListNodesResponse(nodes=[], count=0)
        return json.loads(result.model_dump_json())


@mcp_handler
async def list_ops(input_data, ctx: Context):
    """List available operations, optionally filtered by node.
    
    Parameters (JSON format):
    - node_name (str, optional): Filter operations by specific node. If not provided, lists all operations.
    
    Example usage:
    {"node_name": "ook"}
    {} (lists all operations)
    """
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
            
            result = ListOpsResponse(
                ops=sorted(ops), 
                count=len(ops), 
                node_name=parsed_input.node_name
            )
            return json.loads(result.model_dump_json())
        else:
            # List all unique ops across all nodes
            all_ops = set()
            for node_ops in NODE_AVAILABLE_OPS.values():
                all_ops.update(node_ops)
            
            ops = sorted(list(all_ops))
            log.info(f"Found {len(ops)} unique ops across all nodes")
            
            result = ListOpsResponse(ops=ops, count=len(ops))
            return json.loads(result.model_dump_json())
        
    except Exception as e:
        log.error(f"Error listing ops: {e}")
        result = ListOpsResponse(ops=[], count=0, node_name=parsed_input.node_name)
        return json.loads(result.model_dump_json())



# Export available tools for discoverability
__all__ = list(_REGISTRY.keys())