"""MCP tool handlers with registration decorator."""

import asyncio
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Callable, Dict

from mcp.server.fastmcp import Context

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



@mcp_handler
async def convert_strokelist_to_batch(input_data, ctx: Context):
    """Convert StrokeList to StrokeBatch using GPU-accelerated IK via shared NFS.
    
    This tool requires GPU support and should only be available on GPU-enabled nodes.
    Uses shared NFS for efficient file-based communication between nodes.
    
    Parameters (JSON format):
    - strokes_file_path (str, required): Path to strokes YAML file on shared NFS
    - strokebatch_file_path (str, required): Path where strokebatch should be saved on shared NFS
    - scene_name (str, required): Scene name for conversion parameters
    - first_last_rest (bool, optional): Apply first/last rest positions. Default: true
    - use_ee_offsets (bool, optional): Apply end-effector offsets. Default: true
    
    Returns:
    - success (bool): Whether conversion succeeded
    - message (str): Status message with file path
    
    Example usage:
    {"strokes_file_path": "/nfs/path/strokes.yaml", "strokebatch_file_path": "/nfs/path/batch.safetensors", "scene_name": "tatbotlogo"}
    """
    import base64
    import io

    import safetensors.numpy
    import yaml

    from tatbot.data.stroke import StrokeList
    from tatbot.gen.batch import strokebatch_from_strokes
    from tatbot.mcp.models import ConvertStrokeListInput, ConvertStrokeListResponse

    # Parse input data
    parsed_input = _parse_input_data(input_data, ConvertStrokeListInput)
    
    try:
        # Check if this node has GPU support
        import hydra
        from pathlib import Path
        import yaml
        
        cfg = hydra.compose(config_name="config")
        node_name = ctx.fastmcp.name.split(".", 1)[1] if "." in ctx.fastmcp.name else ctx.fastmcp.name
        
        # Load node-specific MCP config or fall back to current config
        config_dir = Path(__file__).parent.parent.parent / "conf" / "mcp"
        node_config_file = config_dir / f"{node_name}.yaml"
        
        if node_config_file.exists():
            with open(node_config_file, 'r') as f:
                node_cfg = yaml.safe_load(f)
        else:
            node_cfg = cfg.mcp
        
        if "gpu" not in node_cfg.get("extras", []):
            return ConvertStrokeListResponse(
                strokebatch_base64="",
                success=False,
                message=f"Node {node_name} does not have GPU support"
            ).model_dump()
        
        await ctx.info(f"Converting StrokeList to StrokeBatch on GPU node {node_name}")
        
        # Wait for NFS file synchronization with timeout
        import time
        from pathlib import Path
        
        strokes_path = Path(parsed_input.strokes_file_path)
        max_wait_time = 10  # seconds
        start_time = time.time()
        
        await ctx.report_progress(0.1, 1.0, "Waiting for NFS file synchronization...")
        
        while not strokes_path.exists():
            if time.time() - start_time > max_wait_time:
                raise FileNotFoundError(f"Strokes file not found after {max_wait_time}s: {strokes_path}")
            
            log.info(f"Waiting for NFS sync of {strokes_path}...")
            time.sleep(0.5)
        
        # Also check for the array files that should be created by to_yaml_with_arrays
        base_dir = strokes_path.parent
        expected_arrays = []
        
        # Wait a bit more to ensure all array files are synced
        await ctx.report_progress(0.15, 1.0, "Verifying array files are synced...")
        time.sleep(1.0)
        
        # Load StrokeList from file path using the proper method
        from tatbot.data.stroke import StrokeList
        strokes = StrokeList.from_yaml_with_arrays(parsed_input.strokes_file_path)
        log.info(f"Successfully loaded StrokeList with {len(strokes.strokes)} stroke pairs")
        
        # Load scene configuration
        from tatbot.main import compose_and_validate_scene
        scene = compose_and_validate_scene(parsed_input.scene_name)
        
        await ctx.report_progress(0.2, 1.0, "Loaded strokes and scene")
        
        # Perform GPU-accelerated conversion
        strokebatch = strokebatch_from_strokes(
            scene, 
            strokes, 
            first_last_rest=parsed_input.first_last_rest,
            use_ee_offsets=parsed_input.use_ee_offsets
        )
        
        await ctx.report_progress(0.8, 1.0, "Conversion complete, saving to shared NFS")
        
        # Save strokebatch directly to the shared NFS path
        strokebatch.save(parsed_input.strokebatch_file_path)
        
        await ctx.report_progress(1.0, 1.0, "Conversion successful")
        
        result = ConvertStrokeListResponse(
            strokebatch_base64="",  # Not needed since file is saved to NFS
            success=True,
            message=f"Successfully converted StrokeList to StrokeBatch at {parsed_input.strokebatch_file_path}"
        )
        return json.loads(result.model_dump_json())
        
    except Exception as e:
        log.error(f"Error converting strokelist to batch: {e}")
        result = ConvertStrokeListResponse(
            strokebatch_base64="",
            success=False,
            message=f"Conversion failed: {str(e)}"
        )
        return json.loads(result.model_dump_json())


# Export available tools for discoverability
__all__ = list(_REGISTRY.keys())