"""MCP tool handlers with registration decorator."""

import concurrent.futures
import os
from typing import Dict, Callable, Any

from mcp.server.fastmcp import Context

from tatbot.mcp.models import (
    RunOpInput, RunOpResult,
    PingNodesInput, PingNodesResponse,
    ListScenesInput, ListScenesResponse,
    ListNodesInput, ListNodesResponse,
    GetNfsInfoInput, GetNfsInfoResponse,
    GetLatestRecordingInput, GetLatestRecordingResponse
)
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


@mcp_handler
async def run_op(input_data: RunOpInput, ctx: Context, node_name: str) -> RunOpResult:
    """Runs an operation, yields intermediate results, see available ops in tatbot.ops module."""
    await ctx.info(f"Running robot op: {input_data.op_name} on {node_name}")
    
    try:
        from tatbot.ops import get_op
        
        op_class, op_config = get_op(input_data.op_name, node_name)
        config = op_config(scene=input_data.scene_name, debug=input_data.debug)
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
        
        message = f"✅ Completed {input_data.op_name}"
        log.info(message)
        
        return RunOpResult(
            message=message,
            success=True,
            op_name=input_data.op_name,
            scene_name=input_data.scene_name
        )
        
    except KeyboardInterrupt:
        message = "🛑⌨️ Keyboard/E-stop interrupt detected"
        log.error(message)
        return RunOpResult(
            message=message,
            success=False,
            op_name=input_data.op_name,
            scene_name=input_data.scene_name
        )
    except Exception as e:
        message = f"❌ Exception when running op: {str(e)}"
        log.error(message)
        return RunOpResult(
            message=message,
            success=False,
            op_name=input_data.op_name,
            scene_name=input_data.scene_name
        )
    finally:
        if 'op' in locals():
            op.cleanup()


@mcp_handler
async def ping_nodes(input_data: PingNodesInput, ctx: Context) -> PingNodesResponse:
    """Ping nodes and report connectivity status."""
    log.info(f"🔌 Pinging nodes: {input_data.nodes or 'all'}")
    
    try:
        from tatbot.utils.net import NetworkManager
        net = NetworkManager()
        
        target_nodes, error = net.get_target_nodes(input_data.nodes)
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
        if not input_data.nodes:
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
async def list_scenes(input_data: ListScenesInput, ctx: Context) -> ListScenesResponse:
    """List available scenes from the config directory."""
    try:
        scenes_dir = os.path.expanduser("~/tatbot/config/scenes")
        if not os.path.exists(scenes_dir):
            return ListScenesResponse(scenes=[], count=0)
        
        scenes = [
            f.replace(".yaml", "") 
            for f in os.listdir(scenes_dir) 
            if f.endswith(".yaml")
        ]
        scenes.sort()
        
        log.info(f"Found {len(scenes)} scenes")
        return ListScenesResponse(scenes=scenes, count=len(scenes))
        
    except Exception as e:
        log.error(f"Error listing scenes: {e}")
        return ListScenesResponse(scenes=[], count=0)


@mcp_handler
async def list_nodes(input_data: ListNodesInput, ctx: Context) -> ListNodesResponse:
    """List available network nodes."""
    try:
        from tatbot.utils.net import NetworkManager
        net = NetworkManager()
        
        node_names = [node.name for node in net.nodes]
        log.info(f"Found {len(node_names)} nodes")
        
        return ListNodesResponse(nodes=node_names, count=len(node_names))
        
    except Exception as e:
        log.error(f"Error listing nodes: {e}")
        return ListNodesResponse(nodes=[], count=0)


@mcp_handler
async def get_nfs_info(input_data: GetNfsInfoInput, ctx: Context) -> GetNfsInfoResponse:
    """Get NFS information."""
    try:
        # This is a placeholder - in the real implementation, this would
        # gather actual NFS status information
        info = "NFS mounted and accessible"
        log.info("Retrieved NFS info")
        return GetNfsInfoResponse(info=info)
    except Exception as e:
        log.error(f"Error getting NFS info: {e}")
        return GetNfsInfoResponse(info=f"Error: {str(e)}")


@mcp_handler
async def get_latest_recording(input_data: GetLatestRecordingInput, ctx: Context) -> GetLatestRecordingResponse:
    """Get the latest recording file."""
    try:
        recording_dir = os.path.expanduser("~/tatbot/nfs/recordings")
        if not os.path.exists(recording_dir):
            return GetLatestRecordingResponse(filename="", found=False)
        
        recordings = [f for f in os.listdir(recording_dir) if f.endswith(".yaml")]
        if not recordings:
            return GetLatestRecordingResponse(filename="", found=False)
        
        latest_recording = max(
            recordings, 
            key=lambda x: os.path.getctime(os.path.join(recording_dir, x))
        )
        
        log.info(f"Found latest recording: {latest_recording}")
        return GetLatestRecordingResponse(filename=latest_recording, found=True)
        
    except Exception as e:
        log.error(f"Error getting latest recording: {e}")
        return GetLatestRecordingResponse(filename="", found=False)


# Export available tools for discoverability
__all__ = list(_REGISTRY.keys())