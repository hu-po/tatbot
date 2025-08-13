"""Control tools for managing visualization servers."""

from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.viz import get_server_status, list_servers, stop_server
from tatbot.tools.viz.models import (
    ListVizServersInput,
    ListVizServersOutput,
    StatusVizInput,
    StatusVizOutput,
    StopVizInput,
    StopVizOutput,
)
from tatbot.utils.log import get_logger

log = get_logger("tools.viz.control", "🎛️")


@tool(
    name="stop_viz_server",
    nodes=["oop", "ook", "eek"],
    description="Stop a running visualization server",
    input_model=StopVizInput,
    output_model=StopVizOutput,
    requires=["viz"],
)
async def stop_viz_server(input_data: StopVizInput, ctx: ToolContext):
    """
    Stop a running visualization server by name.
    
    Parameters:
    - server_name: Name of the server to stop (e.g., "stroke_viz", "teleop_viz", "map_viz")
    
    Returns information about whether the server was running and successfully stopped.
    """
    yield {"progress": 0.3, "message": f"Attempting to stop {input_data.server_name}..."}
    
    was_running = stop_server(input_data.server_name)
    
    if was_running:
        yield {"progress": 1.0, "message": f"Successfully stopped {input_data.server_name}"}
        yield StopVizOutput(
            success=True,
            message=f"✅ Stopped visualization server: {input_data.server_name}",
            server_name=input_data.server_name,
            was_running=True,
        )
    else:
        yield {"progress": 1.0, "message": f"Server {input_data.server_name} was not running"}
        yield StopVizOutput(
            success=False,
            message=f"❌ Server {input_data.server_name} was not running",
            server_name=input_data.server_name,
            was_running=False,
        )


@tool(
    name="list_viz_servers",
    nodes=["oop", "ook", "eek"],
    description="List all running visualization servers",
    input_model=ListVizServersInput,
    output_model=ListVizServersOutput,
    requires=["viz"],
)
async def list_viz_servers_tool(input_data: ListVizServersInput, ctx: ToolContext):
    """
    List all currently running visualization servers.
    
    No parameters required. Returns a list of server names that are currently active.
    """
    yield {"progress": 0.5, "message": "Checking for running servers..."}
    
    servers = list_servers()
    
    yield {"progress": 1.0, "message": f"Found {len(servers)} running servers"}
    
    yield ListVizServersOutput(
        success=True,
        message=f"Found {len(servers)} running visualization servers",
        servers=servers,
        count=len(servers),
    )


@tool(
    name="status_viz_server",
    nodes=["oop", "ook", "eek"],
    description="Get detailed status information for a visualization server",
    input_model=StatusVizInput,
    output_model=StatusVizOutput,
    requires=["viz"],
)
async def status_viz_server(input_data: StatusVizInput, ctx: ToolContext):
    """
    Get detailed status information for a visualization server.
    
    Parameters:
    - server_name: Name of the server to check (e.g., "stroke_viz", "teleop_viz", "map_viz")
    
    Returns detailed information including URL, host, port, thread status, etc.
    """
    yield {"progress": 0.5, "message": f"Checking status of {input_data.server_name}..."}
    
    status = get_server_status(input_data.server_name, ctx.node_name)
    
    if status["running"]:
        message = f"✅ Server {input_data.server_name} is running at {status['server_url']}"
    else:
        message = f"❌ Server {input_data.server_name} is not running"
    
    yield {"progress": 1.0, "message": message}
    
    yield StatusVizOutput(
        success=True,
        message=message,
        **status
    )