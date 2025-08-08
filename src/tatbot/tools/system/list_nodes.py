"""List nodes tool for discovering available tatbot nodes."""

from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.system.models import ListNodesInput, ListNodesOutput
from tatbot.utils.log import get_logger
from tatbot.utils.net import NetworkManager

log = get_logger("tools.list_nodes", "🔗")


@tool(
    name="list_nodes",
    nodes=["*"],
    description="List available tatbot nodes",
    input_model=ListNodesInput,
    output_model=ListNodesOutput,
)
async def list_nodes(input_data: ListNodesInput, ctx: ToolContext):
    """
    List available tatbot nodes.
    
    No parameters required. Returns list of available node names.
    
    Example usage:
    {}
    """
    yield {"progress": 0.1, "message": "Loading network configuration..."}
    
    try:
        net = NetworkManager()
        
        yield {"progress": 0.5, "message": "Discovering nodes..."}
        
        node_names = [node.name for node in net.nodes]
        log.info(f"Found {len(node_names)} nodes")
        
        yield ListNodesOutput(
            success=True,
            message=f"Found {len(node_names)} available nodes",
            nodes=node_names,
            count=len(node_names)
        )
        
    except Exception as e:
        log.error(f"Error listing nodes: {e}")
        yield ListNodesOutput(
            success=False,
            message=f"❌ Error listing nodes: {e}",
            nodes=[],
            count=0
        )