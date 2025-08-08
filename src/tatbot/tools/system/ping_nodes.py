"""Ping nodes tool for connectivity testing."""

import concurrent.futures

from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.system.models import PingNodesInput, PingNodesOutput
from tatbot.utils.log import get_logger
from tatbot.utils.net import NetworkManager

log = get_logger("tools.ping_nodes", "🏓")


@tool(
    name="ping_nodes",
    nodes=["rpi1"],
    description="Ping nodes and report connectivity status",
    input_model=PingNodesInput,
    output_model=PingNodesOutput,
)
async def ping_nodes(input_data: PingNodesInput, ctx: ToolContext):
    """
    Ping nodes and report connectivity status.
    
    Parameters:
    - nodes (list, optional): List of node names to ping. If not provided, pings all nodes.
    
    Example usage:
    {"nodes": ["ook", "trossen-ai"]}
    {}
    """
    yield {"progress": 0.1, "message": f"Pinging nodes: {input_data.nodes or 'all'}"}
    
    log.info(f"🔌 Pinging nodes: {input_data.nodes or 'all'}")
    
    try:
        log.info("Creating NetworkManager...")
        net = NetworkManager()
        log.info(f"NetworkManager created, loaded {len(net.nodes)} nodes")
        
        target_nodes, error = net.get_target_nodes(input_data.nodes)
        log.info(f"get_target_nodes returned: {len(target_nodes)} nodes, error: {error}")
        
        if error:
            yield PingNodesOutput(
                success=False,
                message=error,
                details=[],
                all_success=False
            )
            return
        
        if not target_nodes:
            yield PingNodesOutput(
                success=True,
                message="No nodes to ping.",
                details=[],
                all_success=True
            )
            return

        yield {"progress": 0.3, "message": f"Testing connectivity to {len(target_nodes)} nodes..."}

        messages = []
        all_success = True

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_node = {
                executor.submit(net._test_node_connection, node): node 
                for node in target_nodes
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_node):
                _, success, message = future.result()
                messages.append(message)
                if not success:
                    all_success = False
                
                completed += 1
                progress = 0.3 + (completed / len(target_nodes)) * 0.6
                yield {"progress": progress, "message": f"Tested {completed}/{len(target_nodes)} nodes"}

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

        yield PingNodesOutput(
            success=True,
            message=header,
            details=sorted(messages),
            all_success=all_success
        )
        
    except Exception as e:
        log.error(f"Error pinging nodes: {e}")
        yield PingNodesOutput(
            success=False,
            message=f"❌ Error pinging nodes: {str(e)}",
            details=[],
            all_success=False
        )