"""State management tools for MCP integration."""

import asyncio
from typing import Dict

from tatbot.state.manager import StateManager
from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.state.models import (
    GetStateInput,
    GetStateOutput,
    PublishEventInput,
    PublishEventOutput,
    SetStateInput,
    SetStateOutput,
    SubscribeStateInput,
    SubscribeStateOutput,
    SystemStatusInput,
    SystemStatusOutput,
)
from tatbot.utils.log import get_logger

log = get_logger("tools.state", "📊")


@tool(
    name="get_state",
    nodes=["oop", "ook", "eek", "hog", "ojo", "rpi1", "rpi2"],
    description="Get state data from Redis parameter server",
    input_model=GetStateInput,
    output_model=GetStateOutput,
)
async def get_state_tool(input_data: GetStateInput, ctx: ToolContext):
    """
    Get state data from the distributed parameter server.
    
    This tool retrieves state information from Redis, including robot state,
    stroke progress, node health, and sensor data.
    
    Parameters:
    - key: State key to retrieve (e.g. 'robot:state', 'stroke:progress')
    - node_id: Optional node ID for node-specific state
    
    Returns:
    - success: Whether the operation succeeded
    - message: Status message
    - data: Retrieved state data (if successful)
    """
    yield {"progress": 0.1, "message": f"Connecting to state server..."}
    
    try:
        state_manager = StateManager(node_id=ctx.node_name)
        
        async with state_manager:
            yield {"progress": 0.3, "message": f"Retrieving state key: {input_data.key}"}
            
            # Handle special state keys
            if input_data.key == "stroke:progress":
                data = await state_manager.get_stroke_progress()
                if data:
                    yield GetStateOutput(
                        success=True,
                        message=f"✅ Retrieved stroke progress",
                        data=data.model_dump()
                    )
                else:
                    yield GetStateOutput(
                        success=False,
                        message="No active stroke session found"
                    )
                return
            
            elif input_data.key.startswith("robot:state"):
                node_id = input_data.node_id or ctx.node_name
                data = await state_manager.get_robot_state(node_id)
                if data:
                    yield GetStateOutput(
                        success=True,
                        message=f"✅ Retrieved robot state for {node_id}",
                        data=data.model_dump()
                    )
                else:
                    yield GetStateOutput(
                        success=False,
                        message=f"No robot state found for {node_id}"
                    )
                return
            
            elif input_data.key.startswith("node:health"):
                node_id = input_data.node_id or ctx.node_name
                data = await state_manager.get_node_health(node_id)
                if data:
                    yield GetStateOutput(
                        success=True,
                        message=f"✅ Retrieved node health for {node_id}",
                        data=data.model_dump()
                    )
                else:
                    yield GetStateOutput(
                        success=False,
                        message=f"No health data found for {node_id}"
                    )
                return
            
            # Generic key retrieval
            value = await state_manager.redis.get(input_data.key)
            if value:
                yield GetStateOutput(
                    success=True,
                    message=f"✅ Retrieved state for key: {input_data.key}",
                    data={"key": input_data.key, "value": value}
                )
            else:
                yield GetStateOutput(
                    success=False,
                    message=f"Key not found: {input_data.key}"
                )
                
    except Exception as e:
        log.error(f"Error getting state: {e}")
        yield GetStateOutput(
            success=False,
            message=f"❌ Failed to get state: {e}"
        )


@tool(
    name="set_state",
    nodes=["oop", "ook", "eek", "hog", "ojo", "rpi1", "rpi2"],
    description="Set state data in Redis parameter server",
    input_model=SetStateInput,
    output_model=SetStateOutput,
)
async def set_state_tool(input_data: SetStateInput, ctx: ToolContext):
    """
    Set state data in the distributed parameter server.
    
    This tool stores state information in Redis with optional expiration.
    
    Parameters:
    - key: State key to set
    - data: State data to store
    - ttl: Optional time-to-live in seconds
    
    Returns:
    - success: Whether the operation succeeded
    - message: Status message
    """
    yield {"progress": 0.1, "message": f"Connecting to state server..."}
    
    try:
        state_manager = StateManager(node_id=ctx.node_name)
        
        async with state_manager:
            yield {"progress": 0.3, "message": f"Setting state key: {input_data.key}"}
            
            success = await state_manager.redis.set(
                input_data.key, 
                input_data.data, 
                ex=input_data.ttl
            )
            
            if success:
                yield SetStateOutput(
                    success=True,
                    message=f"✅ Successfully set state for key: {input_data.key}"
                )
            else:
                yield SetStateOutput(
                    success=False,
                    message=f"Failed to set state for key: {input_data.key}"
                )
                
    except Exception as e:
        log.error(f"Error setting state: {e}")
        yield SetStateOutput(
            success=False,
            message=f"❌ Failed to set state: {e}"
        )


@tool(
    name="publish_event",
    nodes=["oop", "ook", "eek", "hog", "ojo", "rpi1", "rpi2"],
    description="Publish event to Redis pub/sub channel",
    input_model=PublishEventInput,
    output_model=PublishEventOutput,
)
async def publish_event_tool(input_data: PublishEventInput, ctx: ToolContext):
    """
    Publish event to Redis pub/sub channel.
    
    This tool publishes events to Redis channels for real-time communication
    between nodes.
    
    Parameters:
    - channel: Channel name to publish to
    - event_data: Event data to publish
    
    Returns:
    - success: Whether the publish succeeded
    - message: Status message
    - subscribers: Number of subscribers notified
    """
    yield {"progress": 0.1, "message": f"Connecting to state server..."}
    
    try:
        state_manager = StateManager(node_id=ctx.node_name)
        
        async with state_manager:
            yield {"progress": 0.3, "message": f"Publishing to channel: {input_data.channel}"}
            
            subscribers = await state_manager.publish_event(
                input_data.channel,
                input_data.event_data
            )
            
            yield PublishEventOutput(
                success=True,
                message=f"✅ Event published to {input_data.channel}",
                subscribers=subscribers
            )
                
    except Exception as e:
        log.error(f"Error publishing event: {e}")
        yield PublishEventOutput(
            success=False,
            message=f"❌ Failed to publish event: {e}"
        )


@tool(
    name="subscribe_events",
    nodes=["oop", "ook", "eek", "hog", "ojo", "rpi1", "rpi2"],
    description="Subscribe to Redis pub/sub channels for events",
    input_model=SubscribeStateInput,
    output_model=SubscribeStateOutput,
)
async def subscribe_events_tool(input_data: SubscribeStateInput, ctx: ToolContext):
    """
    Subscribe to Redis pub/sub channels for real-time events.
    
    This tool subscribes to event channels and returns received events
    within the specified timeout period.
    
    Parameters:
    - channels: List of channel names to subscribe to
    - timeout: Subscription timeout in seconds
    
    Returns:
    - success: Whether subscription succeeded
    - message: Status message  
    - events: List of received events
    """
    yield {"progress": 0.1, "message": f"Connecting to state server..."}
    
    try:
        state_manager = StateManager(node_id=ctx.node_name)
        events = []
        
        async with state_manager:
            yield {"progress": 0.3, "message": f"Subscribing to channels: {input_data.channels}"}
            
            try:
                # Subscribe with timeout
                async with asyncio.timeout(input_data.timeout):
                    async for message in state_manager.subscribe_events(*input_data.channels):
                        events.append(message)
                        yield {
                            "progress": 0.5,
                            "message": f"Received {len(events)} events..."
                        }
                        
                        # Limit to reasonable number of events
                        if len(events) >= 100:
                            break
                            
            except asyncio.TimeoutError:
                log.info(f"Subscription timeout after {input_data.timeout}s")
            
            yield SubscribeStateOutput(
                success=True,
                message=f"✅ Received {len(events)} events from {len(input_data.channels)} channels",
                events=events
            )
                
    except Exception as e:
        log.error(f"Error subscribing to events: {e}")
        yield SubscribeStateOutput(
            success=False,
            message=f"❌ Failed to subscribe to events: {e}"
        )


@tool(
    name="system_status",
    nodes=["oop", "ook", "eek", "hog", "ojo", "rpi1", "rpi2"],
    description="Get overall system status from parameter server",
    input_model=SystemStatusInput,
    output_model=SystemStatusOutput,
)
async def system_status_tool(input_data: SystemStatusInput, ctx: ToolContext):
    """
    Get overall system status from the distributed parameter server.
    
    This tool provides a comprehensive view of the system state including
    Redis connectivity, active sessions, and node health.
    
    Parameters:
    - detailed: Include detailed node information
    
    Returns:
    - success: Whether the operation succeeded
    - message: Status message
    - status: System status data
    """
    yield {"progress": 0.1, "message": f"Connecting to state server..."}
    
    try:
        state_manager = StateManager(node_id=ctx.node_name)
        
        async with state_manager:
            yield {"progress": 0.3, "message": "Gathering system status..."}
            
            status = await state_manager.get_system_status()
            
            # Add detailed node info if requested
            if input_data.detailed and status.get("nodes_health"):
                detailed_nodes = {}
                for node_id, health_data in status["nodes_health"].items():
                    if health_data:
                        detailed_nodes[node_id] = health_data
                    else:
                        detailed_nodes[node_id] = {"status": "offline"}
                status["detailed_nodes"] = detailed_nodes
            
            yield SystemStatusOutput(
                success=True,
                message=f"✅ System status retrieved - {status['nodes_online']}/{status['total_nodes']} nodes online",
                status=status
            )
                
    except Exception as e:
        log.error(f"Error getting system status: {e}")
        yield SystemStatusOutput(
            success=False,
            message=f"❌ Failed to get system status: {e}"
        )