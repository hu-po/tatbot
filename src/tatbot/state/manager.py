"""StateManager - Singleton class for managing distributed tatbot state."""

import socket
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

from tatbot.utils.log import get_logger

from .client import RedisClient
from .models import (
    NodeHealth,
    RobotState,
    StrokeProgress,
)
from .schemas import RedisKeySchema, TTLConstants

log = get_logger("state.manager", "🎯")


class StateManager:
    """Singleton state manager for distributed tatbot operations."""
    
    _instance: Optional["StateManager"] = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs) -> "StateManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        redis_host: str = "eek",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        node_id: Optional[str] = None,
    ):
        # Prevent re-initialization
        if self._initialized:
            return
            
        self.node_id = node_id or socket.gethostname()
        self.redis = RedisClient(
            host=redis_host,
            port=redis_port, 
            password=redis_password
        )
        
        self._session_id: Optional[str] = None
        self._initialized = True
        
        log.info(f"🎯 StateManager initialized for node: {self.node_id}")
    
    async def connect(self) -> None:
        """Connect to Redis server."""
        await self.redis.connect()
        
        # Send initial heartbeat
        await self.update_node_health()
        
    async def disconnect(self) -> None:
        """Disconnect from Redis server."""
        await self.redis.disconnect()
    
    # Robot State Management
    async def update_robot_state(self, robot_state: RobotState) -> None:
        """Update robot state in Redis."""
        key = RedisKeySchema.robot_state_key(self.node_id)
        await self.redis.hset(key, "state", robot_state.model_dump_json())
        await self.redis.hset(key, "last_updated", datetime.utcnow().isoformat())
        
        # Publish event
        await self.redis.publish(
            RedisKeySchema.ROBOT_EVENTS,
            {
                "type": "state_update",
                "node_id": self.node_id,
                "timestamp": datetime.utcnow().isoformat(),
                "is_connected_l": robot_state.is_connected_l,
                "is_connected_r": robot_state.is_connected_r,
                "current_pose": robot_state.current_pose,
            }
        )
        
        log.debug(f"Updated robot state for {self.node_id}")
    
    async def get_robot_state(self, node_id: Optional[str] = None) -> Optional[RobotState]:
        """Get robot state from Redis."""
        key = RedisKeySchema.robot_state_key(node_id or self.node_id)
        state_data = await self.redis.hget(key, "state")
        
        if state_data:
            return RobotState.model_validate_json(state_data)
        return None
    
    # Stroke Progress Management
    async def start_stroke_session(self, total_strokes: int, stroke_length: int, scene_name: str = "default") -> str:
        """Start a new stroke execution session."""
        session_id = f"stroke_{self.node_id}_{int(datetime.utcnow().timestamp())}"
        self._session_id = session_id
        
        progress = StrokeProgress(
            node_id=self.node_id,
            total_strokes=total_strokes,
            stroke_length=stroke_length,
            scene_name=scene_name,
            is_executing=True,
        )
        
        key = RedisKeySchema.stroke_progress_key(session_id)
        await self.redis.hset(key, "progress", progress.model_dump_json())
        
        # Publish session start event
        await self.redis.publish(
            RedisKeySchema.stroke_events_channel("session"),
            {
                "type": "session_start",
                "session_id": session_id,
                "node_id": self.node_id,
                "total_strokes": total_strokes,
                "scene_name": scene_name,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        
        log.info(f"Started stroke session: {session_id}")
        return session_id
    
    async def update_stroke_progress(
        self,
        stroke_idx: int,
        pose_idx: int = 0,
        stroke_description_l: str = "",
        stroke_description_r: str = "",
        offset_idx_l: int = 0,
        offset_idx_r: int = 0,
        session_id: Optional[str] = None,
    ) -> None:
        """Update stroke execution progress."""
        session_id = session_id or self._session_id
        if not session_id:
            raise ValueError("No active stroke session. Call start_stroke_session() first.")
        
        # Get current progress
        current_progress = await self.get_stroke_progress(session_id)
        if not current_progress:
            raise ValueError(f"Stroke session {session_id} not found")
        
        # Update progress
        current_progress.stroke_idx = stroke_idx
        current_progress.pose_idx = pose_idx
        current_progress.stroke_description_l = stroke_description_l
        current_progress.stroke_description_r = stroke_description_r
        current_progress.offset_idx_l = offset_idx_l
        current_progress.offset_idx_r = offset_idx_r
        current_progress.timestamp = datetime.utcnow()
        
        key = RedisKeySchema.stroke_progress_key(session_id)
        await self.redis.hset(key, "progress", current_progress.model_dump_json())
        
        # Add to stroke history stream
        stream_key = RedisKeySchema.stroke_history_stream(session_id)
        await self.redis.xadd(
            stream_key,
            {
                "stroke_idx": stroke_idx,
                "pose_idx": pose_idx,
                "description_l": stroke_description_l,
                "description_r": stroke_description_r,
                "node_id": self.node_id,
            },
            maxlen=TTLConstants.STROKE_HISTORY_MAXLEN,
        )
        
        # Publish progress event
        await self.redis.publish(
            RedisKeySchema.stroke_events_channel("progress"),
            {
                "type": "progress_update",
                "session_id": session_id,
                "node_id": self.node_id,
                "stroke_idx": stroke_idx,
                "pose_idx": pose_idx,
                "total_strokes": current_progress.total_strokes,
                "stroke_length": current_progress.stroke_length,
                "description_l": stroke_description_l,
                "description_r": stroke_description_r,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        
        log.debug(f"Updated stroke progress: {stroke_idx}/{current_progress.total_strokes}")
    
    async def get_stroke_progress(self, session_id: Optional[str] = None) -> Optional[StrokeProgress]:
        """Get current stroke progress."""
        session_id = session_id or self._session_id
        if not session_id:
            return None
            
        key = RedisKeySchema.stroke_progress_key(session_id)
        progress_data = await self.redis.hget(key, "progress")
        
        if progress_data:
            return StrokeProgress.model_validate_json(progress_data)
        return None
    
    async def end_stroke_session(self, session_id: Optional[str] = None) -> None:
        """End stroke execution session."""
        session_id = session_id or self._session_id
        if not session_id:
            return
        
        # Update progress to mark as completed
        current_progress = await self.get_stroke_progress(session_id)
        if current_progress:
            current_progress.is_executing = False
            current_progress.timestamp = datetime.utcnow()
            
            key = RedisKeySchema.stroke_progress_key(session_id)
            await self.redis.hset(key, "progress", current_progress.model_dump_json())
        
        # Publish session end event
        await self.redis.publish(
            RedisKeySchema.stroke_events_channel("session"),
            {
                "type": "session_end",
                "session_id": session_id,
                "node_id": self.node_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        
        if session_id == self._session_id:
            self._session_id = None
            
        log.info(f"Ended stroke session: {session_id}")
    
    # Node Health Management
    async def update_node_health(self, health: Optional[NodeHealth] = None) -> None:
        """Update node health status."""
        if health is None:
            # Create basic health status
            health = NodeHealth(node_id=self.node_id)
            
            # Add basic connectivity check
            try:
                health.is_reachable = await self.redis.ping()
            except Exception:
                health.is_reachable = False
        
        key = RedisKeySchema.node_health_key(self.node_id)
        await self.redis.hset(key, "health", health.model_dump_json())
        await self.redis.expire(key, TTLConstants.NODE_HEALTH_TTL)
        
        log.debug(f"Updated node health for {self.node_id}")
    
    async def get_node_health(self, node_id: Optional[str] = None) -> Optional[NodeHealth]:
        """Get node health status."""
        key = RedisKeySchema.node_health_key(node_id or self.node_id)
        health_data = await self.redis.hget(key, "health")
        
        if health_data:
            return NodeHealth.model_validate_json(health_data)
        return None
    
    async def get_all_nodes_health(self) -> Dict[str, Optional[NodeHealth]]:
        """Get health status for all nodes."""
        pattern = f"{RedisKeySchema.NODE_HEALTH}:*"
        keys = await self.redis.keys(pattern)
        
        results = {}
        for key in keys:
            node_id = key.split(":", 2)[-1]  # Extract node_id from key
            health = await self.get_node_health(node_id)
            results[node_id] = health
            
        return results
    
    # Pub/Sub Operations
    async def publish_event(self, channel: str, event_data: Dict[str, Any]) -> None:
        """Publish event to channel."""
        event_data["node_id"] = self.node_id
        event_data["timestamp"] = datetime.utcnow().isoformat()
        
        await self.redis.publish(channel, event_data)
        log.debug(f"Published event to {channel}")
    
    async def subscribe_events(self, *channels: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to event channels."""
        async for message in self.redis.subscribe(*channels):
            yield message
    
    # Error Reporting
    async def report_error(self, error_category: str, error_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Report error to error events channel."""
        error_event = {
            "type": "error",
            "category": error_category,
            "message": error_message,
            "details": details or {},
            "node_id": self.node_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        await self.redis.publish(RedisKeySchema.error_events_channel(error_category), error_event)
        log.error(f"Reported {error_category} error: {error_message}")
    
    # Utility Methods
    async def clear_session_data(self, session_id: str) -> None:
        """Clear all data for a specific session."""
        keys_to_delete = [
            RedisKeySchema.stroke_progress_key(session_id),
            RedisKeySchema.stroke_history_stream(session_id),
        ]
        
        existing_keys = []
        for key in keys_to_delete:
            if await self.redis.exists(key):
                existing_keys.append(key)
        
        if existing_keys:
            await self.redis.delete(*existing_keys)
            log.info(f"Cleared session data for {session_id}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status summary."""
        # Get all node health
        nodes_health = await self.get_all_nodes_health()
        
        # Get active stroke sessions
        stroke_keys = await self.redis.keys(f"{RedisKeySchema.STROKE_PROGRESS}:*")
        active_sessions = len(stroke_keys)
        
        # Check Redis connection
        redis_connected = await self.redis.ping()
        
        return {
            "redis_connected": redis_connected,
            "active_stroke_sessions": active_sessions,
            "nodes_online": sum(1 for health in nodes_health.values() if health and health.is_reachable),
            "total_nodes": len(nodes_health),
            "nodes_health": {node_id: health.model_dump() if health else None for node_id, health in nodes_health.items()},
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # Context manager support
    async def __aenter__(self) -> "StateManager":
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()