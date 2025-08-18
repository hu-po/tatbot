"""Redis key schemas and naming conventions for tatbot state."""

from typing import Optional


class RedisKeySchema:
    """Centralized Redis key naming schema for consistency."""
    
    # State prefixes
    ROBOT_STATE = "robot:state"
    STROKE_PROGRESS = "stroke:progress" 
    NODE_HEALTH = "node:health"
    
    # Pub/Sub channels
    STROKE_EVENTS = "stroke:events"
    ROBOT_EVENTS = "robot:events"
    NODE_EVENTS = "node:events"
    ERROR_EVENTS = "error:events"
    
    # Streams (for time-series data)
    STROKE_HISTORY = "stream:stroke_history"
    ROBOT_TELEMETRY = "stream:robot_telemetry" 
    NODE_METRICS = "stream:node_metrics"
    SENSOR_STREAM = "stream:sensor_data"
    
    @classmethod
    def robot_state_key(cls, node_id: Optional[str] = None) -> str:
        """Get robot state key, optionally for specific node."""
        if node_id:
            return f"{cls.ROBOT_STATE}:{node_id}"
        return cls.ROBOT_STATE
        
    @classmethod
    def stroke_progress_key(cls, session_id: Optional[str] = None) -> str:
        """Get stroke progress key, optionally for specific session."""
        if session_id:
            return f"{cls.STROKE_PROGRESS}:{session_id}"
        return cls.STROKE_PROGRESS
        
    @classmethod 
    def node_health_key(cls, node_id: str) -> str:
        """Get node health key for specific node."""
        return f"{cls.NODE_HEALTH}:{node_id}"
        
    @classmethod
    def stroke_events_channel(cls, event_type: Optional[str] = None) -> str:
        """Get stroke events channel, optionally for specific event type."""
        if event_type:
            return f"{cls.STROKE_EVENTS}:{event_type}"
        return cls.STROKE_EVENTS
        
    @classmethod
    def node_events_channel(cls, node_id: Optional[str] = None) -> str:
        """Get node events channel, optionally for specific node."""
        if node_id:
            return f"{cls.NODE_EVENTS}:{node_id}"
        return cls.NODE_EVENTS
        
    @classmethod
    def error_events_channel(cls, error_category: Optional[str] = None) -> str:
        """Get error events channel, optionally for specific category."""
        if error_category:
            return f"{cls.ERROR_EVENTS}:{error_category}"
        return cls.ERROR_EVENTS
        
    @classmethod
    def stroke_history_stream(cls, session_id: Optional[str] = None) -> str:
        """Get stroke history stream, optionally for specific session."""
        if session_id:
            return f"{cls.STROKE_HISTORY}:{session_id}"
        return cls.STROKE_HISTORY
        
    @classmethod
    def node_metrics_stream(cls, node_id: str) -> str:
        """Get node metrics stream for specific node."""
        return f"{cls.NODE_METRICS}:{node_id}"


# Common TTL values (in seconds)
class TTLConstants:
    """Time-to-live constants for different data types."""
    
    NODE_HEALTH_TTL = 60  # Node health expires after 1 minute
    ERROR_LOG_TTL = 86400  # Error logs kept for 24 hours
    
    # Stream max lengths
    STROKE_HISTORY_MAXLEN = 10000  # Keep last 10k stroke events
    TELEMETRY_MAXLEN = 5000       # Keep last 5k telemetry points