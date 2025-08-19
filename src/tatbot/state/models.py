"""Pydantic models for tatbot distributed state management."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class BaseStateModel(BaseModel):
    """Base class for all state models with timestamp."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    node_id: str = Field(..., description="ID of the node that created this state")


class RobotState(BaseStateModel):
    """Complete robot state including both arms."""
    
    # Connection status
    is_connected_l: bool = Field(default=False, description="Left arm connection status")
    is_connected_r: bool = Field(default=False, description="Right arm connection status")
    
    # Joint positions (serialize as simple lists for compatibility)
    joints_l: Optional[List[float]] = Field(
        default=None, description="Left arm joint positions (radians)"
    )
    joints_r: Optional[List[float]] = Field(
        default=None, description="Right arm joint positions (radians)"
    )
    
    # Joint velocities
    velocities_l: Optional[List[float]] = Field(
        default=None, description="Left arm joint velocities (rad/s)"
    )
    velocities_r: Optional[List[float]] = Field(
        default=None, description="Right arm joint velocities (rad/s)"
    )
    
    # Current pose
    current_pose: str = Field(default="unknown", description="Current robot pose (ready, sleep, etc.)")
    
    # Goal time settings
    goal_time_slow: float = Field(default=2.0, description="Slow movement goal time")
    goal_time_fast: float = Field(default=0.5, description="Fast movement goal time")

    # No special numpy config required (we serialize as lists)


class StrokeProgress(BaseStateModel):
    """Progress tracking for stroke execution."""
    
    # Current execution state
    stroke_idx: int = Field(default=0, description="Current stroke index")
    pose_idx: int = Field(default=0, description="Current pose within stroke")
    total_strokes: int = Field(default=0, description="Total number of strokes")
    stroke_length: int = Field(default=0, description="Number of poses per stroke")
    
    # Descriptions
    stroke_description_l: str = Field(default="", description="Left arm stroke description")
    stroke_description_r: str = Field(default="", description="Right arm stroke description")
    
    # Offset indices for needle depth control
    offset_idx_l: int = Field(default=0, description="Left arm offset index")
    offset_idx_r: int = Field(default=0, description="Right arm offset index")
    
    # Execution status
    is_executing: bool = Field(default=False, description="Whether stroke execution is active")
    is_paused: bool = Field(default=False, description="Whether execution is paused")
    
    # Scene information
    scene_name: str = Field(default="default", description="Current scene configuration")
    dataset_name: Optional[str] = Field(default=None, description="Recording dataset name")


class NodeHealth(BaseStateModel):
    """Health status of individual nodes."""
    
    # System metrics
    cpu_percent: float = Field(default=0.0, description="CPU usage percentage")
    memory_percent: float = Field(default=0.0, description="Memory usage percentage") 
    disk_percent: float = Field(default=0.0, description="Disk usage percentage")
    
    # GPU status (if available)
    gpu_available: bool = Field(default=False, description="Whether GPU is available")
    gpu_percent: Optional[float] = Field(default=None, description="GPU usage percentage")
    gpu_memory_percent: Optional[float] = Field(default=None, description="GPU memory usage")
    
    # Network connectivity
    is_reachable: bool = Field(default=True, description="Whether node is network reachable")
    ping_latency_ms: Optional[float] = Field(default=None, description="Network ping latency")
    
    # Service status
    mcp_server_running: bool = Field(default=False, description="MCP server status")
    mcp_port: Optional[int] = Field(default=None, description="MCP server port")
    
    # Last heartbeat
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)



