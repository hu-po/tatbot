"""Pydantic models for TUI monitoring tools."""

from typing import Optional

from pydantic import BaseModel, Field


class StartTUIMonitorInput(BaseModel):
    """Input for starting TUI system monitor."""
    
    background: bool = Field(
        default=False,
        description="Run monitor in background (detached)"
    )
    redis_host: str = Field(
        default="eek",
        description="Redis server host (default: eek, auto-falls back to 192.168.1.97)"
    )
    no_active_health_check: bool = Field(
        default=False,
        description="Disable active node health checking (only use Redis data)"
    )


class StartTUIMonitorOutput(BaseModel):
    """Output for starting TUI system monitor."""
    
    success: bool = Field(description="Whether the monitor started successfully")
    message: str = Field(description="Status message")
    process_id: Optional[int] = Field(default=None, description="Process ID if running in background")


class StopTUIMonitorInput(BaseModel):
    """Input for stopping TUI system monitor."""
    
    process_id: Optional[int] = Field(
        default=None,
        description="Specific process ID to stop (if None, stops all)"
    )


class StopTUIMonitorOutput(BaseModel):
    """Output for stopping TUI system monitor."""
    
    success: bool = Field(description="Whether the monitor stopped successfully")
    message: str = Field(description="Status message")
    stopped_processes: int = Field(description="Number of processes stopped")


class ListTUIMonitorsInput(BaseModel):
    """Input for listing running TUI monitors."""
    pass


class ListTUIMonitorsOutput(BaseModel):
    """Output for listing running TUI monitors."""
    
    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Status message")
    monitors: list = Field(default_factory=list, description="List of running monitor processes")