"""Pydantic models for state management tools."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GetStateInput(BaseModel):
    """Input for get_state tool."""
    
    key: str = Field(..., description="State key to retrieve")
    node_id: Optional[str] = Field(default=None, description="Node ID (optional)")


class GetStateOutput(BaseModel):
    """Output for get_state tool."""
    
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(..., description="Status message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Retrieved state data")


class SetStateInput(BaseModel):
    """Input for set_state tool."""
    
    key: str = Field(..., description="State key to set")
    data: Dict[str, Any] = Field(..., description="State data to store")
    ttl: Optional[int] = Field(default=None, description="Time-to-live in seconds (optional)")


class SetStateOutput(BaseModel):
    """Output for set_state tool."""
    
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(..., description="Status message")


class SubscribeStateInput(BaseModel):
    """Input for subscribe_state tool."""
    
    channels: List[str] = Field(..., description="Channels to subscribe to")
    timeout: Optional[int] = Field(default=30, description="Subscription timeout in seconds")


class SubscribeStateOutput(BaseModel):
    """Output for subscribe_state tool."""
    
    success: bool = Field(..., description="Whether subscription succeeded")
    message: str = Field(..., description="Status message")
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Received events")


class PublishEventInput(BaseModel):
    """Input for publish_event tool."""
    
    channel: str = Field(..., description="Channel to publish to")
    event_data: Dict[str, Any] = Field(..., description="Event data to publish")


class PublishEventOutput(BaseModel):
    """Output for publish_event tool."""
    
    success: bool = Field(..., description="Whether publish succeeded")
    message: str = Field(..., description="Status message")
    subscribers: int = Field(default=0, description="Number of subscribers notified")


class SystemStatusInput(BaseModel):
    """Input for system_status tool."""
    
    detailed: bool = Field(default=False, description="Include detailed node information")


class SystemStatusOutput(BaseModel):
    """Output for system_status tool."""
    
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(..., description="Status message")
    status: Optional[Dict[str, Any]] = Field(default=None, description="System status data")