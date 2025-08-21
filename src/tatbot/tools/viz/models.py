"""Pydantic models for visualization tools."""

from typing import List, Optional, Tuple

from tatbot.tools.base import ToolInput, ToolOutput


class BaseVizInput(ToolInput):
    """Base input for visualization tools."""
    scene: str = "default"
    env_map_hdri: str = "forest"
    view_camera_position: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    view_camera_look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    enable_robot: bool = False
    enable_depth: bool = False
    speed: float = 1.0
    fps: float = 30.0
    bind_host: str = "0.0.0.0"


class BaseVizOutput(ToolOutput):
    """Base output for visualization tools."""
    server_url: Optional[str] = None
    server_name: str
    running: bool


class StrokeVizInput(BaseVizInput):
    """Input for stroke visualization tool."""
    align: bool = False
    enable_state_sync: bool = False
    design_pointcloud_point_size: float = 0.001
    design_pointcloud_point_shape: str = "rounded"
    path_highlight_radius: int = 3
    pose_highlight_radius: int = 6


class StrokeVizOutput(BaseVizOutput):
    """Output for stroke visualization tool."""
    num_strokes: int
    stroke_length: int


class TeleopVizInput(BaseVizInput):
    """Input for teleoperation visualization tool."""
    transform_control_scale: float = 0.2
    transform_control_opacity: float = 0.2


class TeleopVizOutput(BaseVizOutput):
    """Output for teleoperation visualization tool."""
    ee_links: List[str]


class MapVizInput(BaseVizInput):
    """Input for surface mapping visualization tool."""
    stroke_point_size: float = 0.0005
    stroke_point_shape: str = "rounded"
    skin_ply_point_size: float = 0.0005
    skin_ply_point_shape: str = "rounded"
    transform_control_scale: float = 0.1
    transform_control_opacity: float = 0.8


class MapVizOutput(BaseVizOutput):
    """Output for surface mapping visualization tool."""
    num_strokes: int
    ply_files_count: int


class StopVizInput(ToolInput):
    """Input for stopping a visualization server."""
    server_name: str


class StopVizOutput(ToolOutput):
    """Output for stopping a visualization server."""
    server_name: str
    was_running: bool


class ListVizServersInput(ToolInput):
    """Input for listing visualization servers."""
    pass


class ListVizServersOutput(ToolOutput):
    """Output for listing visualization servers."""
    servers: List[str]
    count: int


class StatusVizInput(ToolInput):
    """Input for getting detailed status of a visualization server."""
    server_name: str


class StatusVizOutput(ToolOutput):
    """Output for visualization server status."""
    server_name: str
    running: bool
    server_url: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    thread_alive: Optional[bool] = None
    started_at: Optional[str] = None


class VGGTCompareVizInput(BaseVizInput):
    """Input for VGGT vs RealSense comparison viz."""
    dataset_dir: str
    vggt_pointcloud_point_size: float = 0.001
    rs_pointcloud_point_size: float = 0.001


class VGGTCompareVizOutput(BaseVizOutput):
    """Output for VGGT comparison viz tool."""
    pass
