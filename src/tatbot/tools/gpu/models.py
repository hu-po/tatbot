"""Pydantic models for GPU tools."""

from tatbot.tools.base import ToolInput, ToolOutput


class ConvertStrokesInput(ToolInput):
    """Input for convert_strokes tool."""
    strokes_file_path: str
    strokebatch_file_path: str
    scene: str
    first_last_rest: bool = True
    use_ee_offsets: bool = True


class ConvertStrokesOutput(ToolOutput):
    """Output for convert_strokes tool."""
    strokebatch_base64: str = ""


class VGGTReconInput(ToolInput):
    """Input for VGGT reconstruction GPU tool."""
    image_dir: str
    output_pointcloud_path: str
    output_frustums_path: str
    output_colmap_dir: str
    scene: str
    meta: str | None = None
    weights_path: str | None = None
    vggt_conf_threshold: float = 0.0
    shared_camera: bool = False


class VGGTReconOutput(ToolOutput):
    """Output for VGGT reconstruction GPU tool."""
    point_count: int = 0
    frustum_count: int = 0
