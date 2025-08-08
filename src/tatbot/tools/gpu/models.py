"""Pydantic models for GPU tools."""

from tatbot.tools.base import ToolInput, ToolOutput


class ConvertStrokesInput(ToolInput):
    """Input for convert_strokes tool."""
    strokes_file_path: str
    strokebatch_file_path: str
    scene_name: str
    first_last_rest: bool = True
    use_ee_offsets: bool = True


class ConvertStrokesOutput(ToolOutput):
    """Output for convert_strokes tool."""
    strokebatch_base64: str = ""