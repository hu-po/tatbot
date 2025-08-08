"""Pydantic models for robot operation tools."""

from typing import List

from tatbot.tools.base import ToolInput, ToolOutput


class RobotOpInput(ToolInput):
    """Base input for robot operation tools."""
    scene_name: str = "default"


class RobotOpOutput(ToolOutput):
    """Base output for robot operation tools."""
    pass


class AlignInput(RobotOpInput):
    """Input for align tool."""
    pass


class AlignOutput(RobotOpOutput):
    """Output for align tool."""
    stroke_count: int = 0


class StrokeInput(RobotOpInput):
    """Input for stroke tool."""
    enable_joystick: bool = False
    enable_realsense: bool = False
    resume: bool = False
    fps: int = 10


class StrokeOutput(RobotOpOutput):
    """Output for stroke tool."""
    stroke_count: int = 0


class SenseInput(RobotOpInput):
    """Input for sense tool."""
    num_plys: int = 2
    calibrate_extrinsics: bool = True
    reference_tag_id: int = 0
    max_deviation_warning: float = 0.05


class SenseOutput(RobotOpOutput):
    """Output for sense tool."""
    captured_files: List[str] = []


class ResetInput(RobotOpInput):
    """Input for reset tool."""
    pass


class ResetOutput(RobotOpOutput):
    """Output for reset tool."""
    pass