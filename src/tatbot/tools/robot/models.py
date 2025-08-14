"""Pydantic models for robot operation tools."""

from typing import List

from pydantic import ConfigDict

from tatbot.tools.base import ToolInput, ToolOutput


class RobotToolInput(ToolInput):
    """Base input for robot operation tools."""
    model_config = ConfigDict(extra="ignore")
    scene: str = "default"


class RobotToolOutput(ToolOutput):
    """Base output for robot operation tools."""
    pass


class AlignInput(RobotToolInput):
    """Input for align tool."""
    pass


class AlignOutput(RobotToolOutput):
    """Output for align tool."""
    stroke_count: int = 0


class StrokeInput(RobotToolInput):
    """Input for stroke tool."""
    enable_joystick: bool = True
    enable_realsense: bool = True
    resume: bool = False
    fps: int = 10


class StrokeOutput(RobotToolOutput):
    """Output for stroke tool."""
    stroke_count: int = 0


class SenseInput(RobotToolInput):
    """Input for sense tool."""
    num_plys: int = 2
    calibrate_extrinsics: bool = True
    reference_tag_id: int = 0
    max_deviation_warning: float = 0.05


class SenseOutput(RobotToolOutput):
    """Output for sense tool."""
    captured_files: List[str] = []


class ResetInput(RobotToolInput):
    """Input for reset tool."""
    pass


class ResetOutput(RobotToolOutput):
    """Output for reset tool."""
    pass