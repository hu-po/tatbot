"""Acquired DBV3 paths -> tatbot program (ros/README.md section 4.2). Pure Python: numpy, pyyaml.

Units are metres and seconds; points are page-frame (x, y) with the origin at the page centre,
x along stencil u (right), y toward the top of the print.
"""
from __future__ import annotations

from tatbot_ink.errors import CompileError
from tatbot_ink.preview import write_preview
from tatbot_ink.program import DEFAULT_SPEED_M_S, FORMAT, MAX_SEGMENT_S, VERSION, compile, write_program

__all__ = ["CompileError", "DEFAULT_SPEED_M_S", "FORMAT", "MAX_SEGMENT_S", "VERSION", "compile",
           "write_preview", "write_program"]
