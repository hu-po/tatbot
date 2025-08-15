from pathlib import Path
from typing import Tuple

from pydantic import field_validator

from tatbot.data.base import BaseCfg


class URDF(BaseCfg):
    path: Path
    """Path to the URDF file for the robot."""

    ee_link_names: Tuple[str, str]
    """Names of the ee (end effector) links in the URDF."""
    tag_link_names: Tuple[str, ...]
    """Names of the tag (apriltag) links in the URDF."""
    cam_link_names: Tuple[str, ...]
    """Names of the camera links in the URDF."""
    ink_link_names: Tuple[str, ...]
    """Names of the inkcap links in the URDF."""
    calibrator_link_name: str
    """Name of the calibrator link in the URDF."""
    lasercross_link_name: str
    """Name of the lasercross link in the URDF."""
    root_link_name: str
    """Name of the origin/root link in the URDF."""

    @field_validator('path', mode='before')
    def expand_user_path(cls, v: str) -> Path:
        return Path(v).expanduser()

    @field_validator('path')
    def path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v
