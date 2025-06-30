import os
from dataclasses import dataclass

from . import Yaml
from .pose import Pose


@dataclass
class CameraIntrinsics:
    fov: float = 0.0
    """Field of view in radians."""
    aspect: float = 0.0
    """Aspect ratio."""
    fx: float = 0.0
    """Focal length in x-direction."""
    fy: float = 0.0
    """Focal length in y-direction."""
    ppx: float = 0.0
    """Principal point in x-direction."""
    ppy: float = 0.0
    """Principal point in y-direction."""
    
@dataclass
class CameraConfig(Yaml):
    """Configuration for a single camera."""
    ip: str
    username: str
    password: str
    rtsp_port: int
    stream_path: str
    resolution: tuple[int, int]
    intrinsics: CameraIntrinsics
    extrinsics: Pose
    fps: int
    yaml_dir: str = os.path.expanduser("~/tatbot/config/cameras")