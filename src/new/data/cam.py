from dataclasses import dataclass

from .pose import Pose
from . import Yaml

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
class CameraConfig:
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

@dataclass
class CameraManagerConfig(Yaml):
    """Configuration for all cameras."""
    cameras: dict[str, CameraConfig]
    output_dir: str
    image_format: str
    image_quality: int