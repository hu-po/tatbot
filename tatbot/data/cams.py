from dataclasses import dataclass

from tatbot.data import Yaml
from tatbot.data.pose import Pose

@dataclass
class Instrinsics(Yaml):
    fov: float
    """Field of view in radians."""
    aspect: float
    """Aspect ratio."""
    fx: float
    """Focal length in x-direction."""
    fy: float
    """Focal length in y-direction."""
    ppx: float
    """Principal point in x-direction."""
    ppy: float
    """Principal point in y-direction."""
    
@dataclass
class CameraConfig(Yaml):
    name: str
    """Name of the camera."""
    width: int
    """Width of the image in pixels."""
    height: int
    """Height of the image in pixels."""
    fps: int
    """Frames per second."""
    intrinsics: Instrinsics
    """Intrinsics of the camera."""
    extrinsics: Pose
    """Extrinsics of the camera: pose in the global frame."""

@dataclass
class RealSenseCameraConfig(CameraConfig):
    serial_number: str
    """Serial number of the camera (only for realsense cameras)."""

@dataclass
class IPCameraConfig(CameraConfig):
    ip: str
    """IP address of the camera (only for ip cameras)."""
    username: str
    """Username for the camera (only for ip cameras)."""
    password: str
    """Password for the camera (only for ip cameras)."""
    rtsp_port: int
    """RTSP port of the camera (only for ip cameras)."""

@dataclass
class Cams(Yaml):
    realsenses: tuple[RealSenseCameraConfig, ...]
    """List of camera configurations."""
    ipcameras: tuple[IPCameraConfig, ...]
    """List of camera configurations."""
    yaml_dir: str = "~/tatbot/config/cams"
    """Directory containing the config yaml files."""