from dataclasses import dataclass
from typing import Union

from tatbot.data import Yaml
from tatbot.data.pose import Pose


@dataclass
class Intrinsics(Yaml):
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
    fov_depth: float | None = None
    """Field of view in radians for depth camera."""
    aspect_depth: float | None = None
    """Aspect ratio for depth camera."""
    
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
    intrinsics: Intrinsics
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
    realsenses: list[RealSenseCameraConfig]
    """List of camera configurations."""
    ipcameras: list[IPCameraConfig]
    """List of camera configurations."""
    yaml_dir: str = "~/tatbot/config/cams"
    """Directory containing the config yaml files."""

    def get_camera(self, camera_name: str) -> CameraConfig:
        camera_idx = int(camera_name[-1]) - 1
        if "realsense" in camera_name:
            return self.realsenses[camera_idx]
        else:
            return self.ipcameras[camera_idx]

    def set_camera(self, camera_name: str, camera_config: CameraConfig):
        camera_idx = int(camera_name[-1]) - 1
        if "realsense" in camera_name:
            self.realsenses[camera_idx] = camera_config
        else:
            self.ipcameras[camera_idx] = camera_config