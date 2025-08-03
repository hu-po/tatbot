from pydantic import field_validator
from typing import List, Optional
import ipaddress

from tatbot.data.base import BaseCfg
from tatbot.data.pose import Pose

class Intrinsics(BaseCfg):
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
    fov_depth: Optional[float] = None
    """Field of view in radians for depth camera."""
    aspect_depth: Optional[float] = None
    """Aspect ratio for depth camera."""

class CameraConfig(BaseCfg):
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
    urdf_link_name: str
    """Name of the link in the URDF that the camera is attached to."""

class RealSenseCameraConfig(CameraConfig):
    serial_number: str
    """Serial number of the camera (only for realsense cameras)."""

class IPCameraConfig(CameraConfig):
    ip: str
    """IP address of the camera (only for ip cameras)."""
    username: str
    """Username for the camera (only for ip cameras)."""
    password: str
    """Password for the camera (only for ip cameras)."""
    rtsp_port: int
    """RTSP port of the camera (only for ip cameras)."""

    @field_validator('ip')
    def validate_ip(cls, v):
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError(f"'{v}' is not a valid IP address")
        return v

class Cams(BaseCfg):
    realsenses: List[RealSenseCameraConfig]
    """List of camera configurations."""
    ipcameras: List[IPCameraConfig]
    """List of camera configurations."""

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
