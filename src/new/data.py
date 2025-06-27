# shared dataclasses for all modules, no jaxdc

from dataclasses import dataclass

@dataclass
class BotConfig:
    urdf_path: str = "~/tatbot/assets/urdf/tatbot.urdf"
    """Local path to the URDF file for the robot."""

    ee_link_names: tuple[str, str] = ("left/tattoo_needle", "right/tattoo_needle")
    """Names of the ee links in the URDF for left and right ik solving."""

    rest_pose: tuple[float, ...] = (
        -1.0, 0.1, 0.5, -1.2, 0.0, 0.0, 0.0, 0.0, # left arm
        1.0, 0.1, 0.5, -1.2, 0.0, 0.0, 0.0, 0.0, # right arm
    )
    """Rest pose for the robot."""

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
class CameraExtrinsics:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)