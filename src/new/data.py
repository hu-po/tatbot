# shared dataclasses for all modules, no jaxdc

from dataclasses import dataclass, field

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
class Node:
    name: str
    """Name of the node (a computer within the tatbot network)."""
    ip: str
    """IP address of the node, used for SSH connection."""
    user: str
    """Username for SSH connection."""
    emoji: str = "🌐"
    """Emoji to use for logging."""
    deps: str = "."
    """Dependencies to install on the node, see pyproject.toml."""

@dataclass
class InkCap:
    """Individual cylindrical inkcap."""
    urdf_link_name: str = "inkcap_large"
    """URDF link name of the inkcap."""
    diameter_m: float = 0.008
    """Diameter of the inkcap (meters)."""
    depth_m: float = 0.01
    """Depth of the inkcap (meters)."""
    color: str = "black"
    """Natural language description of the color of the ink inside the inkcap."""

@dataclass
class InkPalette:
    urdf_link_name: str = "inkpalette"
    """URDF link name of the inkpalette."""
    inkcaps: dict[str, InkCap] = field(default_factory=lambda: {
        "small_1": InkCap(
            urdf_link_name="inkcap_small_1",
            color="pink"
        ),
        "large": InkCap(
            urdf_link_name="inkcap_large",
            diameter_m=0.014,
            depth_m=0.014,
            color="black"
        ),
        "small_2": InkCap(
            urdf_link_name="inkcap_small_2",
            color="blue"
        ),
        "small_3": InkCap(
            urdf_link_name="inkcap_small_3",
            color="white"
        ),
        "medium_1": InkCap(
            urdf_link_name="inkcap_medium_1",
            diameter_m=0.012,
            color="red"
        ),
        "medium_2": InkCap(
            urdf_link_name="inkcap_medium_2",
            diameter_m=0.012,
            color="green"
        ),
    })

@dataclass
class TagConfig:
    family: str = "tag16h5"
    """Family of AprilTags to use."""
    size_m: float = 0.041
    """Size of AprilTags: distance between detection corners (meters)."""
    enabled_tags: dict[int, str] = field(default_factory=lambda: {
        6: "arm_l",
        7: "arm_r",
        9: "palette",
        10: "origin",
        11: "skin",
    })
    """ Dictionary of enabled AprilTag IDs."""
    urdf_link_names: dict[int, str] = field(default_factory=lambda: {
        6: "tag6",
        7: "tag7",
        9: "tag9",
        10: "tag10",
        11: "tag11",
    })
    """ Dictionary of AprilTag IDs to URDF link names."""
    decision_margin: float = 20.0
    """Minimum decision margin for AprilTag detection filtering."""

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

@dataclass
class Scan:
    name: str = "scan"
    """Name of the scan."""

    bot_config: BotConfig = field(default_factory=BotConfig)
    """Bot configuration to use for the scan."""
    ink_config: InkConfig = field(default_factory=InkConfig)
    """Config containig InkCaps and palette position."""
    tag_config: TagConfig = field(default_factory=TagConfig)
    """Config containing AprilTag parameters."""

    optical_frame_urdf_link_names: dict[str, str] = field(default_factory=lambda: {
        "realsense1": "realsense1_color_optical_frame",
        "realsense2": "realsense2_color_optical_frame",
        "camera1": "camera1_optical_frame",
        "camera2": "camera2_optical_frame",
        "camera3": "camera3_optical_frame",
        "camera4": "camera4_optical_frame",
        "camera5": "camera5_optical_frame",
    })
    """URDF link names for each camera's optical frame."""

    tag_poses: dict[str, dict[int, TagPose]] = field(default_factory=dict)
    """Tag poses for each tag."""

    extrinsics: dict[str, CameraExtrinsics] = field(default_factory=lambda: {
        "realsense1": CameraExtrinsics(),
        "realsense2": CameraExtrinsics(),
        "camera1": CameraExtrinsics(),
        "camera2": CameraExtrinsics(),
        "camera3": CameraExtrinsics(),
        "camera4": CameraExtrinsics(),
        "camera5": CameraExtrinsics(),
    })
    """Extrinsics for each camera."""

    intrinsics: dict[str, CameraIntrinsics] = field(default_factory=lambda: {
        "realsense1": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "realsense2": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "camera1": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "camera2": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "camera3": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "camera4": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "camera5": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
    })
    """Intrinsics for each camera."""