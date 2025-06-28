# shared dataclasses for all modules, no jaxdc

from dataclasses import dataclass, field

@dataclass
class URDF:
    path: str = "~/tatbot/assets/urdf/tatbot.urdf"
    """Path to the URDF file for the robot."""
    ee_link_names: tuple[str, str] = ("left/tattoo_needle", "right/tattoo_needle")
    """Names of the ee (end effector) links in the URDF."""
    tag_link_names: tuple[str, ...] = ("tag6", "tag7", "tag9", "tag10", "tag11")
    """Names of the tag (apriltag) links in the URDF."""
    cam_link_names: tuple[str, ...] = ("realsense1", "realsense2", "camera1", "camera2", "camera3", "camera4", "camera5")
    """Names of the camera links in the URDF."""
    ink_link_names: tuple[str, ...] = ("inkcap_large", "inkcap_small_1", "inkcap_small_2", "inkcap_small_3", "inkcap_medium_1", "inkcap_medium_2")
    """Names of the inkcap links in the URDF."""
    palette_link_name: str = "inkpalette"
    """Name of the inkpalette link in the URDF."""
    origin_link_name: str = "origin"
    """Name of the origin link in the URDF."""
    skin_link_name: str = "skin"
    """Name of the skin link in the URDF."""

@dataclass
class Scene:
    """Main configuration for the scene."""
    rest_pose: tuple[float, ...] = (
        -1.0, 0.1, 0.5, -1.2, 0.0, 0.0, 0.0, 0.0, # left arm
        1.0, 0.1, 0.5, -1.2, 0.0, 0.0, 0.0, 0.0, # right arm
    )
    """Rest pose for the robot."""
    urdf: URDF
    """URDF configuration for the robot."""

@dataclass
class Node:
    """Node in the tatbot network."""
    name: str
    """Name of the node."""
    ip: str
    """IP address of the node, used for SSH connection."""
    user: str
    """Username for SSH connection."""
    emoji: str = "🌐"
    """Emoji to use for logging."""
    deps: str = "."
    """Dependencies to install on the node, see pyproject.toml."""

@dataclass
class InkColor:
    name: str = "black"
    """Natural language description of the color of the ink inside the inkcap."""
    rgb: tuple[int, int, int] = (0, 0, 0)
    """RGB values of the color of the ink inside the inkcap."""

@dataclass
class InkCap:
    """Individual cylindrical inkcap."""
    urdf_link_name: str = "inkcap_large"
    """URDF link name of the inkcap."""
    diameter_m: float = 0.008
    """Diameter of the inkcap (meters)."""
    depth_m: float = 0.01
    """Depth of the inkcap (meters)."""
    color: InkColor = InkColor(name="black", rgb=(0, 0, 0))
    """Color of the ink inside the inkcap."""

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
class Pose:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position in meters (xyz)."""
    wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation quaternion (wxyz)."""

@dataclass
class Scene:
    name: str = "scene"
    """Name of the scene."""
    
    design_pos: np.ndarray = field(default_factory=lambda: np.array([0.14, 0.02, 0.03], dtype=np.float32))
    """position in meters (xyz) of origin of design frame."""
    design_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    """orientation quaternion (wxyz) of the design frame."""

    # TODO: these will have to be updated to be relative to the design frame
    ee_wxyz_l: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, -0.5], dtype=np.float32))
    """orientation quaternion (wxyz) of left arm end effector when performing a path."""
    ee_wxyz_r: np.ndarray = field(default_factory=lambda: np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float32))
    """orientation quaternion (wxyz) of right arm end effector when performing a path."""

    hover_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.006], dtype=np.float32))
    """position offset when hovering over point, relative to current ee frame."""
    needle_offset_l: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.0056], dtype=np.float32))
    """position offset to ensure needle touches skin, relative to current ee frame."""
    needle_offset_r: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.0062], dtype=np.float32))
    """position offset to ensure needle touches skin, relative to current ee frame."""

    inkpalette_pos: np.ndarray = field(default_factory=lambda: np.array([-0.03, 0.0, -0.0055], dtype=np.float32))
    """position (xyz, meters) of the inkpalette in global frame."""
    inkpalette_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    """orientation quaternion (wxyz) of the inkpalette in global frame."""
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

@dataclass
class Stroke:
    description: str | None = None
    """Natural language description of the path."""
    arm: str | None = None
    """Arm that will execute the path, either left or right."""
    color: str | None = None
    """Natural language description of the color of the path."""

    pixel_coords: np.ndarray | None = None
    """Numpy array of pixel coordinates for each pose in path <x (0-width), y (0-height)>."""

    meter_coords: np.ndarray | None = None
    """Numpy array of coordinates for each pose in path in meters <x, y, z>."""
    meters_center: np.ndarray | None = None
    """Center of Mass of the path in meters."""

    norm_coords: np.ndarray | None = None
    """Numpy array of coordinates for each pose in path in normalized image coordinates <x (0-1), y (0-1)>."""
    norm_center: np.ndarray | None = None
    """Center of Mass of the path in normalized image coordinates."""

    is_inkdip: bool = False
    """Whether the path is an inkdip."""
    inkcap: str | None = None
    """Name of the inkcap which provided the ink for the stroke."""


@dataclass
class Plan:
    name: str = "plan"
    """Name of the plan."""

    dirpath: str = ""
    """Path to the directory containing the plan files."""

    strokes: dict[str, Stroke] = field(default_factory=dict)
    """Dictionary of path metadata objects."""
    path_idx_to_strokes: list[list[Stroke]] = field(default_factory=list)
    """Map from pathbatch idx to list of strokes that make up that path."""

    image_width_m: float = 0.074 # A7 size
    """Width of the image in meters."""
    image_height_m: float = 0.105 # A7 size
    """Height of the image in meters."""
    image_width_px: int | None = None
    """Width of the image in pixels."""
    image_height_px: int | None = None
    """Height of the image in pixels."""

    ik_batch_size: int = 1024
    """Batch size for IK computation."""
    path_length: int = 108
    """All paths will be resampled to this length."""
    path_dt_fast: float = 0.1
    """Time between poses in seconds for fast movement."""
    path_dt_slow: float = 2.0
    """Time between poses in seconds for slow movement."""


@dataclass
class FullConfig:
    urdf: URDF
    """URDF configuration for the robot."""
    bot: BotConfig
    """Bot configuration to use for the plan."""
    ink: InkConfig
    """Config containig InkCaps and palette position."""
    tag: TagConfig
    """Config containing AprilTag parameters."""
    scan: Scan
    """Scan configuration to use for the plan."""