

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