from dataclasses import dataclass

@dataclass
class URDF:
    path: str = "~/tatbot/assets/urdf/tatbot.urdf"
    """Path to the URDF file for the robot."""
    ee_link_names: tuple[str, str] = (
        "left/tattoo_needle", 
        "right/tattoo_needle"
    )
    """Names of the ee (end effector) links in the URDF."""
    tag_link_names: tuple[str, ...] = (
        "tag6", 
        "tag7", 
        "tag9", 
        "tag10", 
        "tag11"
    )
    """Names of the tag (apriltag) links in the URDF."""
    cam_link_names: tuple[str, ...] = (
        "realsense1", 
        "realsense2", 
        "camera1", 
        "camera2", 
        "camera3", 
        "camera4", 
        "camera5"
    )
    """Names of the camera links in the URDF."""
    ink_link_names: tuple[str, ...] = (
        "inkcap_large", 
        "inkcap_small_1", 
        "inkcap_small_2", 
        "inkcap_small_3", 
        "inkcap_medium_1", 
        "inkcap_medium_2"
    )
    """Names of the inkcap links in the URDF."""
    palette_link_name: str = "inkpalette"
    """Name of the inkpalette link in the URDF."""
    origin_link_name: str = "origin"
    """Name of the origin link in the URDF."""
    skin_link_name: str = "skin"
    """Name of the skin link in the URDF."""