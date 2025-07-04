from dataclasses import dataclass

from tatbot.data import Yaml
from tatbot.data.pose import Pos, Rot


@dataclass
class Plan(Yaml):
    name: str
    """Name of the plan."""

    image_width_m: float
    """Width of the image in meters."""
    image_height_m: float
    """Height of the image in meters."""
    image_width_px: int
    """Width of the image in pixels."""
    image_height_px: int
    """Height of the image in pixels."""

    left_arm_pen_names: list[str]
    """Name of pens that will be drawn using left arm."""
    right_arm_pen_names: list[str]
    """Name of pens that will be drawn using right arm."""

    stroke_length: int
    """All strokes will be resampled to this length."""
    ik_batch_size: int
    """Batch size for IK computation."""

    ee_rot_l: Rot
    """<w, x, y, z> quaternion of left arm end effector when performing a stroke."""
    ee_rot_r: Rot
    """<w, x, y, z> quaternion of right arm end effector when performing a stroke."""

    inkdip_hover_offset: Pos
    """<x, y, z> offset for inkdip hover in meters."""
    needle_hover_offset: Pos
    """<x, y, z> offset for needle hover in meters."""

    needle_offset_l: Pos
    """<x, y, z> offset for needle stroke in meters."""
    needle_offset_r: Pos
    """<x, y, z> offset for needle stroke in meters."""

    yaml_dir: str = "~/tatbot/config/plans"
    """Directory containing the config yaml files."""