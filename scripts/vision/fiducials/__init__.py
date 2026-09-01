"""Shared fiducial configuration, detection, geometry, and wire models."""

from .config import (
    DEFAULT_INVENTORY_PATH,
    DetectorProfile,
    FiducialInventory,
    TargetSpec,
    load_inventory,
)
from .geometry import (
    TAG_CORNER_SIGNS,
    invert,
    matrix_from_pose,
    normalize_corners,
    pose_vector,
    rotation_distance_deg,
    tag_model_corners,
    transform_from_vector,
    transform_points,
)

__all__ = [
    "DEFAULT_INVENTORY_PATH",
    "DetectorProfile",
    "FiducialInventory",
    "TAG_CORNER_SIGNS",
    "TargetSpec",
    "invert",
    "load_inventory",
    "matrix_from_pose",
    "normalize_corners",
    "pose_vector",
    "rotation_distance_deg",
    "tag_model_corners",
    "transform_from_vector",
    "transform_points",
]
