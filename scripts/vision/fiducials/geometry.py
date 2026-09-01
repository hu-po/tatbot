"""One corner and SE(3) convention shared by calibration, live, and sim."""

from __future__ import annotations

import numpy as np

TAG_CORNER_SIGNS = np.array(
    [[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]], dtype=np.float64
)


def normalize_corners(corners) -> np.ndarray:
    """Return detector corners as TL/TR/BR/BL tag-frame coordinates."""
    out = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(out).all():
        raise ValueError("fiducial corners must be finite")
    return out


def tag_model_corners(edge_m: float) -> np.ndarray:
    if not np.isfinite(edge_m) or edge_m <= 0:
        raise ValueError("tag edge must be positive")
    return np.c_[TAG_CORNER_SIGNS * (edge_m / 2.0), np.zeros(4)]


def matrix_from_pose(rotation, translation) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    out[:3, 3] = np.asarray(translation, dtype=np.float64)
    return out


def pose_vector(transform: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    transform = np.asarray(transform, dtype=np.float64)
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]]


def transform_from_vector(vector: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    vector = np.asarray(vector, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    out[:3, 3] = vector[3:6]
    return out


def invert(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ transform[:3, 3]
    return out


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    relative = left[:3, :3].T @ right[:3, :3]
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))
