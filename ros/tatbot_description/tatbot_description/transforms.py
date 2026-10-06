"""Shared URDF and message transforms (metres, quaternion x/y/z/w)."""
import numpy as np
from scipy.spatial.transform import Rotation


def _pose(xyz, rotation):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, np.asarray(xyz, float)
    return out


def rpy_matrix(xyz, rpy):
    return _pose(xyz, Rotation.from_euler('xyz', rpy).as_matrix())


def quat_matrix(xyz, quaternion_xyzw):
    return _pose(xyz, Rotation.from_quat(quaternion_xyzw).as_matrix())


def matrix_quat(matrix):
    return tuple(float(v) for v in Rotation.from_matrix(np.asarray(matrix)[:3, :3]).as_quat(canonical=True))
