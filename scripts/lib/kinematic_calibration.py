"""Shared encoder-zero calibration for ROS geometry and wrist-camera poses.

Offsets change the model, never controller encoder values, commands or limits.
Probe fits are increments against the model loaded for that bound workspace.
"""
from __future__ import annotations

import math

from tool_spec import _rpy_matrix

KEY = 'joint_offsets_rad'


def validate_offsets(values=None):
    values = [0.] * 7 if values is None else values
    if (not isinstance(values, (list, tuple)) or len(values) != 7
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or any(values[i] != 0 for i in (0, 5, 6))):
        raise ValueError('joint offsets require seven finite values, with only joints 1-4 nonzero')
    return list(map(float, values))


def apply_origins(joints, prefix, offsets):
    """R_origin *= R_axis(offset) makes FK(raw q) equal nominal FK(q+offset)."""
    by_name = {joint.get('name'): joint for joint in joints}
    for i, angle in enumerate(validate_offsets(offsets)):
        if angle == 0:
            continue
        joint = by_name[f'{prefix}/joint_{i}']
        if joint.get('type') not in ('revolute', 'continuous'):
            raise ValueError('encoder-zero calibration requires a revolute joint')
        origin = joint.find('origin')
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
        axis = [float(v) for v in joint.find('axis').get('xyz').split()]
        norm = math.sqrt(sum(v*v for v in axis))
        if norm == 0:
            raise ValueError('joint axis is zero')
        x, y, z = [v/norm for v in axis]
        c, s, t = math.cos(angle), math.sin(angle), 1-math.cos(angle)
        rotation = ((t*x*x+c, t*x*y-s*z, t*x*z+s*y),
                    (t*x*y+s*z, t*y*y+c, t*y*z-s*x),
                    (t*x*z-s*y, t*y*z+s*x, t*z*z+c))
        old = _rpy_matrix(rpy)
        r = [[sum(old[a][k]*rotation[k][b] for k in range(3)) for b in range(3)] for a in range(3)]
        pitch = math.asin(max(-1., min(1., -r[2][0])))
        if abs(math.cos(pitch)) > 1e-9:
            roll, yaw = math.atan2(r[2][1], r[2][2]), math.atan2(r[1][0], r[0][0])
        else:
            roll, yaw = math.atan2(-r[1][2], r[1][1]), 0.
        origin.set('rpy', ' '.join(map(repr, (roll, pitch, yaw))))
