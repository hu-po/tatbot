"""Explicit pinhole projection for rendered captures in the existing NPZ format.

The current camera registry owns optical-frame identity. The capture owns the
pixel intrinsics and depth units actually emitted. This descriptor does not
pretend to be a RealSense device profile or measured camera calibration.
"""

from __future__ import annotations

import math

SCHEMA = 'tatbot.capture-projection/1'


def pinhole(camera, *, units_m=0.001):
    if not math.isfinite(units_m) or units_m <= 0:
        raise ValueError('capture depth unit must be positive')
    return {'schema': SCHEMA, 'kind': 'pinhole', 'source': 'simulation',
            'camera': camera.as_dict(), 'optical_frame': camera.optical_frame,
            'depth_axis': 'optical-z', 'units_m': units_m, 'hardware_authority': False}


def validate(profile, camera, *, shape, intrinsics, units_m):
    """Return the registered optical frame after checking the stored pixels."""
    if (not isinstance(profile, dict) or profile.get('schema') != SCHEMA
            or profile.get('kind') != 'pinhole' or profile.get('source') != 'simulation'
            or profile.get('hardware_authority') is not False or profile.get('depth_axis') != 'optical-z'):
        raise ValueError('invalid capture projection descriptor')
    description = profile.get('camera')
    if (not isinstance(description, dict) or description.get('role') != camera.role
            or description.get('arm') != camera.arm
            or description.get('optical_frame') != camera.optical_frame
            or profile.get('optical_frame') != camera.optical_frame):
        raise ValueError('capture projection belongs to another camera frame')
    expected = [description.get('height'), description.get('width')]
    if len(shape) != 2 or list(shape) != expected or len(intrinsics) != 6:
        raise ValueError('capture projection dimensions differ from pixels')
    values = description.get('intrinsic')
    if (not isinstance(values, (list, tuple)) or len(values) != 4
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
            or min(values[:2]) <= 0 or not math.isfinite(units_m) or units_m <= 0
            or profile.get('units_m') != units_m
            or list(intrinsics) != [*values, expected[1], expected[0]]):
        raise ValueError('capture projection intrinsics or depth units differ from pixels')
    return camera.optical_frame
