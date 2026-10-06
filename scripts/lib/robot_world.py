"""Robot-world registration uses the URDF root, including fixed arm mounts."""

import math

import numpy as np

# A registration carried onto a refreshed bundle is the same world only when
# the refresh kept the fixed cameras: the golden's own carry receipt bounds it.
CARRY_TRANSLATION_M = 0.001
CARRY_ROTATION_DEG = 0.05


def rigid_transform(value):
    """Require a finite SE(3) matrix; an inverse must not disguise bad data."""
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3],
                               np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError('robot-world transform must be a finite rigid 4x4 matrix')
    return matrix


_UNSPECIFIED = object()


def root_from_world(record, *, calibration_id=_UNSPECIFIED):
    # Legacy serialized name: the robot-world solve (removed 2026-09-29) fitted Z @ UrdfChain.link_pose,
    # and link_pose already includes root -> arm base. Preserve those bytes and
    # their digests; adding the arm mount again changes the measured geometry.
    if calibration_id is not _UNSPECIFIED and (not calibration_id
            or record.get('calibration_id') != calibration_id):
        raise ValueError('camera and robot-world calibration IDs differ')
    world_from_root = rigid_transform(record['world_from_base'])
    return np.linalg.inv(world_from_root)


def rotation_angle_deg(matrix):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(matrix[:3, :3]) - 1) / 2))))


def registration(arm, bundle, record, golden):
    """world <- arm base from the arm's own registration, valid in the bundle's world.

    The registration is accepted on the bundle it was solved against, or on a
    bundle whose golden robot-world was carried from that one with the fixed
    cameras kept (an identity world change within tolerance). Anything else is
    another world and is refused."""
    if record.get('schema') != 'tatbot.arm-registration/1' or record.get('arm') != arm:
        raise ValueError(f'registration is not a tatbot.arm-registration/1 record for the {arm} arm')
    world_from_arm_base = rigid_transform(record['world_from_arm_base'])
    solved = record['calibration_id']
    provenance = {'calibration_id': solved, 'carried': None}
    if solved != bundle['bundle_id']:
        carried = (golden or {}).get('carried') or {}
        change = carried.get('world_change')
        if ((golden or {}).get('calibration_id') != bundle['bundle_id']
                or carried.get('from_calibration_id') != solved or change is None):
            raise ValueError('registration was solved against another camera bundle and was not carried onto this one')
        change = rigid_transform(change)
        if (np.linalg.norm(change[:3, 3]) > CARRY_TRANSLATION_M
                or rotation_angle_deg(change) > CARRY_ROTATION_DEG):
            raise ValueError('the bundle refresh moved the fixed-camera world; re-solve the registration')
        world_from_arm_base = change @ world_from_arm_base
        provenance['carried'] = {'onto': bundle['bundle_id'], 'translation_mm': float(np.linalg.norm(change[:3, 3]) * 1000),
                                 'rotation_deg': rotation_angle_deg(change)}
    return world_from_arm_base, provenance
