import jax.numpy as jnp
import numpy as np

from tatbot.data.pose import Pose, Pos
from tatbot.data.stroke import StrokeList
from tatbot.data.strokebatch import StrokeBatch
from tatbot.gpu.ik import batch_ik, transform_and_offset
from tatbot.utils.log import get_logger

log = get_logger('gen.strokebatch', '💠')

def strokebatch_from_strokes(
    strokelist: StrokeList,
    path_length: int,
    batch_size: int,
    joints: np.ndarray,
    urdf_path: str,
    link_names: tuple[str, ...],
    design_pose: Pose,
    needle_hover_offset: Pos,
) -> StrokeBatch:
    """
    Convert a list of (Stroke, Stroke) tuples into a StrokeBatch, running IK to fill in joint values.
    Each tuple is (left_stroke, right_stroke) for a single stroke step.
    """
    b = len(strokelist.strokes)
    l = path_length
    # Fill arrays from strokes
    ee_pos_l = np.zeros((b, l, 3), dtype=np.float32)
    ee_pos_r = np.zeros((b, l, 3), dtype=np.float32)
    ee_wxyz_l = np.zeros((b, l, 4), dtype=np.float32)
    ee_wxyz_r = np.zeros((b, l, 4), dtype=np.float32)
    dt = np.zeros((b, l), dtype=np.float32)
    for i, (stroke_l, stroke_r) in enumerate(strokelist.strokes):
        if not stroke_l.is_inkdip and not stroke_l.is_alignment:
            ee_pos_l[i] = transform_and_offset(
                stroke_l.ee_pos,
                design_pose.pos.xyz,
                design_pose.rot.wxyz
            )
        else:
            ee_pos_l[i] = stroke_l.ee_pos
        if not stroke_r.is_inkdip and not stroke_r.is_alignment:
            ee_pos_r[i] = transform_and_offset(
                stroke_r.ee_pos,
                design_pose.pos.xyz,
                design_pose.rot.wxyz
            )
        else:
            ee_pos_r[i] = stroke_r.ee_pos
        ee_wxyz_l[i] = stroke_l.ee_wxyz
        ee_wxyz_r[i] = stroke_r.ee_wxyz
        dt[i] = stroke_l.dt.squeeze() if hasattr(stroke_l.dt, 'squeeze') else stroke_l.dt
        # first and last poses in each stroke are offset by hover offset
        ee_pos_l[i, 0] += needle_hover_offset.xyz
        ee_pos_l[i, -1] += needle_hover_offset.xyz
        ee_pos_r[i, 0] += needle_hover_offset.xyz
        ee_pos_r[i, -1] += needle_hover_offset.xyz
    # Prepare IK targets: shape (b*l, 2, ...)
    target_pos = np.stack([ee_pos_l, ee_pos_r], axis=2).reshape(b * l, 2, 3)
    target_wxyz = np.stack([ee_wxyz_l, ee_wxyz_r], axis=2).reshape(b * l, 2, 4)
    # Run IK in batches
    joints_out = np.zeros((b * l, 16), dtype=np.float32)
    for start in range(0, b * l, batch_size):
        end = min(start + batch_size, b * l)
        batch_pos = jnp.array(target_pos[start:end])
        batch_wxyz = jnp.array(target_wxyz[start:end])
        batch_joints = batch_ik(
            target_wxyz=batch_wxyz,
            target_pos=batch_pos,
            joints=joints,
            urdf_path=urdf_path,
            link_names=link_names,
        )
        joints_out[start:end] = np.asarray(batch_joints, dtype=np.float32)
    # Reshape to (b, l, 16)
    joints_out = joints_out.reshape(b, l, 16)
    # HACK: the right arm of the first (not counting alignments) stroke should be at rest while left arm is ink dipping
    joints_out[2, :, 8:] = np.tile(joints[8:], (path_length, 1))
    # HACK: the left arm of the final path should be at rest since last stroke is right-only
    joints_out[-1, :, :8] = np.tile(joints[:8], (path_length, 1))

    strokebatch = StrokeBatch(
        ee_pos_l=jnp.array(ee_pos_l),
        ee_pos_r=jnp.array(ee_pos_r),
        ee_wxyz_l=jnp.array(ee_wxyz_l),
        ee_wxyz_r=jnp.array(ee_wxyz_r),
        joints=jnp.array(joints_out),
        dt=jnp.array(dt),
    )
    return strokebatch
