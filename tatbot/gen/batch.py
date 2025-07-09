import time

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jaxtyping import Array, Float
from typing import Optional
import numpy as np

from tatbot.data.pose import Pos, Pose
from tatbot.data.stroke import StrokeList
from tatbot.data.strokebatch import StrokeBatch
from tatbot.gen.ik import batch_ik
from tatbot.utils.log import get_logger

log = get_logger('gen.strokebatch', '💠')

@jdc.jit
def transform_and_offset(
    target_pos: Float[Array, "b 3"],
    frame_pos: Float[Array, "3"],
    frame_wxyz: Float[Array, "4"],
    offsets: Optional[Float[Array, "b 3"]] = None,
) -> Float[Array, "b 3"]:
    log.debug(f"transforming and offsetting {target_pos.shape[0]} points")
    start_time = time.time()
    if offsets is None:
        offsets = jnp.zeros_like(target_pos)
    if offsets.shape[0] != target_pos.shape[0]:
        offsets = jnp.tile(offsets, (target_pos.shape[0], 1))
    frame_transform = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(frame_wxyz), frame_pos)
    result = jax.vmap(lambda pos, offset: frame_transform @ pos + offset)(target_pos, offsets)
    log.debug(f"transform and offset time: {time.time() - start_time:.4f}s")
    return result


def strokebatch_from_strokes(
    strokelist: StrokeList,
    stroke_length: int,
    joints: np.ndarray,
    urdf_path: str,
    link_names: tuple[str, ...],
    design_pose: Pose,
    hover_offset: Pos,
    ee_offset_l: Pos,
    ee_offset_r: Pos,
    batch_size: int = 256,
) -> StrokeBatch:
    """
    Convert a list of (Stroke, Stroke) tuples into a StrokeBatch, running IK to fill in joint values.
    Each tuple is (left_stroke, right_stroke) for a single stroke step.
    """
    b = len(strokelist.strokes)
    l = stroke_length
    # Fill arrays from strokes
    ee_pos_l = np.zeros((b, l, 3), dtype=np.float32)
    ee_pos_r = np.zeros((b, l, 3), dtype=np.float32)
    ee_rot_l = np.zeros((b, l, 4), dtype=np.float32)
    ee_rot_r = np.zeros((b, l, 4), dtype=np.float32)
    dt = np.zeros((b, l), dtype=np.float32)
    for i, (stroke_l, stroke_r) in enumerate(strokelist.strokes):
        if not stroke_l.is_inkdip:
            ee_pos_l[i] = transform_and_offset(
                stroke_l.ee_pos,
                design_pose.pos.xyz,
                design_pose.rot.wxyz,
                ee_offset_l.xyz,
            )
        else:
            ee_pos_l[i] = stroke_l.ee_pos
        if not stroke_r.is_inkdip:
            ee_pos_r[i] = transform_and_offset(
                stroke_r.ee_pos,
                design_pose.pos.xyz,
                design_pose.rot.wxyz,
                ee_offset_r.xyz,
            )
        else:
            ee_pos_r[i] = stroke_r.ee_pos
        # first and last poses in each stroke are offset by hover offset
        ee_pos_l[i, 0] += hover_offset.xyz
        ee_pos_l[i, -1] += hover_offset.xyz
        ee_pos_r[i, 0] += hover_offset.xyz
        ee_pos_r[i, -1] += hover_offset.xyz
        # rotations are hardcoded
        ee_rot_l[i] = stroke_l.ee_rot
        ee_rot_r[i] = stroke_r.ee_rot
        dt[i] = stroke_l.dt.squeeze() if hasattr(stroke_l.dt, 'squeeze') else stroke_l.dt
    # Prepare IK targets: shape (b*l, 2, ...)
    target_pos = np.stack([ee_pos_l, ee_pos_r], axis=2).reshape(b * l, 2, 3)
    target_wxyz = np.stack([ee_rot_l, ee_rot_r], axis=2).reshape(b * l, 2, 4)
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
    joints_out[2, :, 8:] = np.tile(joints[8:], (stroke_length, 1))
    # HACK: the left arm of the final path should be at rest since last stroke is right-only
    joints_out[-1, :, :8] = np.tile(joints[:8], (stroke_length, 1))

    strokebatch = StrokeBatch(
        ee_pos_l=jnp.array(ee_pos_l),
        ee_pos_r=jnp.array(ee_pos_r),
        ee_rot_l=jnp.array(ee_rot_l),
        ee_rot_r=jnp.array(ee_rot_r),
        joints=jnp.array(joints_out),
        dt=jnp.array(dt),
    )
    return strokebatch
