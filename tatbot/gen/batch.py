import time

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jaxtyping import Array, Float
from typing import Optional
import numpy as np

from tatbot.data.scene import Scene
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


def strokebatch_from_strokes(scene: Scene, strokelist: StrokeList, batch_size: int = 256) -> StrokeBatch:
    """
    Convert a list of (Stroke, Stroke) tuples into a StrokeBatch, running IK to fill in joint values.
    Each tuple is (left_stroke, right_stroke) for a single stroke step.
    """
    b = len(strokelist.strokes)
    l = scene.stroke_length
    # Fill arrays from strokes
    ee_pos_l = np.zeros((b, l, 3), dtype=np.float32)
    ee_pos_r = np.zeros((b, l, 3), dtype=np.float32)

    # HACK: hardcoded orientations for left and right arm end effectors
    ee_rot_l = np.tile(scene.ee_rot_l.wxyz, (b, l, 1))
    ee_rot_r = np.tile(scene.ee_rot_r.wxyz, (b, l, 1))

    # default time between poses is fast movement
    dt = np.full((b, l), scene.arms.goal_time_fast)
    # slow movement to and from hover positions
    dt[:, :2] = scene.arms.goal_time_slow
    dt[:, -2:] = scene.arms.goal_time_slow

    for i, (stroke_l, stroke_r) in enumerate(strokelist.strokes):
        if not stroke_l.is_inkdip:
            # if the stroke is not an inkdip, ee_pos is in the design frame
            ee_pos_l[i] = transform_and_offset(
                stroke_l.ee_pos,
                scene.skin.design_pose.pos.xyz,
                scene.skin.design_pose.rot.wxyz,
                scene.ee_offset_l.xyz,
            )
        else:
            ee_pos_l[i] = stroke_l.ee_pos
        if not stroke_r.is_inkdip:
            ee_pos_r[i] = transform_and_offset(
                stroke_r.ee_pos,
                scene.skin.design_pose.pos.xyz,
                scene.skin.design_pose.rot.wxyz,
                scene.ee_offset_r.xyz,
            )
        else:
            ee_pos_r[i] = stroke_r.ee_pos
        # first and last poses in each stroke are offset by hover offset
        ee_pos_l[i, 0] += scene.hover_offset.xyz
        ee_pos_l[i, -1] += scene.hover_offset.xyz
        ee_pos_r[i, 0] += scene.hover_offset.xyz
        ee_pos_r[i, -1] += scene.hover_offset.xyz

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
            joints=scene.ready_pos_full,
            urdf_path=scene.urdf.path,
            link_names=scene.urdf.ee_link_names,
        )
        joints_out[start:end] = np.asarray(batch_joints, dtype=np.float32)
    # Reshape to (b, l, 16)
    joints_out = joints_out.reshape(b, l, 16)
    # HACK: the right arm of the first (not counting alignments) stroke should be at rest while left arm is ink dipping
    joints_out[2, :, 8:] = np.tile(scene.ready_pos_r.joints, (l, 1))
    # HACK: the left arm of the final path should be at rest since last stroke is right-only
    joints_out[-1, :, :8] = np.tile(scene.ready_pos_l.joints, (l, 1))

    strokebatch = StrokeBatch(
        ee_pos_l=jnp.array(ee_pos_l),
        ee_pos_r=jnp.array(ee_pos_r),
        ee_rot_l=jnp.array(ee_rot_l),
        ee_rot_r=jnp.array(ee_rot_r),
        joints=jnp.array(joints_out),
        dt=jnp.array(dt),
    )
    return strokebatch
