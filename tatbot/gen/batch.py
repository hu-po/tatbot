import time

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jaxtyping import Array, Float
from typing import Optional
import numpy as np

from tatbot.data.scene import Scene
from tatbot.data.stroke import StrokeList, StrokeBatch
from tatbot.gen.ik import batch_ik
from tatbot.utils.log import get_logger

log = get_logger('gen.batch', '💠')

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


def strokebatch_from_strokes(scene: Scene, strokelist: StrokeList, batch_size: int = 1024) -> StrokeBatch:
    """
    Convert a list of (Stroke, Stroke) tuples into a StrokeBatch, running IK to fill in joint values.
    Each tuple is (left_stroke, right_stroke) for a single stroke step.
    """
    b = len(strokelist.strokes)              # strokes in list
    l = scene.stroke_length                  # poses per stroke
    o = scene.offset_num                     # offset samples

    # Fill arrays from strokes
    ee_pos_l = np.zeros((b, l, o, 3), dtype=np.float32)
    ee_pos_r = np.zeros((b, l, o, 3), dtype=np.float32)

    # HACK: hardcoded orientations for left and right arm end effectors
    ee_rot_l = np.tile(scene.ee_rot_l.wxyz, (b, l, o, 1))
    ee_rot_r = np.tile(scene.ee_rot_r.wxyz, (b, l, o, 1))

    # default time between poses is fast movement
    dt = np.full((b, l, o), scene.arms.goal_time_fast, dtype=np.float32)
    # slow movement to and from hover positions
    dt[:, :2, :] = scene.arms.goal_time_slow
    dt[:, -2:, :] = scene.arms.goal_time_slow

    for i, (stroke_l, stroke_r) in enumerate(strokelist.strokes):
        if not stroke_l.is_inkdip:
            # if the stroke is not an inkdip, ee_pos is in the design frame
            base_l = transform_and_offset(
                stroke_l.ee_pos,
                scene.skin.design_pose.pos.xyz,
                scene.skin.design_pose.rot.wxyz,
                scene.ee_offset_l.xyz,
            )
            base_l = base_l.reshape(l, 3)
            ee_pos_l[i] = np.repeat(base_l[:, None, :], o, axis=1)
        else:
            ee_pos_l[i] = np.repeat(stroke_l.ee_pos.reshape(l, 1, 3), o, 1)
        if not stroke_r.is_inkdip:
            base_r = transform_and_offset(
                stroke_r.ee_pos,
                scene.skin.design_pose.pos.xyz,
                scene.skin.design_pose.rot.wxyz,
                scene.ee_offset_r.xyz,
            )
            base_r = base_r.reshape(l, 3)
            ee_pos_r[i] = np.repeat(base_r[:, None, :], o, 1)
        else:
            ee_pos_r[i] = np.repeat(stroke_r.ee_pos.reshape(l, 1, 3), o, 1)

        # first and last poses in each stroke are offset by hover offset
        ee_pos_l[i, 0] += scene.hover_offset.xyz
        ee_pos_l[i, -1] += scene.hover_offset.xyz
        ee_pos_r[i, 0] += scene.hover_offset.xyz
        ee_pos_r[i, -1] += scene.hover_offset.xyz

    # offset depths
    offsets = np.linspace(scene.offset_range[0], scene.offset_range[1], o).astype(np.float32)
    depth_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    ee_pos_l += offsets[None, None, :, None] * depth_axis
    ee_pos_r += offsets[None, None, :, None] * depth_axis

    # ------------------------------------------------------------------ #
    # stack ALONG AXIS 3  → (b, l, o, 2, 3/4) so that every row in the
    # later (b·l·o, 2, …) tensor contains the **left & right arm for the
    # SAME (stroke, pose, offset)**.  Stacking on axis 2 (as before) gave
    # (b, l, 2, o, …) and produced mismatched pairs after reshape.
    # ------------------------------------------------------------------ #
    target_pos   = (
        np.stack([ee_pos_l, ee_pos_r], axis=3)     # (b, l, o, 2, 3)
        .reshape(b * l * o, 2, 3)
    )
    target_wxyz  = (
        np.stack([ee_rot_l, ee_rot_r], axis=3)     # (b, l, o, 2, 4)
        .reshape(b * l * o, 2, 4)
    )

    # Run IK in batches
    joints_out = np.zeros((b * l * o, 16), dtype=np.float32)
    for start in range(0, b * l * o, batch_size):
        end = min(start + batch_size, b * l * o)
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
    joints_out = joints_out.reshape(b, l, o, 16)

    # HACK: the right arm of the first stroke should be at rest while left arm is ink dipping
    joints_out[0, :, :, 8:] = np.tile(scene.ready_pos_r.joints, (l, o, 1))
    # HACK: the left arm of the final path should be at rest since last stroke is right-only
    joints_out[-1, :, :, :8] = np.tile(scene.ready_pos_l.joints, (l, o, 1))

    strokebatch = StrokeBatch(
        ee_pos_l=jnp.array(ee_pos_l),
        ee_pos_r=jnp.array(ee_pos_r),
        ee_rot_l=jnp.array(ee_rot_l),
        ee_rot_r=jnp.array(ee_rot_r),
        joints=jnp.array(joints_out),
        dt=jnp.array(dt),
    )
    return strokebatch
