import jax
import jax.numpy as jnp
import jaxlie
import numpy as np

from tatbot.data.scene import Scene
from tatbot.data.stroke import StrokeBatch, StrokeList
from tatbot.gen.ik import batch_ik
from tatbot.utils.log import get_logger

log = get_logger("gen.batch", "💠")


def strokebatch_from_strokes(
    scene: Scene, 
    strokelist: StrokeList, 
    batch_size: int = 256, 
    first_last_rest: bool = True, 
    use_ee_offsets: bool = True
) -> StrokeBatch:
    """
    Convert a list of (Stroke, Stroke) tuples into a StrokeBatch, running IK to fill in joint values.
    Each tuple is (left_stroke, right_stroke) for a single stroke step.
    
    Args:
        scene: Scene configuration with robot and workspace info
        strokelist: List of stroke pairs to process
        batch_size: IK batch processing size
        first_last_rest: Whether to keep arms at rest during first/last strokes
        use_ee_offsets: Whether to apply end-effector offsets
    """
    b = len(strokelist.strokes)
    l = scene.stroke_length
    o = scene.arms.offset_num

    # Fill arrays from strokes
    ee_pos_l = np.zeros((b, l, o, 3), dtype=np.float32)
    ee_pos_r = np.zeros((b, l, o, 3), dtype=np.float32)

    # Center of laser cross is the origin of the tattoo design
    lasercross_tf = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(scene.lasercross_pose.rot.wxyz), 
        scene.lasercross_pose.pos.xyz
    )
    
    # Calibrator transformation for inkdip strokes
    calibrator_tf = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(scene.calibrator_pose.rot.wxyz), 
        scene.calibrator_pose.pos.xyz
    )

    # Transform end effector orientations to be relative to different frames
    base_ee_rot_l = jaxlie.SO3(scene.arms.ee_rot_l.wxyz)
    base_ee_rot_r = jaxlie.SO3(scene.arms.ee_rot_r.wxyz)
    
    # Apply lasercross rotation to base EE orientations (for normal strokes)
    lasercross_relative_ee_rot_l = lasercross_tf.rotation() @ base_ee_rot_l
    lasercross_relative_ee_rot_r = lasercross_tf.rotation() @ base_ee_rot_r
    
    # Apply calibrator rotation to base EE orientations (for inkdip strokes)
    calibrator_relative_ee_rot_l = calibrator_tf.rotation() @ base_ee_rot_l
    calibrator_relative_ee_rot_r = calibrator_tf.rotation() @ base_ee_rot_r
    
    # Start with lasercross-relative orientations for all strokes (will override inkdip strokes later)
    ee_rot_l = np.tile(lasercross_relative_ee_rot_l.wxyz, (b, l, o, 1))
    ee_rot_r = np.tile(lasercross_relative_ee_rot_r.wxyz, (b, l, o, 1))

    def _pad_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
        """Pad or trim first axis to target_len by repeating last element."""
        if arr.shape[0] >= target_len:
            return arr[:target_len]
        if arr.shape[0] == 0:
            # pad with zeros if empty
            return np.zeros((target_len, *arr.shape[1:]), dtype=arr.dtype)
        pad_count = target_len - arr.shape[0]
        last = arr[-1:]
        pads = np.repeat(last, pad_count, axis=0)
        return np.concatenate([arr, pads], axis=0)

    for i, (stroke_l, stroke_r) in enumerate(strokelist.strokes):
        # Left arm
        if not stroke_l.is_inkdip:
            if stroke_l.meter_coords is not None:
                n_real_l = int(stroke_l.meter_coords.shape[0])
                base_l_full = jax.vmap(lambda pos: lasercross_tf @ pos)(stroke_l.meter_coords)  # (Nl,3)
                base_l = _pad_to_length(base_l_full.astype(np.float32), l)
                ee_pos_l[i] = np.repeat(base_l[:, None, :], o, axis=1)
            else:
                n_real_l = 0
                ee_pos_l[i] = np.zeros((l, o, 3), dtype=np.float32)
        else:
            # inkdips may have ee_pos provided (already in world frame target)
            n_real_l = int(stroke_l.ee_pos.shape[0]) if stroke_l.ee_pos is not None else l
            ee_l = stroke_l.ee_pos if stroke_l.ee_pos is not None else np.zeros((l, 3), dtype=np.float32)
            ee_l = _pad_to_length(ee_l.astype(np.float32), l)
            ee_pos_l[i] = np.repeat(ee_l.reshape(l, 1, 3), o, 1)
            ee_rot_l[i] = np.tile(calibrator_relative_ee_rot_l.wxyz, (l, o, 1))

        # Right arm
        if not stroke_r.is_inkdip:
            if stroke_r.meter_coords is not None:
                n_real_r = int(stroke_r.meter_coords.shape[0])
                base_r_full = jax.vmap(lambda pos: lasercross_tf @ pos)(stroke_r.meter_coords)  # (Nr,3)
                base_r = _pad_to_length(base_r_full.astype(np.float32), l)
                ee_pos_r[i] = np.repeat(base_r[:, None, :], o, 1)
            else:
                n_real_r = 0
                ee_pos_r[i] = np.zeros((l, o, 3), dtype=np.float32)
        else:
            n_real_r = int(stroke_r.ee_pos.shape[0]) if stroke_r.ee_pos is not None else l
            ee_r = stroke_r.ee_pos if stroke_r.ee_pos is not None else np.zeros((l, 3), dtype=np.float32)
            ee_r = _pad_to_length(ee_r.astype(np.float32), l)
            ee_pos_r[i] = np.repeat(ee_r.reshape(l, 1, 3), o, 1)
            ee_rot_r[i] = np.tile(calibrator_relative_ee_rot_r.wxyz, (l, o, 1))

        # add ee_offset
        if use_ee_offsets:
            log.debug("Using ee offsets")
            ee_pos_l[i] += scene.arms.ee_offset_l.xyz
            ee_pos_r[i] += scene.arms.ee_offset_r.xyz

        # Apply hover to first pose always
        ee_pos_l[i, 0] += scene.arms.hover_offset.xyz
        ee_pos_r[i, 0] += scene.arms.hover_offset.xyz

        # Handle hover offsets for padded regions (avoid double application)
        if n_real_l < l and n_real_l > 0:
            tail_start = max(1, n_real_l)  # Start from 1 to avoid double-hover on first pose
            # Set entire tail to last real pose + hover
            last_real_pose = ee_pos_l[i, n_real_l - 1] if n_real_l > 0 else ee_pos_l[i, 0]
            ee_pos_l[i, tail_start:l] = last_real_pose + scene.arms.hover_offset.xyz
        elif n_real_l == l:
            # No padding: just hover last pose
            ee_pos_l[i, -1] += scene.arms.hover_offset.xyz
        # If n_real_l == 0, all poses are already at hover level from first pose

        if n_real_r < l and n_real_r > 0:
            tail_start = max(1, n_real_r)  # Start from 1 to avoid double-hover on first pose  
            last_real_pose = ee_pos_r[i, n_real_r - 1] if n_real_r > 0 else ee_pos_r[i, 0]
            ee_pos_r[i, tail_start:l] = last_real_pose + scene.arms.hover_offset.xyz
        elif n_real_r == l:
            # No padding: just hover last pose
            ee_pos_r[i, -1] += scene.arms.hover_offset.xyz
        # If n_real_r == 0, all poses are already at hover level from first pose

    # Apply depth offsets
    offsets = np.linspace(scene.arms.offset_range[0], scene.arms.offset_range[1], o).astype(np.float32)
    depth_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ee_pos_l += offsets[None, None, :, None] * depth_axis
    ee_pos_r += offsets[None, None, :, None] * depth_axis

    # Stack left and right arm data along axis 3 to ensure matching pairs
    target_pos = np.stack([ee_pos_l, ee_pos_r], axis=3).reshape(b * l * o, 2, 3)  # (b, l, o, 2, 3)
    target_wxyz = np.stack([ee_rot_l, ee_rot_r], axis=3).reshape(b * l * o, 2, 4)  # (b, l, o, 2, 4)

    # Run IK in batches
    log.info("Using JAX-based IK solver")
    joints_out = np.zeros((b * l * o, 14), dtype=np.float32)
    for start in range(0, b * l * o, batch_size):
        end = min(start + batch_size, b * l * o)
        
        # Use JAX IK solver
        batch_pos = jnp.array(target_pos[start:end])
        batch_wxyz = jnp.array(target_wxyz[start:end])
        batch_joints = batch_ik(
            target_wxyz=batch_wxyz,
            target_pos=batch_pos,
            joints=scene.ready_pos_full.joints,
            urdf_path=scene.urdf.path,
            link_names=scene.urdf.ee_link_names,
        )
        joints_out[start:end] = np.asarray(batch_joints, dtype=np.float32)
    joints_out = joints_out.reshape(b, l, o, 14)

    if first_last_rest:
        log.debug("Using first and last rest")
        # Keep right arm at rest during first stroke (left arm ink dipping)
        joints_out[0, :, :, 7:] = np.tile(scene.ready_pos_r.joints, (l, o, 1))
        # Keep left arm at rest during final stroke (right arm only)
        joints_out[-1, :, :, :7] = np.tile(scene.ready_pos_l.joints, (l, o, 1))

    strokebatch = StrokeBatch(
        ee_pos_l=jnp.array(ee_pos_l),
        ee_pos_r=jnp.array(ee_pos_r),
        ee_rot_l=jnp.array(ee_rot_l),
        ee_rot_r=jnp.array(ee_rot_r),
        joints=jnp.array(joints_out),
    )
    return strokebatch
