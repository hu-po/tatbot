
import jax.numpy as jnp

from tatbot.data.pose import Pose
from tatbot.data.stroke import Stroke
from tatbot.data.strokebatch import StrokeBatch
from tatbot.gpu.ik import batch_ik
from tatbot.utils.log import get_logger

log = get_logger('gen.strokebatch', '💠')

def strokebatch_from_strokes(
    strokes: list[tuple[Stroke, Stroke]],
    path_length: int,
    batch_size: int,
    joints: tuple[float, ...],
    urdf_path: str,
    link_names: tuple[str, ...],
    design_pose: Pose,
) -> StrokeBatch:
    log.info(f"Creating empty strokebatch of length {path_length}")
    strokebatch = StrokeBatch.empty(path_length)

    flat_target_pos   : list[list[jnp.ndarray]] = []
    flat_target_wxyz  : list[list[jnp.ndarray]] = []
    index_map: list[tuple[int, int]] = [] # (i, j)
    for i, (stroke_l, stroke_r) in enumerate(strokes):
        for j in range(path_length):
            index_map.append((i, j))
            flat_target_pos.append([stroke_l.ee_pos[j], stroke_r.ee_pos[j]])
            flat_target_wxyz.append([stroke_l.ee_wxyz[j], stroke_r.ee_wxyz[j]])
    target_pos   = jnp.array(flat_target_pos)    # (B, 2, 3)
    target_wxyz  = jnp.array(flat_target_wxyz)   # (B, 2, 4)
    for start in range(0, target_pos.shape[0], batch_size):
        end = start + batch_size
        batch_pos   = target_pos[start:end]       # (b, 2, 3)
        batch_wxyz  = target_wxyz[start:end]      # (b, 2, 4)
        batch_joints = batch_ik(
            target_wxyz=batch_wxyz,
            target_pos=batch_pos,
            joints=joints,
            urdf_path=urdf_path,
            link_names=link_names,
        )                                         # (b, 16)
        # write results back into the corresponding path / pose slots
        for k, joints in enumerate(batch_joints):
            _i, j = index_map[start + k]
            paths[_i].joints[j] = np.asarray(joints, dtype=np.float32)

#     # HACK: the right arm of the first (not counting hack paths) path should be at rest while left arm is ink dipping
#     paths[2].joints[:, 8:] = np.tile(BotConfig().rest_pose[8:], (self.path_length, 1))
#     # HACK: the left arm of the final path should be at rest since last stroke is right-only
#     paths[-1].joints[:, :8] = np.tile(BotConfig().rest_pose[:8], (self.path_length, 1))

    return strokebatch