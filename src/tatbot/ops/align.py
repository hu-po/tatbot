import os
from dataclasses import dataclass

from lerobot.datasets.utils import build_dataset_frame

from tatbot.bot.ops.record import RecordOp, RecordOpConfig
from tatbot.data.stroke import StrokeBatch, StrokeList
from tatbot.gen.align import make_align_strokes
from tatbot.gen.batch import strokebatch_from_strokes
from tatbot.utils.log import get_logger

log = get_logger("bot.ops.align", "🔍")


@dataclass
class AlignOpConfig(RecordOpConfig):
    pass


class AlignOp(RecordOp):

    op_name: str = "align"

    async def _run(self):
        _msg = "Generating alignment strokes..."
        log.info(_msg)
        yield {
            'progress': 0.2,
            'message': _msg,
        }
        strokes: StrokeList = make_align_strokes(self.scene)
        strokes.to_yaml(os.path.join(self.dataset_dir, "strokes.yaml"))

        _msg = "Creating stroke batch from strokes..."
        log.info(_msg)
        yield {
            'progress': 0.21,
            'message': _msg,
        }
        strokebatch: StrokeBatch = strokebatch_from_strokes(self.scene, strokes, first_last_rest=False)
        strokebatch.save(os.path.join(self.dataset_dir, "strokebatch.safetensors"))

        # maximally retracted when performing alignment operation
        offset_idx_l = self.scene.arms.offset_num - 1
        offset_idx_r = self.scene.arms.offset_num - 1

        for stroke_idx, (stroke_l, stroke_r) in enumerate(strokes.strokes):
            _msg = f"🔍 Executing stroke {stroke_idx + 1}/{len(strokes.strokes)}: left={stroke_l.description}, right={stroke_r.description}"
            log.info(_msg)
            yield {
                'progress': 0.3 + (0.6 * stroke_idx / len(strokes.strokes)),
                'message': _msg,
            }

            for pose_idx in range(self.scene.stroke_length):
                observation = self.robot.get_observation()
                observation_frame = build_dataset_frame(self.dataset.features, observation, prefix="observation")
                
                joints = strokebatch.offset_joints(stroke_idx, pose_idx, offset_idx_l, offset_idx_r)
                robot_action = self.robot._urdf_joints_to_action(joints)
                goal_time = float(
                    strokebatch.dt[stroke_idx, pose_idx, offset_idx_l]
                )  # TODO: this is a hack, currently dt is the same for both arms
                sent_action = self.robot.send_action(robot_action, goal_time=goal_time, block="none")

                action_frame = build_dataset_frame(self.dataset.features, sent_action, prefix="action")
                frame = {**observation_frame, **action_frame}
                self.dataset.add_frame(frame, task=f"left: {stroke_l.description}, right: {stroke_r.description}")

            self.dataset.save_episode()