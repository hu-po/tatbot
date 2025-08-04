import logging
import os
import shutil
import time
from dataclasses import dataclass
from io import StringIO

from lerobot.datasets.utils import build_dataset_frame
from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.teleoperators.gamepad import AtariTeleoperator, AtariTeleoperatorConfig
from lerobot.utils.robot_utils import busy_wait

from tatbot.data.stroke import StrokeBatch, StrokeList
from tatbot.gen.batch import strokebatch_from_strokes
from tatbot.gen.gcode import make_gcode_strokes
from tatbot.ops.record import RecordOp, RecordOpConfig
from tatbot.utils.log import LOG_FORMAT, TIME_FORMAT, get_logger

log = get_logger("ops.stroke", "🖌️")


@dataclass
class StrokeOpConfig(RecordOpConfig):
    pass


class StrokeOp(RecordOp):

    op_name: str = "stroke"

    # TODO: the realsense cameras are just too slow on trossen-ai
    #       we need them for finetuning VLA, but recording is just too slow
    #       some kind of long term solution is needed
    # def make_robot(self) -> Robot:
    #     """Make a robot from the config."""
    #     return make_robot_from_config(
    #         TatbotConfig(
    #             ip_address_l=self.scene.arms.ip_address_l,
    #             ip_address_r=self.scene.arms.ip_address_r,
    #             arm_l_config_filepath=self.scene.arms.arm_l_config_filepath,
    #             arm_r_config_filepath=self.scene.arms.arm_r_config_filepath,
    #             goal_time=self.scene.arms.goal_time_slow,
    #             connection_timeout=self.scene.arms.connection_timeout,
    #             home_pos_l=self.scene.sleep_pos_l.joints,
    #             home_pos_r=self.scene.sleep_pos_r.joints,
    #             rs_cameras={
    #                 cam.name : RealSenseCameraConfig(
    #                     fps=cam.fps,
    #                     width=cam.width,
    #                     height=cam.height,
    #                     serial_number_or_name=cam.serial_number,
    #                 ) for cam in self.scene.cams.realsenses
    #             },
    #             ip_cameras={},
    #         )
    #     )

    async def _run(self):
        _msg = "Creating episode logger..."
        log.info(_msg)
        yield {
            'progress': 0.2,
            'message': _msg,
        }
        logs_dir = os.path.join(self.dataset_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        episode_log_buffer = StringIO()

        class EpisodeLogHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                episode_log_buffer.write(msg + "\n")

        episode_handler = EpisodeLogHandler()
        episode_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=TIME_FORMAT))
        logging.getLogger().addHandler(episode_handler)

        _msg = "Connecting to joystick..."
        log.info(_msg)
        yield {
            'progress': 0.21,
            'message': _msg,
        }
        atari_teleop = AtariTeleoperator(AtariTeleoperatorConfig())
        atari_teleop.connect()

        _msg = "Loading strokes..."
        log.info(_msg)
        yield {
            'progress': 0.22,
            'message': _msg,
        }
        strokes_path = os.path.join(self.dataset_dir, "strokes.yaml")
        strokebatch_path = os.path.join(self.dataset_dir, "strokebatch.safetensors")
        if self.config.resume:
            log.info(f"🔄 Resuming from {self.dataset_dir}")
            assert os.path.exists(strokes_path), f"❌ Strokes file {strokes_path} does not exist"
            assert os.path.exists(strokebatch_path), f"❌ Strokebatch file {strokebatch_path} does not exist"
            strokes: StrokeList = StrokeList.from_yaml(strokes_path)
            strokebatch: StrokeBatch = StrokeBatch.load(strokebatch_path)
        else:
            strokes: StrokeList = make_gcode_strokes(self.scene)
            strokes.to_yaml(strokes_path)
            strokebatch: StrokeBatch = strokebatch_from_strokes(self.scene, strokes)
            strokebatch.save(strokebatch_path)
        num_strokes = len(strokes.strokes)

        # offset index controls needle depth
        mid_offset_idx: int = self.scene.arms.offset_num // 2
        offset_idx_l: int = mid_offset_idx
        offset_idx_r: int = mid_offset_idx
        # inkdip-specific offset index
        inkdip_offset_idx_l: int = mid_offset_idx
        inkdip_offset_idx_r: int = mid_offset_idx

        # one episode is a single path
        # when resuming, start from the idx of the next episode
        log.info(f"Recording {num_strokes} paths...")
        for stroke_idx in range(self.dataset.num_episodes, num_strokes):

            # reset in-memory log buffer for the new episode
            episode_log_buffer.seek(0)
            episode_log_buffer.truncate(0)

            # make sure robot is connected and in ready position
            if not self.robot.is_connected:
                log.warning("⚠️ Robot is not connected. Attempting to reconnect...")
                self.robot.connect()
                if not self.robot.is_connected:
                    raise RuntimeError("❌ Failed to connect to robot")
            self.robot.send_action(self.robot._urdf_joints_to_action(self.scene.ready_pos_full.joints), safe=True)

            # get the strokes that will be executed this episode
            stroke_l, stroke_r = strokes.strokes[stroke_idx]
            _msg = f"Executing stroke {stroke_idx + 1}/{num_strokes}: left={stroke_l.description}, right={stroke_r.description}"
            log.info(_msg)
            yield {
                'progress': 0.3 + (0.6 * stroke_idx / num_strokes),
                'message': _msg,
            }

            # Per-episode conditioning information is stored in seperate directory
            episode_cond = {}
            episode_cond_dir = os.path.join(self.dataset_dir, f"episode_{stroke_idx:06d}")
            os.makedirs(episode_cond_dir, exist_ok=True)
            log.debug(f"🗃️ Creating episode-specific condition directory at {episode_cond_dir}...")
            episode_cond["stroke_l"] = stroke_l.to_dict()
            episode_cond["stroke_r"] = stroke_r.to_dict()
            if stroke_l.frame_path is not None:
                shutil.copy(stroke_l.frame_path, os.path.join(episode_cond_dir, "stroke_l.png"))
            if stroke_r.frame_path is not None:
                shutil.copy(stroke_r.frame_path, os.path.join(episode_cond_dir, "stroke_r.png"))

            log.info(f"🤖 recording path {stroke_idx} of {num_strokes}")
            for pose_idx in range(self.scene.stroke_length):
                start_loop_t = time.perf_counter()
                
                log.debug(f"pose_idx: {pose_idx}/{self.scene.stroke_length}")
                
                action = atari_teleop.get_action()
                if action.get("red_button", False):
                    raise KeyboardInterrupt()
                if action.get("y", None) is not None:
                    if stroke_l.is_inkdip:
                        inkdip_offset_idx_l += int(action["y"])
                        inkdip_offset_idx_l = min(inkdip_offset_idx_l, self.scene.arms.offset_num - 1)
                        inkdip_offset_idx_l = max(0, inkdip_offset_idx_l)
                    else:
                        offset_idx_l += int(action["y"])
                        offset_idx_l = min(offset_idx_l, self.scene.arms.offset_num - 1)
                        offset_idx_l = max(0, offset_idx_l)
                if action.get("x", None) is not None:
                    if stroke_r.is_inkdip:
                        inkdip_offset_idx_r += int(action["x"])
                        inkdip_offset_idx_r = min(inkdip_offset_idx_r, self.scene.arms.offset_num - 1)
                        inkdip_offset_idx_r = max(0, inkdip_offset_idx_r)
                    else:
                        offset_idx_r += int(action["x"])
                        offset_idx_r = min(offset_idx_r, self.scene.arms.offset_num - 1)
                        offset_idx_r = max(0, offset_idx_r)

                observation = self.robot.get_observation()
                observation_frame = build_dataset_frame(self.dataset.features, observation, prefix="observation")

                _left_first = True # default move left arm first
                if stroke_l.is_inkdip:
                    _offset_idx_l = inkdip_offset_idx_l
                    log.info(f"🎮 left inkdip offset index: {_offset_idx_l}")
                else:
                    _offset_idx_l = offset_idx_l
                    log.info(f"🎮 left offset index: {_offset_idx_l}")
                if stroke_r.is_inkdip:
                    _left_first = False
                    _offset_idx_r = inkdip_offset_idx_r
                    log.info(f"🎮 right inkdip offset index: {_offset_idx_r}")
                else:
                    _offset_idx_r = offset_idx_r
                    log.info(f"🎮 right offset index: {_offset_idx_r}")
                joints = strokebatch.offset_joints(stroke_idx, pose_idx, _offset_idx_l, _offset_idx_r)
                robot_action = self.robot._urdf_joints_to_action(joints)
                if pose_idx in (0, 1):
                    # use slow movements for initial pose and hover pose
                    sent_action = self.robot.send_action(robot_action, self.scene.arms.goal_time_slow, safe=True, left_first=_left_first)
                else:
                    sent_action = self.robot.send_action(robot_action, self.scene.arms.goal_time_fast)

                action_frame = build_dataset_frame(self.dataset.features, sent_action, prefix="action")
                frame = {**observation_frame, **action_frame}
                self.dataset.add_frame(frame, task=f"left: {stroke_l.description}, right: {stroke_r.description}")

                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / self.config.fps - dt_s)

            log_path = os.path.join(logs_dir, f"episode_{stroke_idx:06d}.txt")
            log.info(f"🗃️ Writing episode log to {log_path}")
            with open(log_path, "w") as f:
                f.write(episode_log_buffer.getvalue())

            self.dataset.save_episode(episode_cond=episode_cond)

        logging.getLogger().removeHandler(episode_handler)