import logging
import os
import time
from dataclasses import dataclass

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.utils.control_utils import (
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)

from tatbot.ops.base import BaseOp, BaseOpConfig
from tatbot.utils.log import TIME_FORMAT, get_logger

log = get_logger("ops.record", "🤖")


@dataclass
class RecordOpConfig(BaseOpConfig):
    """Robot Operation that Records a LeRobot Dataset."""

    output_dir: str = "~/tatbot/nfs/recordings"
    """Directory to save the dataset."""

    hf_username: str = "tatbot"
    """Hugging Face username."""
    dataset_name: str | None = None
    """Dataset will be saved to Hugging Face Hub repository ID, e.g. 'hf_username/dataset_name'."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""
    tags: tuple[str, ...] = ("tatbot", "wxai", "trossen")
    """Tags to add to the dataset on Hugging Face."""
    private: bool = False
    """Whether to push the dataset to a private repository."""
    fps: int = 16
    """Frames per second."""
    resume: bool = False
    """If true, resumes recording from the last episode, dataset name must match."""


class RecordOp(BaseOp):

    op_name: str = "record"

    def __init__(self, config: RecordOpConfig):
        super().__init__(config)
        if config.debug:
            logging.getLogger("lerobot").setLevel(logging.DEBUG)
        self.dataset_dir: str | None = None
        self.dataset: LeRobotDataset | None = None
        self.robot: Robot | None = None
        self.num_camera_threads: int = 0

    def make_robot(self) -> Robot:
        """Make a robot from the config."""
        return make_robot_from_config(
            TatbotConfig(
                ip_address_l=self.scene.arms.ip_address_l,
                ip_address_r=self.scene.arms.ip_address_r,
                arm_l_config_filepath=self.scene.arms.arm_l_config_filepath,
                arm_r_config_filepath=self.scene.arms.arm_r_config_filepath,
                goal_time=self.scene.arms.goal_time_slow,
                connection_timeout=self.scene.arms.connection_timeout,
                home_pos_l=self.scene.sleep_pos_l.joints,
                home_pos_r=self.scene.sleep_pos_r.joints,
                # base record op does not use cameras
                rs_cameras={},
                ip_cameras={},
            )
        )

    def cleanup(self):
        if self.robot is not None:
            self.robot.disconnect()

    async def _run(self):
        log.warning("🤖 _run is not implemented")
        return

    async def run(self):
        output_dir = os.path.expanduser(self.config.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        _msg = f"🗃️ Creating output directory at {output_dir}..."
        log.info(_msg)
        yield {
            'progress': 0.01,
            'message': _msg,
        }

        dataset_name = self.config.dataset_name or f"{self.op_name}-{self.scene.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
        self.dataset_dir = f"{output_dir}/{dataset_name}"
        _msg = f"🗃️ Creating dataset directory at {self.dataset_dir}..."
        log.info(_msg)
        os.makedirs(self.dataset_dir, exist_ok=True)
        yield {
            'progress': 0.02,
            'message': _msg,
        }

        # copy the scene yaml to the output directory
        scene_path = os.path.join(self.dataset_dir, "scene.yaml")
        self.scene.to_yaml(scene_path)

        _msg = "🤖 Creating robot from config..."
        log.info(_msg)
        yield {
            'progress': 0.05,
            'message': _msg,
        }
        self.robot = self.make_robot()

        _msg = "🤖 Connecting to robot..."
        log.info(_msg)
        yield {
            'progress': 0.06,
            'message': _msg,
        }
        self.robot.connect()
        self.num_camera_threads = 0
        if hasattr(self.robot, "rs_cameras") and len(self.robot.rs_cameras) > 0:
            self.num_camera_threads += 4 * len(self.robot.rs_cameras)
        if hasattr(self.robot, "ip_cameras") and len(self.robot.ip_cameras) > 0:
            self.num_camera_threads += 4 * len(self.robot.ip_cameras)
        _msg = f"🤖 Connected to robot with {self.num_camera_threads} camera threads..."
        log.info(_msg)
        yield {
            'progress': 0.061,
            'message': _msg,
        }
        
        action_features = hw_to_dataset_features(self.robot.action_features, "action", True)
        obs_features = hw_to_dataset_features(self.robot.observation_features, "observation", True)
        dataset_features = {**action_features, **obs_features}
        repo_id = f"{self.config.hf_username}/{dataset_name}"
        if self.config.resume:
            _msg = f"📦🤗 Resuming LeRobot dataset at {repo_id}"
            self.dataset = LeRobotDataset(repo_id, root=self.dataset_dir)
            if self.num_camera_threads > 0:
                self.dataset.start_image_writer(num_processes=0, num_threads=self.num_camera_threads)
            sanity_check_dataset_robot_compatibility(self.dataset, self.robot, self.config.fps, dataset_features)
        else:
            _msg = f"📦🤗 Created new LeRobot dataset at {repo_id}"
            sanity_check_dataset_name(repo_id, None)
            self.dataset = LeRobotDataset.create(
                repo_id,
                self.config.fps,
                root=self.dataset_dir,
                robot_type=self.robot.name,
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=self.num_camera_threads,
            )
        log.info(_msg)
        yield {
            'progress': 0.07,
            'message': _msg,
        }

        _msg = "🤖 Sending robot to ready position..."
        log.info(_msg)
        yield {
            'progress': 0.08,
            'message': _msg,
        }
        self.robot.send_action(self.robot._urdf_joints_to_action(self.scene.ready_pos_full), safe=True)
        
        async for progress_update in self._run():
            yield progress_update

        _msg = "🤖 Sending robot to ready position..."
        log.info(_msg)
        yield {
            'progress': 0.998,
            'message': _msg,
        }
        self.robot.send_action(self.robot._urdf_joints_to_action(self.scene.ready_pos_full), safe=True)

        if self.config.push_to_hub:
            _msg = "📦🤗 Pushing dataset to Hugging Face Hub..."
            log.info(_msg)
            yield {
                'progress': 0.999,
                'message': _msg,
            }
            self.dataset.push_to_hub(tags=list(self.config.tags), private=self.config.private)
            _msg = "✅ Pushed dataset to Hugging Face Hub"
            log.info(_msg)
            yield {
                'progress': 0.9991,
                'message': _msg,
            }

        _msg = f"✅ Robot operation {self.op_name} completed"
        log.info(_msg)
        yield {
            'progress': 1.0,
            'message': _msg,
        }
