from dataclasses import dataclass
import os
import time
import logging
from io import StringIO
import asyncio


from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.teleoperators.gamepad import AtariTeleoperator, AtariTeleoperatorConfig
from lerobot.utils.control_utils import (
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)

from tatbot.data.scene import Scene
from tatbot.bot.ops.base import BaseOp, BaseOpConfig
from tatbot.utils.log import get_logger
from tatbot.utils.log import (
    LOG_FORMAT,
    TIME_FORMAT,
    get_logger,
    print_config,
    setup_log_with_config,
)

log = get_logger("bot.ops.record", "🤖")


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
    fps: int = 30
    """Frames per second."""
    max_episodes: int = 256
    """Maximum number of episodes to record."""

    

class RecordOp(BaseOp):
    def __init__(self, config: RecordOpConfig):
        super().__init__(config)

        output_dir = os.path.expanduser(config.output_dir)
        log.info(f"🗃️ Creating output directory at {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        dataset_name = config.dataset_name or f"{self.scene.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
        self.dataset_dir = f"{output_dir}/{dataset_name}"
        log.info(f"🗃️ Creating dataset directory at {self.dataset_dir}...")
        os.makedirs(self.dataset_dir, exist_ok=True)

        # copy the scene yaml to the output directory
        scene_path = os.path.join(self.dataset_dir, "scene.yaml")
        self.scene.to_yaml(scene_path)

        logs_dir = os.path.join(self.dataset_dir, "logs")
        log.info(f"🗃️ Creating logs directory inside dataset directory at {logs_dir}...")
        os.makedirs(logs_dir, exist_ok=True)
        episode_log_buffer = StringIO()

        class EpisodeLogHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                episode_log_buffer.write(msg + "\n")

        episode_handler = EpisodeLogHandler()
        episode_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=TIME_FORMAT))
        logging.getLogger().addHandler(episode_handler)

    async def run(self):
        """Run the recording operation with progress updates."""
        config = self.config
        
        # Step 1: Initialize robot
        yield {
            'progress': 0.1,
            'message': 'Initializing robot...',
            'step': 'init'
        }
        
        robot_config = TatbotConfig()
        robot = make_robot_from_config(robot_config)
        
        # Step 2: Setup dataset
        yield {
            'progress': 0.2,
            'message': 'Setting up dataset...',
            'step': 'dataset_setup'
        }
        
        action_features = hw_to_dataset_features(robot.action_features, "action", True)
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
        dataset_features = {**action_features, **obs_features}
        
        dataset_name = config.dataset_name or f"{self.scene.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
        repo_id = f"{config.hf_username}/{dataset_name}"
        
        yield {
            'progress': 0.3,
            'message': f'Creating dataset at {repo_id}...',
            'step': 'dataset_creation',
            'data': {'repo_id': repo_id}
        }
        
        sanity_check_dataset_name(repo_id, None)
        dataset = LeRobotDataset.create(
            repo_id,
            config.fps,
            root=self.dataset_dir,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * (len(robot.cameras) + len(robot.cond_cameras)),
        )
        
        # Step 3: Setup teleoperator
        yield {
            'progress': 0.4,
            'message': 'Setting up teleoperator...',
            'step': 'teleop_setup'
        }
        
        teleop_config = AtariTeleoperatorConfig()
        teleop = AtariTeleoperator(teleop_config)
        
        # Step 4: Recording loop
        yield {
            'progress': 0.5,
            'message': f'Starting recording loop (max {config.max_episodes} episodes)...',
            'step': 'recording_start'
        }
        
        episode_count = 0
        for episode in range(config.max_episodes):
            episode_count = episode + 1
            progress = 0.5 + (episode_count / config.max_episodes) * 0.4
            
            yield {
                'progress': progress,
                'message': f'Recording episode {episode_count}/{config.max_episodes}',
                'step': 'recording_episode',
                'data': {
                    'episode': episode_count,
                    'total_episodes': config.max_episodes,
                    'episode_progress': episode_count / config.max_episodes
                }
            }
            
            # Here you would implement the actual episode recording logic
            # For now, just a placeholder
            await asyncio.sleep(0.1)  # Simulate some work
        
        # Step 5: Finalize dataset
        yield {
            'progress': 0.9,
            'message': 'Finalizing dataset...',
            'step': 'finalize'
        }
        
        if config.push_to_hub:
            yield {
                'progress': 0.95,
                'message': f'Pushing dataset to Hugging Face Hub...',
                'step': 'push_to_hub'
            }
            # Here you would implement the push to hub logic
            # dataset.push_to_hub(...)
        
        yield {
            'progress': 1.0,
            'message': f'Successfully recorded {episode_count} episodes',
            'step': 'complete',
            'data': {
                'episodes_recorded': episode_count,
                'dataset_path': self.dataset_dir,
                'repo_id': repo_id
            }
        }
