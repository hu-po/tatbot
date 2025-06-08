"""

```bash
python -m lerobot.record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, camera_index: 0, width: 640, height: 480}}" \
    --robot.id=black \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --dataset.repo_id=aliberts/record-test \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the cube"
```

"""

import logging
import os
import time
from pprint import pformat
from pathlib import Path
from dataclasses import asdict, dataclass


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hw_to_dataset_features
from lerobot.common.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.common.robots.tatbot.tatbot import Tatbot
from lerobot.lerobot.record import record_loop, RecordConfig, DatasetRecordConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

@dataclass
class VizerConfig:
    seed: int = 42
    """Seed for random behavior."""
    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""
    

class VizerTeleop:

    def __init__(self, config: VizerConfig):
        log.info(f"🌱 Setting random seed to {config.seed}...")
        rng = jax.random.PRNGKey(config.seed)

        log.info("🚀 Starting viser server...")
        server: viser.ViserServer = viser.ViserServer()
        server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

    def get_action(self):
        return {
            "left.joint_0.pos": 0.0,
            "left.joint_1.pos": 0.0,
            "left.joint_2.pos": 0.0,
            "left.joint_3.pos": 0.0,
            "left.joint_4.pos": 0.0,
            "left.joint_5.pos": 0.0,
            "left.gripper.pos": 0.0,
            "right.joint_0.pos": 0.0,
            "right.joint_1.pos": 0.0,
            "right.joint_2.pos": 0.0,
            "right.joint_3.pos": 0.0,
            "right.joint_4.pos": 0.0,
            "right.joint_5.pos": 0.0,
            "right.gripper.pos": 0.0,
        }

    def connect(self):
        pass

    def disconnect(self):
        pass

def record(cfg: RecordConfig):
    logging.basicConfig(level=logging.INFO)
    logging.info(pformat(asdict(cfg)))

    robot = Tatbot(cfg.robot)
    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
    else:
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * (len(getattr(robot, "cameras", [])) or 1),
        )

    robot.connect()

    for episode in range(cfg.dataset.num_episodes):
        logging.info(f"Recording episode {episode+1}/{cfg.dataset.num_episodes}")
        # Use canonical record_loop, but with hardcoded action logic for now
        events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
        teleop = VizerTeleop()
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            teleop=teleop,
            dataset=dataset,
            control_time_s=cfg.dataset.episode_time_s,
            single_task=cfg.dataset.single_task,
            display_data=cfg.display_data,
        )
        dataset.save_episode()
        if events["stop_recording"]:
            break

    robot.disconnect()
    if hasattr(teleop, "disconnect"):
        teleop.disconnect()
    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
    logging.info("Exiting")
    return dataset

if __name__ == "__main__":
    cfg = RecordConfig(
        robot=TatbotConfig(),
        dataset=DatasetRecordConfig(
            repo_id="user/tatbot" + str(int(time.time())),
            single_task="Sleeping",
            fps=10,
            episode_time_s=10,
            num_episodes=1,
            video=False,
            push_to_hub=False,
        ),
        display_data=False,
        play_sounds=False,
        resume=False,
    )
    record(cfg)
