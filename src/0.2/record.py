"""

```bash
python -m lerobot.record \
    --robot.type=tatbot \
    --robot.cameras="{wrist: {type: intelrealsense, camera_index: 0, width: 640, height: 480}}" \
    --robot.id=black \
    --dataset.repo_id=$HF_USERNAME/record-test \
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

import lerobot.record
from lerobot.common.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.record import RecordConfig, DatasetRecordConfig
from lerobot.common.teleoperators.config import TeleoperatorConfig

# HACK: monkeypatch to use Vizier Web Teleoperator
from vizer_teleop import VizerTeleop, VizerTeleopConfig

original_make_teleoperator_from_config = lerobot.record.make_teleoperator_from_config


def make_teleoperator_from_config(config: TeleoperatorConfig):
    if isinstance(config, VizerTeleopConfig):
        return VizerTeleop(config)
    return original_make_teleoperator_from_config(config)


lerobot.record.make_teleoperator_from_config = make_teleoperator_from_config

if __name__ == "__main__":
    cfg = RecordConfig(
        robot=TatbotConfig(),
        dataset=DatasetRecordConfig(
            repo_id="hu-po/tatbot-test" + str(int(time.time())),
            single_task="Grab the red triangle",
            root=os.path.expanduser("~/tatbot/output/teleop"),
            fps=10,
            episode_time_s=6,
            num_episodes=2,
            video=True,
            tags=["tatbot", "widowx"],
            push_to_hub=False,
        ),
        teleop=VizerTeleopConfig(),
        display_data=False,
        play_sounds=True,
        resume=False,
    )
    lerobot.record.record(cfg)
