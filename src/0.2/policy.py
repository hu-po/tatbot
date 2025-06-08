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
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from transformers import AutoProcessor




if __name__ == "__main__":
    cfg = RecordConfig(
        robot=TatbotConfig(),
        dataset=DatasetRecordConfig(
            repo_id="hu-po/tatbot-test" + str(int(time.time())),
            single_task="Grab the red triangle",
            root=os.path.expanduser("~/tatbot/output/policy"),
            fps=10,
            episode_time_s=6,
            num_episodes=2,
            video=True,
            tags=["tatbot", "widowx"],
            push_to_hub=False,
        ),
        policy=SmolVLAConfig(
            pretrained_path=os.path.expanduser("~/tatbot/outputs/train/2025-06-05/08-54-14_smolvla/checkpoints/last/pretrained_model")
        ),
        display_data=False,
        play_sounds=True,
        resume=False,
    )
    lerobot.record.record(cfg)
