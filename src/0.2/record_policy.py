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
import tyro

log = logging.getLogger('tatbot')

@dataclass
class CLIArgs:
    ckpt_path: str = os.path.expanduser("~/tatbot/outputs/train/2025-06-05/08-54-14_smolvla/checkpoints/last/pretrained_model")
    """Path to the checkpoint of the policy."""
    task: str = "Move slightly up then slightly down"
    """Task to record."""
    debug: bool = False
    """Enable debug logging."""
    dataset_name: str = f"policy-test-{int(time.time())}"
    """Name of the dataset to record."""
    output_dir: str = os.path.expanduser("~/tatbot/output/record")
    """Directory to save the dataset."""
    episode_time_s: float = 10.0
    """Time of each episode."""
    num_episodes: int = 1
    """Number of episodes to record."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""

if __name__ == "__main__":
    args = tyro.cli(CLIArgs)
    if args.debug:
        log.setLevel(logging.DEBUG)
        # logging.getLogger('lerobot').setLevel(logging.DEBUG)
        log.debug("🐛 Debug mode enabled.")
    os.makedirs(args.output_dir, exist_ok=True)
    log.info(f"💾 Saving output to {args.output_dir}")
    log.info(f"🎯 Task: {args.task}")
    cfg = RecordConfig(
        robot=TatbotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=f"hu-po/tatbot-policy-{args.dataset_name}",
            single_task=args.task,
            root=f"{args.output_dir}/{args.dataset_name}",
            fps=10,
            episode_time_s=args.episode_time_s,
            num_episodes=args.num_episodes,
            video=True,
            tags=["tatbot", "wxai", "trossen"],
            push_to_hub=False,
        ),
        policy=SmolVLAConfig(pretrained_path=args.ckpt_path),
        display_data=False,
        play_sounds=True,
        resume=False,
    )
    log.info(pformat(asdict(cfg)))
    lerobot.record.record(cfg)
