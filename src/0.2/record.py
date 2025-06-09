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
import tyro

@dataclass
class CLIArgs:
    dataset_name: str = f"test-{int(time.time())}"
    """Name of the dataset to record."""
    output_dir: str = os.path.expanduser("~/tatbot/output/record")
    """Directory to save the dataset."""

# HACK: monkeypatch to use Vizier Web Teleoperator
from vizer_teleop import VizerTeleop, VizerTeleopConfig

original_make_teleoperator_from_config = lerobot.record.make_teleoperator_from_config


def make_teleoperator_from_config(config: TeleoperatorConfig):
    if isinstance(config, VizerTeleopConfig):
        return VizerTeleop(config)
    return original_make_teleoperator_from_config(config)


lerobot.record.make_teleoperator_from_config = make_teleoperator_from_config

if __name__ == "__main__":
    args = tyro.cli(CLIArgs)
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = RecordConfig(
        robot=TatbotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=f"hu-po/{args.dataset_name}",
            single_task="Grab the red triangle",
            root=f"{args.output_dir}/{args.dataset_name}",
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
