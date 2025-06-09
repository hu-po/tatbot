"""
> cd ~/tatbot/src/0.2
> git pull
> deactivate && rm -rf .venv && rm uv.lock
> uv venv && source .venv/bin/activate && uv pip install .
> DISPLAY=:0 uv run record.py --debug

[esc] stop recording
[left arrow] rerecord the last episode
[right arrow] exit recording loop
"""

import logging
import os
import time
from dataclasses import dataclass

import lerobot.record
from lerobot.common.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.record import RecordConfig, DatasetRecordConfig
from lerobot.common.teleoperators.config import TeleoperatorConfig
import tyro

@dataclass
class CLIArgs:
    debug: bool = False
    """Enable debug logging."""
    teleop: str = "iktargets"
    """Type of custom teleoperator to use, one of: iktarget, toolpath"""
    dataset_name: str = f"test-{int(time.time())}"
    """Name of the dataset to record."""
    output_dir: str = os.path.expanduser("~/tatbot/output/record")
    """Directory to save the dataset."""
    episode_time_s: float = 5.0
    """Time of each episode."""
    num_episodes: int = 2
    """Number of episodes to record."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""

# HACK: monkeypatch custom teleoperators into lerobot record types
from teleop_iktarget import IKTargetTeleop, IKTargetTeleopConfig
from teleop_toolpath import ToolpathTeleop, ToolpathTeleopConfig

original_make_teleoperator_from_config = lerobot.record.make_teleoperator_from_config

def make_teleoperator_from_config(config: TeleoperatorConfig):
    if isinstance(config, IKTargetTeleopConfig):
        return IKTargetTeleop(config)
    elif isinstance(config, ToolpathTeleopConfig):
        return ToolpathTeleop(config)
    return original_make_teleoperator_from_config(config)


lerobot.record.make_teleoperator_from_config = make_teleoperator_from_config

if __name__ == "__main__":
    args = tyro.cli(CLIArgs)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.debug_mode:
        logging.info("🐛 Debug mode enabled.")
        logging.basicConfig(level=logging.DEBUG)
    if args.teleop == "iktarget":
        logging.info("🎮 Using IKTargetTeleop.")
        teleop_config = IKTargetTeleopConfig()
    elif args.teleop == "toolpath":
        logging.info("🎮 Using ToolpathTeleop.")
        teleop_config = ToolpathTeleopConfig()
    else:
        raise ValueError(f"Invalid teleoperator: {args.teleop}")

    cfg = RecordConfig(
        robot=TatbotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=f"hu-po/tatbot-{args.dataset_name}",
            single_task="Grab the red triangle",
            root=f"{args.output_dir}/{args.dataset_name}",
            fps=10,
            episode_time_s=args.episode_time_s,
            num_episodes=args.num_episodes,
            video=True,
            tags=["tatbot", "wxai", "trossen"],
            push_to_hub=args.push_to_hub,
        ),
        teleop=teleop_config,
        display_data=True,
        play_sounds=True,
        resume=False,
    )
    lerobot.record.record(cfg)
