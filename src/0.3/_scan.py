from dataclasses import asdict, dataclass, field
import os

import dacite
import numpy as np
import yaml

from _bot import BotConfig
from _cam import CameraIntrinsics
from _ink import InkConfig
from _log import get_logger
from _tag import TagConfig

log = get_logger('_scan')

METADATA_FILENAME: str = "metadata.yaml"
BOT_CONFIG_FILENAME: str = "bot_config.yaml"
INK_CONFIG_FILENAME: str = "ink_config.yaml"
TAG_CONFIG_FILENAME: str = "tag_config.yaml"

@dataclass
class Scan:
    name: str = "scan"
    """Name of the scan."""

    dirpath: str = ""
    """Path to the directory containing the scan files."""

    bot_config: BotConfig = field(default_factory=BotConfig)
    """Bot configuration to use for the scan."""
    ink_config: InkConfig = field(default_factory=InkConfig)
    """Config containig InkCaps and palette position."""
    tag_config: TagConfig = field(default_factory=TagConfig)
    """Config containing AprilTag parameters."""
    
    realsense1_urdf_link_name: str = ""
    realsense2_urdf_link_name: str = ""
    camera1_urdf_link_name: str = ""
    camera2_urdf_link_name: str = ""
    camera3_urdf_link_name: str = ""
    camera4_urdf_link_name: str = ""
    camera5_urdf_link_name: str = ""

    intrinsics: dict[str, CameraIntrinsics] = field(default_factory={
        "realsense1": CameraIntrinsics(),
        "realsense2": CameraIntrinsics(),
        "camera1": CameraIntrinsics(),
        "camera2": CameraIntrinsics(),
        "camera3": CameraIntrinsics(),
        "camera4": CameraIntrinsics(),
        "camera5": CameraIntrinsics(),
    })
    """Intrinsics for each camera."""

    def save(self):
        log.info(f"📡💾 Saving scan to {self.dirpath}")
        os.makedirs(self.dirpath, exist_ok=True)
        meta_path = os.path.join(self.dirpath, METADATA_FILENAME)
        log.info(f"📡💾 Saving metadata to {meta_path}")
        self.bot_config.save_yaml(os.path.join(self.dirpath, BOT_CONFIG_FILENAME))
        self.ink_config.save_yaml(os.path.join(self.dirpath, INK_CONFIG_FILENAME))
        self.tag_config.save_yaml(os.path.join(self.dirpath, TAG_CONFIG_FILENAME))
        meta_dict = asdict(self).copy()
        meta_dict.pop('bot_config', None)
        meta_dict.pop('ink_config', None)
        meta_dict.pop('tag_config', None)
        with open(meta_path, "w") as f:
            yaml.safe_dump(meta_dict, f)

    @classmethod
    def from_yaml(cls, dirpath: str) -> "Scan":
        log.info(f"📡💾 Loading scan from {dirpath}...")
        filepath = os.path.join(dirpath, METADATA_FILENAME)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        scan = dacite.from_dict(cls, data, config=dacite.Config(type_hooks={np.ndarray: np.array}))
        scan.bot_config = BotConfig.from_yaml(os.path.join(dirpath, BOT_CONFIG_FILENAME))
        scan.ink_config = InkConfig.from_yaml(os.path.join(dirpath, INK_CONFIG_FILENAME))
        scan.tag_config = TagConfig.from_yaml(os.path.join(dirpath, TAG_CONFIG_FILENAME))
        scan.dirpath = dirpath
        return scan