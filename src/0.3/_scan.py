from dataclasses import asdict, dataclass, field
import os

import dacite
import numpy as np
import yaml

from _bot import BotConfig
from _ink import InkConfig
from _log import get_logger

log = get_logger('_scan')

METADATA_FILENAME = "metadata.yaml"
BOT_CONFIG_FILENAME: str = "bot_config.yaml"
INKPALETTE_FILENAME: str = "ink_config.yaml"

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
    
    realsense1_urdf_link_name: str = ""
    realsense2_urdf_link_name: str = ""
    camera1_urdf_link_name: str = ""
    camera2_urdf_link_name: str = ""
    camera3_urdf_link_name: str = ""
    camera4_urdf_link_name: str = ""
    camera5_urdf_link_name: str = ""

    realsense1_fov: float = 0.0
    realsense1_aspect: float = 0.0
    realsense2_fov: float = 0.0
    realsense2_aspect: float = 0.0
    camera1_fov: float = 0.0
    camera1_aspect: float = 0.0
    camera2_fov: float = 0.0
    camera2_aspect: float = 0.0
    camera3_fov: float = 0.0
    camera3_aspect: float = 0.0
    camera4_fov: float = 0.0
    camera4_aspect: float = 0.0
    camera5_fov: float = 0.0
    camera5_aspect: float = 0.0

    def save(self):
        log.info(f"📡💾 Saving scan to {self.dirpath}")
        os.makedirs(self.dirpath, exist_ok=True)
        meta_path = os.path.join(self.dirpath, METADATA_FILENAME)
        log.info(f"📡💾 Saving metadata to {meta_path}")
        self.bot_config.save_yaml(os.path.join(self.dirpath, BOT_CONFIG_FILENAME))
        self.ink_config.save_yaml(os.path.join(self.dirpath, INKPALETTE_FILENAME))
        meta_dict = asdict(self).copy()
        meta_dict.pop('bot_config', None)
        meta_dict.pop('ink_config', None)
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
        scan.ink_config = InkConfig.from_yaml(os.path.join(dirpath, INKPALETTE_FILENAME))
        scan.dirpath = dirpath
        return scan