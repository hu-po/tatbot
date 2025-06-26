from dataclasses import asdict, dataclass, field
import os

import dacite
import numpy as np
import yaml

from _bot import BotConfig
from _cam import CameraExtrinsics, CameraIntrinsics
from _ink import InkConfig
from _log import get_logger
from _tag import TagConfig, TagPose

log = get_logger('_scan')

METADATA_FILENAME: str = "metadata.yaml"
BOT_CONFIG_FILENAME: str = "bot_config.yaml"
INK_CONFIG_FILENAME: str = "ink_config.yaml"
TAG_CONFIG_FILENAME: str = "tag_config.yaml"

@dataclass
class Scan:
    name: str = "scan"
    """Name of the scan."""

    bot_config: BotConfig = field(default_factory=BotConfig)
    """Bot configuration to use for the scan."""
    ink_config: InkConfig = field(default_factory=InkConfig)
    """Config containig InkCaps and palette position."""
    tag_config: TagConfig = field(default_factory=TagConfig)
    """Config containing AprilTag parameters."""

    optical_frame_urdf_link_names: dict[str, str] = field(default_factory=lambda: {
        "realsense1": "realsense1_color_optical_frame",
        "realsense2": "realsense2_color_optical_frame",
        "camera1": "camera1_optical_frame",
        "camera2": "camera2_optical_frame",
        "camera3": "camera3_optical_frame",
        "camera4": "camera4_optical_frame",
        "camera5": "camera5_optical_frame",
    })
    """URDF link names for each camera's optical frame."""

    tag_poses: dict[str, dict[int, TagPose]] = field(default_factory=dict)
    """Tag poses for each tag."""

    extrinsics: dict[str, CameraExtrinsics] = field(default_factory=dict)
    """Extrinsics for each camera."""

    intrinsics: dict[str, CameraIntrinsics] = field(default_factory=lambda: {
        "realsense1": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "realsense2": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "camera1": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "camera2": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "camera3": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "camera4": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
        "camera5": CameraIntrinsics(
            fov=1.0,
            aspect=1.0,
            fx=1.0,
            fy=1.0,
            ppx=1.0,
            ppy=1.0,
        ),
    })
    """Intrinsics for each camera."""

    def save(self, dirpath: str):
        dirpath = os.path.expanduser(dirpath)
        log.info(f"📡💾 Saving scan to {dirpath}")
        os.makedirs(dirpath, exist_ok=True)
        meta_path = os.path.join(dirpath, METADATA_FILENAME)
        log.info(f"📡💾 Saving metadata to {meta_path}")
        self.bot_config.save_yaml(os.path.join(dirpath, BOT_CONFIG_FILENAME))
        self.ink_config.save_yaml(os.path.join(dirpath, INK_CONFIG_FILENAME))
        self.tag_config.save_yaml(os.path.join(dirpath, TAG_CONFIG_FILENAME))
        meta_dict = asdict(self).copy()
        meta_dict.pop('bot_config', None)
        meta_dict.pop('ink_config', None)
        meta_dict.pop('tag_config', None)
        with open(meta_path, "w") as f:
            yaml.safe_dump(meta_dict, f)

    @classmethod
    def from_yaml(cls, dirpath: str) -> "Scan":
        dirpath = os.path.expanduser(dirpath)
        log.info(f"📡💾 Loading scan from {dirpath}...")
        filepath = os.path.join(dirpath, METADATA_FILENAME)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        scan = dacite.from_dict(cls, data, config=dacite.Config(type_hooks={np.ndarray: np.array}))
        scan.bot_config = BotConfig.from_yaml(os.path.join(dirpath, BOT_CONFIG_FILENAME))
        scan.ink_config = InkConfig.from_yaml(os.path.join(dirpath, INK_CONFIG_FILENAME))
        scan.tag_config = TagConfig.from_yaml(os.path.join(dirpath, TAG_CONFIG_FILENAME))
        return scan