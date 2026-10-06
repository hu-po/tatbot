"""Episodes to a LeRobot v3 dataset, through LeRobot's own writer.

The dataset is the FLUX.3 Action contract for this arm: the wrist camera
(and the rig's scene camera, as the SO-101 package pairs a scene view with a
wrist view), six joint positions in radians as state, six absolute joint
commands as action, 30 Hz, one task string. Everything the simulator knows beyond that
(expert mode, stroke and position on it, phantom pose, the episode's appearance draw)
goes to ``labels/episode_XXXXXX.npz`` and ``labels/episode_XXXXXX.json``;
static canonical skin addresses go to ``labels/episode_XXXXXX.surface.npz``
next to the dataset, never into the features a policy trains on.

LeRobot is imported lazily: generation, previews and tests run without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

JOINT_NAMES = [f"joint_{i}.pos" for i in range(6)]
CAMERA_KEY = "observation.images.wrist"
SCENE_KEY = "observation.images.scene"
ROBOT_TYPE = "tatbot_travel_blue"
IMAGE_NAMES = ["height", "width", "channels"]


def features(height: int = 480, width: int = 640, scene_hw: tuple[int, int] | None = None) -> dict:
    """The dataset's features; ``scene_hw`` adds the scene camera's (height, width) stream."""
    out = {CAMERA_KEY: {"dtype": "video", "shape": (height, width, 3), "names": IMAGE_NAMES}}
    if scene_hw is not None:
        out[SCENE_KEY] = {"dtype": "video", "shape": (*scene_hw, 3), "names": IMAGE_NAMES}
    out["observation.state"] = {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES}
    out["action"] = {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES}
    return out


class EpisodeLabels:
    """Per-frame simulator labels for one episode, saved beside the dataset."""

    def __init__(self):
        self.rows: dict[str, list] = {}

    def add(self, labels: dict) -> None:
        for key, value in labels.items():
            self.rows.setdefault(key, []).append(np.asarray(value))

    def save(self, directory: Path, index: int, meta: dict, surface: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"episode_{index:06d}.npz", **{k: np.stack(v) for k, v in self.rows.items()})
        if surface is not None:
            np.savez_compressed(directory / f"episode_{index:06d}.surface.npz", **surface)
        (directory / f"episode_{index:06d}.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")


class DatasetSink:
    """Frames into a LeRobot v3 dataset under ``root`` (created fresh)."""

    def __init__(self, root: Path, repo_id: str, task: str, fps: int = 30, *, vcodec: str | None = None,
                 image_writer_threads: int = 4, scene_hw: tuple[int, int] | None = None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root, self.task, self.scene = Path(root), task, scene_hw is not None
        kwargs = {"vcodec": vcodec} if vcodec else {}
        self.dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=self.root, robot_type=ROBOT_TYPE,
                                             features=features(scene_hw=scene_hw), use_videos=True,
                                             image_writer_threads=image_writer_threads, **kwargs)
        self.labels = EpisodeLabels()

    def add(self, frame) -> None:
        row = {CAMERA_KEY: frame.image, "observation.state": frame.state, "action": frame.action, "task": self.task}
        if self.scene:
            row[SCENE_KEY] = frame.scene
        self.dataset.add_frame(row)
        self.labels.add(frame.labels)

    def end_episode(self, meta: dict, surface: dict | None = None) -> int:
        index = self.dataset.meta.total_episodes
        # Generation already runs one episode stream per worker process, and a pool worker cannot start the
        # process pool LeRobot encodes several cameras with.
        self.dataset.save_episode(parallel_encoding=False)
        self.labels.save(self.root / "labels", index, meta, surface)
        self.labels = EpisodeLabels()
        return index

    def finalize(self) -> None:
        self.dataset.finalize()
