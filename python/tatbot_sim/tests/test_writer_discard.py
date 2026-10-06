"""close_batch(keep=...) drops an episode and keeps the dataset contiguous.

2026-09-03: a blank or unreachable demonstration used to be written with a
warning. The generator now discards it, and LeRobot v3 needs episode indices
without gaps, so the kept episodes after a dropped one must be renumbered —
rows, meta and video files alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tatbot_sim.lerobot_writer import LeRobotWriter


def _frames(b: int, size: int) -> dict[str, np.ndarray]:
    return {
        cam: np.full((b, size, size, 3), 128, dtype=np.uint8)
        for cam in ("wrist_upper", "wrist_lower")
    }


def test_dropped_episode_is_deleted_and_survivors_are_renumbered(tmp_path):
    writer = LeRobotWriter(str(tmp_path / "ds"), cameras=("wrist_upper", "wrist_lower"), image_size=(64, 64), codec="libx264", preset="ultrafast")
    writer.open_batch(3, tasks=["a", "b", "c"])
    for _ in range(4):
        writer.add_steps(np.zeros((3, 7), np.float32), np.zeros((3, 14), np.float32), _frames(3, 64))
    kept = writer.close_batch(keep=[True, False, True])
    assert kept == [0, None, 1]
    writer.open_batch(1, tasks=["d"])
    for _ in range(2):
        writer.add_steps(np.zeros((1, 7), np.float32), np.zeros((1, 14), np.float32), _frames(1, 64))
    assert writer.close_batch() == [2]
    writer.finalize()

    base = tmp_path / "ds"
    videos = sorted(p.name for p in (base / "videos" / "observation.images.wrist_upper" / "chunk-000").iterdir())
    assert videos == ["file-000.mp4", "file-001.mp4", "file-002.mp4"]
    episodes = pd.read_parquet(base / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert episodes["episode_index"].tolist() == [0, 1, 2]
    assert episodes["length"].tolist() == [4, 4, 2]
    assert [tasks[0] for tasks in episodes["tasks"]] == ["a", "c", "d"]
    # LeRobot's episode editors assert length == (to - from) * fps: the window
    # is the video's duration, n/fps, not the last frame's timestamp.
    video = "videos/observation.images.wrist_upper"
    window = (episodes[f"{video}/to_timestamp"] - episodes[f"{video}/from_timestamp"]) * writer.fps
    assert np.round(window).astype(int).tolist() == episodes["length"].tolist()
    rows = pd.read_parquet(base / "data" / "chunk-000" / "file-000.parquet")
    assert rows["episode_index"].tolist() == [0] * 4 + [1] * 4 + [2] * 2
    assert rows["index"].tolist() == list(range(10))
