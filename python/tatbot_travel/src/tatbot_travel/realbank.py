"""Real pixels for the simulator: crops of the rig's room, table and ink, from a bank outside the repo.

Checkpoints trained on rendered rooms and flash planned the same motion on the rig whatever they saw: to the
policy, real frames looked like nothing it had trained on. The bank's crops -- what each camera sees around
the rig, the table, violet ink cut from the practice arm -- go onto the simulator's walls, clutter, table and
forearm, beside the procedural textures. The bank shows the lab, so it lives outside the repo:
``TATBOT_TRAVEL_REAL_ASSETS`` names its directory (``wrist_bg/``, ``scene_bg/`` and ``table/`` of JPEGs,
``ink/`` of RGBA PNGs whose alpha is the ink). Without it the generator draws procedural textures only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ENV = "TATBOT_TRAVEL_REAL_ASSETS"
KINDS = ("wrist_bg", "scene_bg", "table", "ink")
SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class RealBank:
    files: dict[str, tuple[Path, ...]]

    def has(self, kind: str) -> bool:
        return bool(self.files.get(kind))

    def _pick(self, rng: np.random.Generator, kind: str) -> Path:
        return self.files[kind][int(rng.integers(len(self.files[kind])))]

    def texture(self, rng: np.random.Generator, kind: str, size: int = 256) -> np.ndarray:
        """An RGB crop of ``kind``, turned and flipped at random (a texture's roll on a wall is arbitrary)."""
        rgb = cv2.cvtColor(cv2.imread(str(self._pick(rng, kind))), cv2.COLOR_BGR2RGB)
        rgb = np.rot90(rgb, int(rng.integers(4)))
        if rng.random() < 0.5:
            rgb = rgb[:, ::-1]
        return cv2.resize(np.ascontiguousarray(rgb), (size, size), interpolation=cv2.INTER_AREA)

    def ink(self, rng: np.random.Generator) -> np.ndarray:
        """An ink patch's coverage (float32, 0..1), turned and flipped at random."""
        rgba = cv2.imread(str(self._pick(rng, "ink")), cv2.IMREAD_UNCHANGED)
        alpha = np.rot90(rgba[..., 3].astype(np.float32) / 255.0, int(rng.integers(4)))
        return np.ascontiguousarray(alpha[:, ::-1] if rng.random() < 0.5 else alpha)


@lru_cache(maxsize=4)
def load(root: str | None = None) -> RealBank | None:
    """The bank under ``root`` (default ``$TATBOT_TRAVEL_REAL_ASSETS``), or None when there is none."""
    root = root or os.environ.get(ENV)
    if not root or not Path(root).is_dir():
        return None
    base = Path(root)
    files = {kind: tuple(sorted(p for p in (base / kind).glob("*") if p.suffix.lower() in SUFFIXES))
             for kind in KINDS if (base / kind).is_dir()}
    return RealBank(files=files)
