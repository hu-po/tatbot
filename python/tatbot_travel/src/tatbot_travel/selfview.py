"""The arm's own pen cradle, as the wrist camera really sees it.

The pen cradle and its clamp cap on the blue arm's carriage sit a few
centimetres from the D405 and fill the lower left of every frame. They are
rigid in the camera frame (the carriage is held at rest), so instead of
rendering a CAD mesh that does not match the installed print, the generator
composites real pixels: layers cut from wrist captures in two rooms, keeping
the pixels that stayed put while the background changed. Nothing in the scene
can come between the camera and the cradle, so drawing it last is also the
correct occlusion order.

The layers are 640x480 RGBA in the real stream's pixel grid. The captures
showed a strip of blue tape the cradle no longer carries; on 26 Sep it was
replaced with the rig's own foam (a raw wrist frame's pixels, levelled and
coloured to the foam around it). Recapture the layers whenever the cradle, the
camera mount or the carriage rest changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

LAYERS = ("selfview-left-a.png", "selfview-left-b.png")


@dataclass(frozen=True)
class SelfViewLayer:
    rgb: np.ndarray  # (H, W, 3) float32, 0..255
    alpha: np.ndarray  # (H, W, 1) float32, 0..1


@lru_cache(maxsize=4)
def load_layers(root: str | None = None) -> tuple[SelfViewLayer, ...]:
    layers = []
    files = sorted(Path(root).glob("*.png")) if root else [resources.files("tatbot_travel.data").joinpath(n)
                                                         for n in LAYERS]
    if not files:
        raise ValueError(f"no self-view layers in {root}")
    for path in files:
        data = path.read_bytes()
        bgra = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.shape != (480, 640, 4) or not bgra[..., 3].any():
            raise ValueError(f"self-view layer must be nonempty 640x480 RGBA: {path}")
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA).astype(np.float32)
        layers.append(SelfViewLayer(rgb=rgba[..., :3], alpha=rgba[..., 3:] / 255.0))
    return tuple(layers)


@dataclass(frozen=True)
class SelfViewLook:
    """Per-episode appearance of the cradle layer."""

    layer: int
    gain: float
    tint: tuple[float, float, float]
    root: str | None = None

    @classmethod
    def sample(cls, rng: np.random.Generator, gain_range=(0.55, 1.5), tint_frac=0.08,
               root: str | None = None) -> SelfViewLook:
        return cls(layer=int(rng.integers(len(load_layers(root)))), gain=float(rng.uniform(*gain_range)),
                   tint=tuple(float(t) for t in 1.0 + rng.uniform(-tint_frac, tint_frac, 3)), root=root)


class Compositor:
    """One episode's cradle layer, premultiplied and cropped to where it has coverage."""

    def __init__(self, look: SelfViewLook):
        layer = load_layers(look.root)[look.layer]
        rows, cols = np.nonzero(layer.alpha[..., 0] > 0.0)
        self.box = (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)
        y0, y1, x0, x1 = self.box
        alpha = layer.alpha[y0:y1, x0:x1]
        tint = np.asarray(look.tint, dtype=np.float32) * look.gain
        self.premultiplied = (layer.rgb[y0:y1, x0:x1] * tint * alpha).astype(np.float32)
        self.keep = (1.0 - alpha).astype(np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Draw the cradle over a rendered frame (in place on a copy)."""
        y0, y1, x0, x1 = self.box
        out = image.copy()
        roi = out[y0:y1, x0:x1].astype(np.float32) * self.keep + self.premultiplied
        out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
        return out


def composite(image: np.ndarray, look: SelfViewLook) -> np.ndarray:
    """One-off composite (previews and calibration views)."""
    return Compositor(look)(image)
