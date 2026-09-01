"""Deterministic policy-facing encoding for Tatbot D405 depth maps.

LeRobot stores Tatbot depth as one-channel millimetres. Vision-language
backbones expect ordinary three-channel images, and GR00T's image conversion
would clamp raw millimetres to an almost entirely white image. ``depth-v1``
keeps range, validity, and local surface relief explicit in uint8 channels.

This module deliberately depends only on NumPy. Training loads this exact file
from the repository, while the robot plugin imports it normally, so offline
and live observations cannot drift into two implementations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DEPTH_ENCODING_VERSION = "depth-v1"


@dataclass(frozen=True)
class DepthEncodingConfig:
    """Frozen ``depth-v1`` semantics, in millimetres."""

    version: str = DEPTH_ENCODING_VERSION
    near_mm: float = 80.0
    far_mm: float = 350.0
    invalid_at_or_below_mm: float = 10.0
    local_relief_mm: float = 25.0

    def validate(self) -> None:
        if self.version != DEPTH_ENCODING_VERSION:
            raise ValueError(f"unsupported depth encoding: {self.version}")
        if not 0 < self.near_mm < self.far_mm:
            raise ValueError("depth encoding requires 0 < near_mm < far_mm")
        if not 0 <= self.invalid_at_or_below_mm < self.near_mm:
            raise ValueError("invalid depth threshold must be below near_mm")
        if self.local_relief_mm <= 0:
            raise ValueError("local_relief_mm must be positive")

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


DEFAULT_DEPTH_ENCODING = DepthEncodingConfig()


def _depth_plane(depth_mm: np.ndarray) -> np.ndarray:
    value = np.asarray(depth_mm)
    if value.ndim == 2:
        return value.astype(np.float32, copy=False)
    if value.ndim == 3 and value.shape[-1] == 1:
        return value[..., 0].astype(np.float32, copy=False)
    if value.ndim == 3 and value.shape[0] == 1:
        return value[0].astype(np.float32, copy=False)
    raise ValueError(f"expected HxW, HxWx1, or 1xHxW depth; got {value.shape}")


def encode_depth_mm(
    depth_mm: np.ndarray,
    config: DepthEncodingConfig = DEFAULT_DEPTH_ENCODING,
) -> np.ndarray:
    """Return an ``H x W x 3`` uint8 policy image.

    Channels are:

    0. proximity: near is bright, far is dark;
    1. validity: valid depth is white, dropout/invalid is black;
    2. local relief: signed difference from valid cardinal neighbours, with
       zero relief at mid-grey.

    Invalid pixels are zero in every channel. Neighbours that are invalid do
    not manufacture an edge; they fall back to the centre pixel.
    """

    config.validate()
    depth = _depth_plane(depth_mm)
    valid = np.isfinite(depth) & (depth > config.invalid_at_or_below_mm)
    clipped = np.clip(depth, config.near_mm, config.far_mm)
    span = config.far_mm - config.near_mm
    proximity = 1.0 - (clipped - config.near_mm) / span

    padded_depth = np.pad(clipped, 1, mode="edge")
    padded_valid = np.pad(valid, 1, mode="edge")
    neighbours = []
    neighbour_valid = []
    for rows, cols in ((slice(0, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
                       (slice(1, -1), slice(0, -2)), (slice(1, -1), slice(2, None))):
        neighbours.append(padded_depth[rows, cols])
        neighbour_valid.append(padded_valid[rows, cols])
    neighbour_sum = np.zeros_like(clipped, dtype=np.float32)
    neighbour_count = np.zeros_like(clipped, dtype=np.float32)
    for neighbour, is_valid in zip(neighbours, neighbour_valid, strict=True):
        neighbour_sum += np.where(is_valid, neighbour, 0.0)
        neighbour_count += is_valid
    local_mean = np.where(
        neighbour_count > 0,
        neighbour_sum / np.maximum(neighbour_count, 1.0),
        clipped,
    )
    relief = 0.5 + (clipped - local_mean) / (2.0 * config.local_relief_mm)
    relief = np.clip(relief, 0.0, 1.0)

    encoded = np.stack((proximity, valid.astype(np.float32), relief), axis=-1)
    encoded[~valid] = 0.0
    return np.rint(encoded * 255.0).astype(np.uint8)

