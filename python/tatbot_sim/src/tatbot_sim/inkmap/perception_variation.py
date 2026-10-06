"""Batch-invariant Inkmap perception variation and leakage-safe splits."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

AXIS_PHASE = {
    "skin_lightness": "reset",
    "skin_roughness": "reset",
    "skin_microtexture": "reset",
    "tattoo_scale": "scenario_compile",
    "tattoo_rotation_deg": "scenario_compile",
    "tattoo_ink_opacity": "reset",
    "light_azimuth_deg": "reset",
    "light_elevation_deg": "reset",
    "light_intensity": "reset",
    "background_lightness": "reset",
    "camera_focal_scale": "reset",
    "camera_position_jitter_m": "reset",
    "camera_look_jitter_m": "reset",
    "rgb_blur_sigma_px": "post_render",
    "depth_noise_std_m": "post_render",
    "depth_dropout_fraction": "post_render",
    "occlusion_fraction": "reconfiguration",
}


def stable_seed(global_seed: int, scene_key: str, stream: str) -> int:
    """Derive an independent 64-bit stream without depending on batch order."""

    raw = hashlib.sha256(f"tatbot.inkmap-seed/1\0{global_seed}\0{scene_key}\0{stream}".encode()).digest()
    return int.from_bytes(raw[:8], "little", signed=False)


@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        if not np.isfinite([self.low, self.high]).all() or self.low > self.high:
            raise ValueError(f"invalid variation range [{self.low}, {self.high}]")
        return float(rng.uniform(self.low, self.high)) if self.low != self.high else float(self.low)


@dataclass(frozen=True)
class PerceptionVariation:
    """Explicit independent axes; zero-width ranges disable an axis exactly."""

    skin_lightness: Range = field(default_factory=lambda: Range(0.75, 1.20))
    skin_roughness: Range = field(default_factory=lambda: Range(0.55, 0.95))
    skin_microtexture: Range = field(default_factory=lambda: Range(0.00, 0.06))
    tattoo_scale: Range = field(default_factory=lambda: Range(0.90, 1.10))
    tattoo_rotation_deg: Range = field(default_factory=lambda: Range(-8.0, 8.0))
    tattoo_ink_opacity: Range = field(default_factory=lambda: Range(0.85, 1.00))
    light_azimuth_deg: Range = field(default_factory=lambda: Range(-55.0, 55.0))
    light_elevation_deg: Range = field(default_factory=lambda: Range(25.0, 70.0))
    light_intensity: Range = field(default_factory=lambda: Range(0.60, 1.10))
    background_lightness: Range = field(default_factory=lambda: Range(0.65, 1.25))
    camera_focal_scale: Range = field(default_factory=lambda: Range(0.95, 1.05))
    camera_position_jitter_m: Range = field(default_factory=lambda: Range(-0.015, 0.015))
    camera_look_jitter_m: Range = field(default_factory=lambda: Range(-0.008, 0.008))
    rgb_blur_sigma_px: Range = field(default_factory=lambda: Range(0.0, 1.2))
    depth_noise_std_m: Range = field(default_factory=lambda: Range(0.0, 0.0015))
    depth_dropout_fraction: Range = field(default_factory=lambda: Range(0.0, 0.04))
    occlusion_fraction: Range = field(default_factory=lambda: Range(0.0, 0.20))

    def sample(self, global_seed: int, scene_key: str) -> tuple[dict[str, Any], dict[str, int]]:
        values: dict[str, Any] = {}
        streams: dict[str, int] = {}
        for name in self.__dataclass_fields__:
            seed = stable_seed(global_seed, scene_key, name)
            streams[name] = seed
            values[name] = getattr(self, name).sample(np.random.default_rng(seed))
        return values, streams

    def record(self) -> dict[str, Any]:
        return {
            name: {**value, "phase": AXIS_PHASE[name]}
            for name, value in asdict(self).items()
        }


def _bucket(seed: int, group: str, namespace: str) -> int:
    digest = hashlib.sha256(f"tatbot.inkmap-split/1\0{seed}\0{namespace}\0{group}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % 10_000


def split_for_groups(
    *,
    artwork_family_sha256: str,
    identity_sha256: str,
    seed: int = 0,
    holdout_basis_points: int = 1000,
) -> str:
    """Assign before augmentation with three distinct held-out partitions.

    The artwork and identity decisions are independently hashed.  Families
    held out only by artwork enter design-held-out; identities held out only by
    identity enter identity-held-out; the intersection enters joint-held-out.
    """

    if not 1 <= holdout_basis_points <= 4_000:
        raise ValueError("holdout_basis_points must be between 1 and 4000")
    design = _bucket(seed, artwork_family_sha256, "design") < holdout_basis_points
    identity = _bucket(seed, identity_sha256, "identity") < holdout_basis_points
    if design and identity:
        return "joint-held-out"
    if design:
        return "design-held-out"
    if identity:
        return "identity-held-out"
    return "train"


def audit_split_records(records: list[dict[str, Any]]) -> list[str]:
    """Return all family/identity leakage and malformed grouping problems."""

    problems: list[str] = []
    family_splits: dict[str, set[str]] = {}
    identity_splits: dict[str, set[str]] = {}
    for index, record in enumerate(records):
        family = record.get("artwork_family_sha256")
        identity = record.get("identity_sha256")
        split = record.get("split")
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(identity, str)
            or not identity
            or not isinstance(split, str)
            or not split
        ):
            problems.append(f"record {index}: missing artwork family, identity, or split")
            continue
        family_splits.setdefault(family, set()).add(split)
        identity_splits.setdefault(identity, set()).add(split)
    for family, splits in sorted(family_splits.items()):
        # One family may be train and identity-held-out because identity is the
        # held-out axis, but it must never cross into a design-held-out class.
        design_class = {item for item in splits if item in {"design-held-out", "joint-held-out"}}
        non_design_class = splits - design_class
        if design_class and non_design_class:
            problems.append(f"artwork family {family} leaks across design holdout: {sorted(splits)}")
    for identity, splits in sorted(identity_splits.items()):
        identity_class = {item for item in splits if item in {"identity-held-out", "joint-held-out"}}
        non_identity_class = splits - identity_class
        if identity_class and non_identity_class:
            problems.append(f"identity {identity} leaks across identity holdout: {sorted(splits)}")
    return problems
