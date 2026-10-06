"""Portable episode overrides; measurements and private asset paths stay beside the external bank."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

from tatbot_travel.episode import EpisodeConfig
from tatbot_travel.postfx import CameraLook


def _overrides(config, values: dict):
    allowed = {f.name for f in fields(config)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {type(config).__name__} fields: {sorted(unknown)}")
    updates = {}
    for name, value in values.items():
        current = getattr(config, name)
        if is_dataclass(current):
            value = _overrides(current, value)
        elif isinstance(value, list):
            value = tuple(value)
        updates[name] = value
    return replace(config, **updates)


def load_profile(path: str | Path | None, seconds: float) -> EpisodeConfig:
    cfg = EpisodeConfig(duration_s=seconds)
    if path is None:
        return cfg
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    spec = json.loads(raw)
    if spec.get("version") != 1:
        raise ValueError("episode profile must have version 1")
    overrides = dict(spec["episode"])
    look = overrides.pop("camera_look", None)
    cfg = _overrides(cfg, overrides)
    world_paths = {}
    for name in ("real_assets", "room_assets", "selfview_assets"):
        value = getattr(cfg.world, name)
        if value is not None:
            target = (source.parent / value).resolve()
            if not target.is_dir():
                raise ValueError(f"missing profile {name}: {target}")
            world_paths[name] = str(target)
    if not 0 <= cfg.start_trace_prob <= 1 or not 0 <= cfg.start_approach_prob <= 1 - cfg.start_trace_prob:
        raise ValueError("profile start probabilities must sum to at most one")
    if cfg.world.phantom_length_m < 0.40:
        raise ValueError("phantom_length_m must retain the whole forearm shell (at least 0.40 m)")
    if cfg.world.phantom_hand_pose not in ("open", "closed"):
        raise ValueError("phantom_hand_pose must be 'open' or 'closed'")
    camera = CameraLook(**look) if look is not None else None
    return replace(cfg, duration_s=seconds, world=replace(cfg.world, **world_paths), camera_look=camera,
                   profile_id=hashlib.sha256(raw).hexdigest())
