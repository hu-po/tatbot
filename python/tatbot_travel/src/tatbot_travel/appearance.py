"""Seeded skin, workspace surfaces and lighting, shared by previews and generation."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np


def _range(name: str, values, *, lower: float | None = None, upper: float | None = None):
    value = np.asarray(values, dtype=float)
    if (value.shape != (2,) or not np.isfinite(value).all() or value[0] > value[1]
            or (lower is not None and value[0] < lower) or (upper is not None and value[1] > upper)):
        raise ValueError(f"invalid appearance range: {name}")


@dataclass(frozen=True)
class SurfaceConfig:
    styles: dict[str, float] = field(default_factory=lambda: {"plain": 0.65, "cloth": 0.25, "speckle": 0.10})
    saturation: tuple[float, float] = (0.0, 0.5)
    value: tuple[float, float] = (0.15, 0.90)
    neutral_prob: float = 0.35
    specular: tuple[float, float] = (0.0, 0.15)
    shininess: tuple[float, float] = (0.05, 0.35)
    randomize_captured_floor: bool = False
    captured_floor_tolerance_m: float = 0.02

    def __post_init__(self):
        weights = np.asarray(list(self.styles.values()), dtype=float)
        if (not self.styles or set(self.styles) - {"plain", "cloth", "speckle", "wood", "grid", "rubber"}
                or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError("surface styles need known names and non-negative weights with a positive sum")
        for name in ("saturation", "value", "specular", "shininess"):
            _range(name, getattr(self, name), lower=0, upper=1)
        if not 0 <= self.neutral_prob <= 1 or not 0 < self.captured_floor_tolerance_m <= 0.05:
            raise ValueError("invalid surface probability or captured floor tolerance")


@dataclass(frozen=True)
class LightingConfig:
    energy: tuple[float, float] = (0.5, 1.8)
    warmth: tuple[float, float] = (-0.25, 0.25)
    x_m: tuple[float, float] = (-1.5, 1.8)
    y_m: tuple[float, float] = (-2.0, 2.0)
    height_m: tuple[float, float] = (0.65, 2.8)
    headlight_diffuse: tuple[float, float] = (0.04, 0.25)
    headlight_ambient: tuple[float, float] = (0.12, 0.35)
    shadow_prob: float = 0.8
    fill_shadow_prob: float = 0.25
    spot_prob: float = 0.4

    def __post_init__(self):
        for name in ("x_m", "y_m", "warmth"):
            _range(name, getattr(self, name))
        for name in ("energy", "height_m", "headlight_diffuse", "headlight_ambient"):
            _range(name, getattr(self, name), lower=0)
        _range("warmth", self.warmth, lower=-0.5, upper=0.5)
        if any(not 0 <= getattr(self, name) <= 1 for name in ("shadow_prob", "fill_shadow_prob", "spot_prob")):
            raise ValueError("lighting probabilities must lie in [0, 1]")


@dataclass(frozen=True)
class ScalarVariation:
    center: float
    std: float
    bounds: tuple[float, float]

    def __post_init__(self):
        _range("skin property", self.bounds)
        if (not np.isfinite([self.center, self.std]).all() or self.std < 0
                or not self.bounds[0] <= self.center <= self.bounds[1]):
            raise ValueError("skin property needs a finite center, non-negative spread and bounds containing the center")

    def sample(self, rng: np.random.Generator, randomize: bool) -> float:
        return float(np.clip(rng.normal(self.center, self.std), *self.bounds)) if randomize else self.center


@dataclass(frozen=True)
class SkinConfig:
    center_rgb: tuple[float, float, float] = (236, 225, 198)
    randomize: bool = True
    tail_prob: float = 0.20
    tail_gain: tuple[float, float] = (0.40, 0.85)
    tone: ScalarVariation = field(default_factory=lambda: ScalarVariation(1.0, 0.10, (0.65, 1.10)))
    warmth: ScalarVariation = field(default_factory=lambda: ScalarVariation(0, 9, (-25, 25)))
    redness: ScalarVariation = field(default_factory=lambda: ScalarVariation(0, 6, (-20, 20)))
    specular: ScalarVariation = field(default_factory=lambda: ScalarVariation(0.30, 0.16, (0.02, 0.75)))
    shininess: ScalarVariation = field(default_factory=lambda: ScalarVariation(0.40, 0.20, (0.04, 0.85)))
    mottling: ScalarVariation = field(default_factory=lambda: ScalarVariation(0.03, 0.03, (0, 0.12)))
    fine_texture: ScalarVariation = field(default_factory=lambda: ScalarVariation(0.015, 0.014, (0, 0.06)))
    pore_density: ScalarVariation = field(default_factory=lambda: ScalarVariation(0.002, 0.003, (0, 0.014)))
    blotch_scale_px: tuple[float, float] = (60, 300)
    fine_scale_px: tuple[float, float] = (4, 24)
    pore_shade: tuple[float, float] = (0.70, 1.0)

    def __post_init__(self):
        rgb = np.asarray(self.center_rgb, dtype=float)
        if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any((rgb < 25) | (rgb > 250)):
            raise ValueError("skin center_rgb must have three channels in [25, 250]")
        if not 0 <= self.tail_prob <= 1:
            raise ValueError("skin tail probability must lie in [0, 1]")
        for name in ("tail_gain", "blotch_scale_px", "fine_scale_px"):
            _range(name, getattr(self, name), lower=0.001)
        _range("tone", self.tone.bounds, lower=0.001)
        _range("pore_shade", self.pore_shade, lower=0, upper=1)
        for name in ("specular", "shininess", "mottling", "fine_texture", "pore_density"):
            _range(name, getattr(self, name).bounds, lower=0, upper=1)


def sample_skin(rng: np.random.Generator, cfg: SkinConfig, center_rgb=None) -> dict:
    center = np.asarray(cfg.center_rgb if center_rgb is None else center_rgb, dtype=float)
    tail = cfg.randomize and rng.random() < cfg.tail_prob
    gain = float(rng.uniform(*cfg.tail_gain)) if tail else cfg.tone.sample(rng, cfg.randomize)
    warmth, redness = cfg.warmth.sample(rng, cfg.randomize), cfg.redness.sample(rng, cfg.randomize)
    rgb = np.clip(center * gain + warmth * np.array([0.5, 0, -1])
                  + redness * np.array([0.9, -0.25, -0.4]), 25, 250)
    look = {"center_rgb": center.tolist(), "rgb": rgb.tolist(), "tone_gain": gain,
            "warmth": warmth, "redness": redness,
            "branch": "reference" if not cfg.randomize else ("tail" if tail else "near")}
    for name in ("specular", "shininess", "mottling", "fine_texture", "pore_density"):
        look[name] = getattr(cfg, name).sample(rng, cfg.randomize)
    for name in ("blotch_scale_px", "fine_scale_px", "pore_shade"):
        bounds = getattr(cfg, name)
        look[name] = float(rng.uniform(*bounds)) if cfg.randomize else float(np.mean(bounds))
    return look


def sample_surface(rng: np.random.Generator, cfg: SurfaceConfig) -> dict:
    styles = list(cfg.styles)
    weights = np.asarray(list(cfg.styles.values()))
    style = str(rng.choice(styles, p=weights / weights.sum()))
    saturation = 0.0 if rng.random() < cfg.neutral_prob else float(rng.uniform(*cfg.saturation))
    rgb = colorsys.hsv_to_rgb(float(rng.random()), saturation, float(rng.uniform(*cfg.value)))
    return {"style": style, "rgb": list(rgb), "specular": float(rng.uniform(*cfg.specular)),
            "shininess": float(rng.uniform(*cfg.shininess))}


def sample_lighting(rng: np.random.Generator, cfg: LightingConfig, max_lights: int) -> dict:
    if max_lights < 1:
        raise ValueError("max_lights must be positive")
    count = int(rng.integers(1, max_lights + 1))
    energy, warmth = float(rng.uniform(*cfg.energy)), float(rng.uniform(*cfg.warmth))
    lights = []
    for index in range(count):
        pos = np.array([rng.uniform(*cfg.x_m), rng.uniform(*cfg.y_m), rng.uniform(*cfg.height_m)])
        target = np.array([rng.uniform(0.1, 0.5), rng.uniform(-0.3, 0.3), 0.0])
        direction = (target - pos) / max(np.linalg.norm(target - pos), 1e-9)
        tint = np.clip(warmth + rng.normal(0, 0.04), -0.5, 0.5)
        colour = energy / count * np.array([1 + tint, 1.0, 1 - tint]) * rng.uniform(0.6, 1.4)
        spot = bool(rng.random() < cfg.spot_prob)
        lights.append({"position": pos.tolist(), "direction": direction.tolist(),
                       "diffuse": np.clip(colour, 0, 1).tolist(),
                       "shadow": bool(rng.random() < (cfg.shadow_prob if index == 0 else cfg.fill_shadow_prob)),
                       "cutoff": float(rng.uniform(25, 70)) if spot else 45.0,
                       "exponent": float(rng.uniform(1, 12)) if spot else 0.0})
    return {"energy": energy, "warmth": warmth, "lights": lights,
            "headlight": [float(rng.uniform(*cfg.headlight_diffuse)), float(rng.uniform(*cfg.headlight_ambient))]}
