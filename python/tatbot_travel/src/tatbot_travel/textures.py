"""Procedural textures: silicone skin, tables, walls, and the pen's chrome.

Everything is numpy, seeded by the episode's generator, and small enough to
rebuild per episode. The real phantom is light-yellow silicone over white
foam; its skin colour is sampled around that with a long tail toward other
practice-skin and skin tones, because the demo will not always meet the
phantom under the light it was photographed in.
"""

from __future__ import annotations

import cv2
import numpy as np

# Linear-ish sRGB anchors (0..255). The first is the practice arm as delivered.
SKIN_ANCHORS = np.array([
    (236, 214, 165),  # light-yellow silicone (the demo phantom)
    (240, 205, 180),  # pale pink practice skin
    (224, 180, 140),
    (198, 150, 110),
    (160, 112, 80),
    (110, 75, 55),
    (245, 232, 210),  # near-white silicone
    (244, 240, 232),  # the rig's practice arm, which its wrist camera sees almost white
], dtype=np.float32)
SKIN_WEIGHTS = np.array([0.3, 0.08, 0.07, 0.06, 0.04, 0.03, 0.3, 0.12])


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], scale_px: float) -> np.ndarray:
    """Zero-mean, unit-ish noise with features about ``scale_px`` wide."""
    small = (max(2, int(shape[0] / scale_px)), max(2, int(shape[1] / scale_px)))
    field = rng.standard_normal(small).astype(np.float32)
    field = cv2.resize(field, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return field / max(float(field.std()), 1e-6)


def skin_colour(rng: np.random.Generator) -> np.ndarray:
    base = SKIN_ANCHORS[rng.choice(len(SKIN_ANCHORS), p=SKIN_WEIGHTS)]
    return np.clip(base * rng.uniform(0.9, 1.08) + rng.normal(0, 6, 3), 20, 250)


def skin_texture(rng: np.random.Generator, shape=(1024, 2048), base: np.ndarray | None = None,
                 properties: dict | None = None) -> np.ndarray:
    """Silicone: a base tone, low-frequency blotches, fine speckle and faint pores."""
    base = skin_colour(rng) if base is None else base
    if properties is None:
        properties = {"mottling": rng.uniform(0.01, 0.05), "blotch_scale_px": rng.uniform(80, 300),
                      "fine_texture": rng.uniform(0.0, 0.03), "fine_scale_px": rng.uniform(6, 20),
                      "pore_density": rng.uniform(0.0, 0.004), "pore_shade": rng.uniform(0.75, 0.95)}
    tex = np.ones((*shape, 3), np.float32) * base
    tex *= 1.0 + properties["mottling"] * smooth_noise(rng, shape, properties["blotch_scale_px"])[..., None]
    tex *= 1.0 + properties["fine_texture"] * smooth_noise(rng, shape, properties["fine_scale_px"])[..., None]
    speckle = rng.random(shape) < properties["pore_density"]
    tex[speckle] *= properties["pore_shade"]
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    tex[-1] = tex[0]  # one periodic field around the cylindrical atlas seam
    return tex


def skin_chart_texture(atlas: np.ndarray, x_range: tuple[float, float], shell,
                       theta_origin: float, shape) -> np.ndarray:
    """Sample the same arm atlas in the canonical forearm ink chart."""
    from tatbot_travel.ink import texel_chart

    x, theta = texel_chart(shell, shape)
    u = (x - x_range[0]) / (x_range[1] - x_range[0])
    v = (0.5 + (theta + theta_origin) / (2 * np.pi)) % 1.0
    cols = np.broadcast_to(u * (atlas.shape[1] - 1), shape).astype(np.float32)
    rows = np.broadcast_to(((1 - v) * (atlas.shape[0] - 1))[:, None], shape).astype(np.float32)
    return cv2.remap(atlas, cols, rows, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def muted_colour(rng: np.random.Generator, lo: float = 10, hi: float = 245) -> np.ndarray:
    """A room's colour: mostly near-neutral (whites, greys, beiges), otherwise a hue at half saturation.

    Real wrist frames are far less saturated than uniformly random RGB (median HSV saturation 68 against 131
    in v4's renders) and higher in contrast (bright rooms around the dark cradle).
    """
    if rng.random() < 0.6:
        return np.clip(rng.uniform(max(lo, 110), hi) + rng.normal(0, 8, 3), 0, 255).astype(np.float32)
    colour = rng.uniform(lo, hi, 3)
    return (colour.mean() + 0.5 * (colour - colour.mean())).astype(np.float32)


def table_texture(rng: np.random.Generator, shape=(512, 512), *, kind: str | None = None,
                  colour: np.ndarray | None = None) -> np.ndarray:
    """Plain, wood-grain, cutting-mat grid, cloth weave, rubber or speckled laminate."""
    kind = kind or rng.choice(["plain", "wood", "grid", "cloth", "speckle"], p=[0.2, 0.25, 0.25, 0.15, 0.15])
    supplied_colour = colour is not None
    colour = muted_colour(rng, 20, 235) if colour is None else np.asarray(colour, dtype=np.float32)
    if kind == "wood" and not supplied_colour and rng.random() < 0.7:  # light woods, like the lab's
        colour = np.array([rng.uniform(170, 225), rng.uniform(135, 180), rng.uniform(95, 140)], np.float32)
    tex = np.ones((*shape, 3), np.float32) * colour
    if kind == "wood":
        grain = np.sin(np.linspace(0, rng.uniform(20, 80), shape[1])[None, :]
                       + 3.0 * smooth_noise(rng, shape, rng.uniform(20, 60)))
        tex *= (1.0 + 0.12 * grain)[..., None]
    elif kind == "grid":
        pitch = int(rng.integers(12, 40))
        line = np.array(colour * rng.uniform(0.4, 1.6), np.float32)
        tex[::pitch, :] = line
        tex[:, ::pitch] = line
    elif kind == "cloth":
        # Keep the weave band-limited at inspection and wrist resolutions;
        # subpixel crossed stripes produce large moire blocks when minified.
        pitch = rng.uniform(18, 32)
        warp = np.sin(np.arange(shape[1])[None, :] * 2 * np.pi / pitch)
        weft = np.sin(np.arange(shape[0])[:, None] * 2 * np.pi / pitch)
        tex *= (1.0 + 0.045 * warp + 0.045 * weft)[..., None]
    elif kind == "speckle":
        dots = rng.random(shape) < 0.04
        tex[dots] = rng.uniform(0, 255, 3)
    elif kind == "rubber":
        tex *= (1.0 + 0.10 * smooth_noise(rng, shape, 3))[..., None]
        pitch = int(rng.integers(8, 18))
        tex[::pitch, :] *= 0.75
    tex *= (1.0 + 0.02 * smooth_noise(rng, shape, 64))[..., None]
    return np.clip(tex, 0, 255).astype(np.uint8)


def paper_texture(rng: np.random.Generator, kind: str, shape=(512, 384)) -> np.ndarray:
    """Blank, ruled, graph or printed paper, with fibers and faint uneven tone."""
    white = rng.uniform(205, 250) * np.array([1.0, 1.0, rng.uniform(0.91, 1.0)])
    tex = np.ones((*shape, 3), np.float32) * white
    tex *= (1 + 0.012 * smooth_noise(rng, shape, 4))[..., None]
    image = np.clip(tex, 0, 255).astype(np.uint8)
    h, w = shape
    pitch = int(rng.integers(18, 32))
    if kind in ("lined", "grid"):
        for y in range(pitch, h, pitch):
            cv2.line(image, (0, y), (w - 1, y), (145, 175, 200), 1)
        if kind == "grid":
            for x in range(pitch, w, pitch):
                cv2.line(image, (x, 0), (x, h - 1), (145, 175, 200), 1)
        else:
            cv2.line(image, (w // 8, 0), (w // 8, h - 1), (195, 135, 135), 1)
    elif kind == "printed":
        for y in range(h // 10, 4 * h // 5, pitch):
            for x in range(w // 10, int(rng.uniform(0.45, 0.9) * w), 18):
                cv2.rectangle(image, (x, y), (x + int(rng.integers(6, 15)), y + 3), (80, 85, 90), -1)
        cv2.rectangle(image, (w // 5, 4 * h // 5), (4 * w // 5, 9 * h // 10), (160, 165, 170), 1)
    elif kind != "plain":
        raise ValueError(f"unknown paper style: {kind}")
    return image


def tag_texture(rng: np.random.Generator, cells: int = 6, cell_px: int = 32) -> np.ndarray:
    """A fiducial-like marker: random black and white cells inside a black border, on a white margin."""
    bits = rng.random((cells, cells)) < 0.5
    grid = np.zeros((cells + 2, cells + 2), bool)
    grid[1:-1, 1:-1] = bits
    tag = np.pad(grid, 1, constant_values=True)  # white margin around the black border
    tex = np.kron(tag.astype(np.float32), np.ones((cell_px, cell_px), np.float32))
    return (np.repeat(tex[..., None], 3, axis=2) * rng.uniform(200, 250)).astype(np.uint8)


def wall_texture(rng: np.random.Generator, shape=(256, 512)) -> np.ndarray:
    """A background surface: a flat colour with blotches, stripes or a few blocks."""
    tex = np.ones((*shape, 3), np.float32) * muted_colour(rng)
    tex *= (1.0 + 0.15 * smooth_noise(rng, shape, rng.uniform(20, 120)))[..., None]
    for _ in range(int(rng.integers(0, 8))):  # clutter: shelves, boxes, bottles
        h, w = rng.integers(10, shape[0] // 2), rng.integers(10, shape[1] // 3)
        y, x = rng.integers(0, shape[0] - h), rng.integers(0, shape[1] - w)
        tex[y:y + h, x:x + w] = muted_colour(rng, 0, 255)
    if rng.random() < 0.3:
        period = rng.uniform(8, 60)
        stripes = (np.sin(np.arange(shape[1]) * 2 * np.pi / period) > 0)[None, :, None]
        tex = np.where(stripes, tex * rng.uniform(0.6, 0.9), tex)
    return np.clip(tex, 0, 255).astype(np.uint8)


def chrome_texture(rng: np.random.Generator, shape=(64, 256)) -> np.ndarray:
    """Environment reflections on a polished cylinder: bands around the axis (u), constant along it."""
    u = np.linspace(0.0, 1.0, shape[1], endpoint=False)
    profile = np.full(shape[1], rng.uniform(110, 190), np.float32)  # the laser pen reads bright silver
    for _ in range(int(rng.integers(3, 9))):
        centre, width = rng.random(), rng.uniform(0.01, 0.12)
        dist = np.minimum(np.abs(u - centre), 1.0 - np.abs(u - centre))
        profile += rng.uniform(-90, 150) * np.exp(-0.5 * (dist / width) ** 2)
    tint = rng.uniform(0.85, 1.15, 3)
    tex = np.clip(profile[None, :, None] * tint[None, None, :], 5, 255)
    return np.repeat(tex, shape[0], axis=0).astype(np.uint8)
