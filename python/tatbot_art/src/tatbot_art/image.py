"""Source images: load, whiten the paper, crop to the artwork and fit it to the drawn size."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import median_filter, minimum_filter

from .color import cielab


def on_paper(im: Image.Image) -> Image.Image:
    """The picture as a person sees it: turned as its EXIF says, transparency laid over white paper. Inkmap reads an
    image the same way (web/inkmap/src/core/art-image.ts), so the art it shows is the art separated here."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return im.convert("RGB")


def load(path: Path | str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(on_paper(im), float) / 255.0


def flatten_paper(rgb: np.ndarray, lightness: float = 85.0, chroma: float = 12.0) -> np.ndarray:
    """Light, nearly neutral pixels become white: a generator's off-white paper, shadow-free border or JPEG haze."""
    lab = cielab(rgb)
    paper = (lab[..., 0] > lightness) & (np.hypot(lab[..., 1], lab[..., 2]) < chroma)
    out = rgb.copy()
    out[paper] = 1.0
    return out


def art_box(rgb: np.ndarray, margin: float = 0.03) -> tuple[int, int, int, int]:
    """The art's rectangle, (x, y, width, height) in pixels from the top left: the ink's bounding box (any
    channel under 0.98, after `flatten_paper`), padded by `margin` of the longer side. Inkmap's `art_px` is this
    rectangle (web/inkmap/src/core/art.ts), so a width chosen there is the width drawn here."""
    ink = (rgb < 0.98).any(axis=-1)
    if not ink.any():
        raise ValueError("the image has no artwork: every pixel is paper")
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    pad = int(round(margin * max(rgb.shape[:2])))
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, rgb.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, rgb.shape[1])
    return int(c0), int(r0), int(c1 - c0), int(r1 - r0)


def crop_to_art(rgb: np.ndarray, margin: float = 0.03) -> np.ndarray:
    x, y, w, h = art_box(rgb, margin)
    return rgb[y:y + h, x:x + w]


def fit(rgb: np.ndarray, width_mm: float, px_per_mm: float, keep_lines: bool = False) -> np.ndarray:
    """Resample to the drawn width at `px_per_mm`, keeping the aspect, then a 3 px median against JPEG noise.

    `keep_lines` first spreads dark pixels over one output pixel's footprint, so line art thinner than an
    output pixel survives the resample as a line instead of averaging into near-white.
    """
    width_px = max(8, int(round(width_mm * px_per_mm)))
    factor = rgb.shape[1] / width_px
    if keep_lines and factor > 1.5:
        rgb = minimum_filter(rgb, size=(int(np.ceil(factor)), int(np.ceil(factor)), 1))
    height_px = max(8, int(round(rgb.shape[0] * width_px / rgb.shape[1])))
    im = Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8))
    out = np.asarray(im.resize((width_px, height_px), Image.Resampling.LANCZOS), float) / 255.0
    return out if keep_lines else median_filter(out, size=(3, 3, 1))


def prepare(path: Path | str, width_mm: float, px_per_mm: float, keep_lines: bool = False) -> np.ndarray:
    return fit(crop_to_art(flatten_paper(load(path))), width_mm, px_per_mm, keep_lines)


def fit_width(path: Path | str, room_mm: tuple[float, float], px_per_mm: float) -> float:
    """The widest drawn width, in whole output pixels, at which `prepare`'s canvas stays inside `room_mm`
    (width, height): the art cropped as `prepare` crops it, so its aspect is the drawn one."""
    _, _, w, h = art_box(flatten_paper(load(path)))
    width_px = int(np.floor(min(room_mm[0], room_mm[1] * w / h) * px_per_mm + 1e-9))
    while width_px > 8 and round(h * width_px / w) > room_mm[1] * px_per_mm + 1e-9:
        width_px -= 1   # the height rounds as `fit` rounds it
    if width_px < 8 or round(h * width_px / w) > room_mm[1] * px_per_mm + 1e-9:
        raise ValueError(f"{Path(path).name}: the page leaves {room_mm[0]:g} x {room_mm[1]:g} mm, too little to draw it")
    return width_px / px_per_mm
