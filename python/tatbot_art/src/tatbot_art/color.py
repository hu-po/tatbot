"""Colour spaces and differences: sRGB, linear light, OKLab, CIELAB (D65) and CIEDE2000.

Arrays are float, channels last, sRGB in 0..1. Every function broadcasts over leading axes.
"""
from __future__ import annotations

import numpy as np

_RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
# Ottosson 2020, linear sRGB -> LMS -> OKLab.
_RGB_TO_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                        [0.2119034982, 0.6806995451, 0.1073969566],
                        [0.0883024619, 0.2817188376, 0.6299787005]])
_LMS_TO_OKLAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                          [1.9779984951, -2.4285922050, 0.4505937099],
                          [0.0259040371, 0.7827717662, -0.8086757660]])


def from_8bit(rgb) -> np.ndarray:
    return np.asarray(rgb, float) / 255.0


def to_8bit(rgb) -> np.ndarray:
    return np.clip(np.rint(np.asarray(rgb, float) * 255.0), 0, 255).astype(np.uint8)


def srgb_to_linear(c) -> np.ndarray:
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c) -> np.ndarray:
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def oklab(srgb) -> np.ndarray:
    lms = srgb_to_linear(srgb) @ _RGB_TO_LMS.T
    return np.cbrt(lms) @ _LMS_TO_OKLAB.T


def oklab_to_srgb(lab) -> np.ndarray:
    lms = (np.asarray(lab, float) @ np.linalg.inv(_LMS_TO_OKLAB).T) ** 3
    return linear_to_srgb(lms @ np.linalg.inv(_RGB_TO_LMS).T)


def oklch(srgb) -> np.ndarray:
    """Lightness, chroma and hue in degrees: the space the colour skill's harmonies are stated in."""
    lab = oklab(srgb)
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    return np.stack([lab[..., 0], np.hypot(lab[..., 1], lab[..., 2]), hue], axis=-1)


def cielab(srgb) -> np.ndarray:
    xyz = (srgb_to_linear(srgb) @ _RGB_TO_XYZ.T) / _WHITE_D65
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def delta_e00(lab1, lab2) -> np.ndarray:
    """CIEDE2000 between CIELAB colours (Sharma, Wu and Dalal 2005), kL = kC = kH = 1."""
    lab1, lab2 = np.asarray(lab1, float), np.asarray(lab2, float)
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dlp, dcp = l2 - l1, c2p - c1p
    zero = (c1p * c2p) == 0
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(zero, 0.0, dh)
    dhp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    lbp, cbp = (l1 + l2) / 2, (c1p + c2p) / 2
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2))
    hbp = np.where(zero, hsum, hbp)
    t = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    d_theta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    rc = 2 * np.sqrt(cbp ** 7 / (cbp ** 7 + 25.0 ** 7))
    sl = 1 + 0.015 * (lbp - 50) ** 2 / np.sqrt(20 + (lbp - 50) ** 2)
    sc, sh = 1 + 0.045 * cbp, 1 + 0.015 * cbp * t
    rt = -np.sin(np.radians(2 * d_theta)) * rc
    return np.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp / sh) ** 2 + rt * (dcp / sc) * (dhp / sh))
