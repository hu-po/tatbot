"""Write an 8-bit RGB image as PNG with the standard library, so rendering needs no imaging package."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path | str, rgb: np.ndarray) -> Path:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("write_png takes an RGB image")
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    data = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b""))
    path = Path(path)
    path.write_bytes(data)
    return path
