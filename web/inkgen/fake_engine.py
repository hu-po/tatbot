"""A stand-in engine: the same interface as `engine.Engine`, no weights.

For the adapter tests and for driving the page or a browser suite without a
GPU (`INKGEN_FAKE_ENGINE=1 python app.py`). It answers every identity
question the real engine does — the mismatch refusal included — and draws a
tiny deterministic PNG whose pixels follow the seed, so two requests with
different seeds give different bytes and one request repeated gives the same.

Stdlib only: the PNG is written by hand, so this file needs neither Pillow nor
torch. It is never the producer of anything a project keeps.
"""
from __future__ import annotations

import struct
import threading
import time
import zlib
from collections.abc import Callable

from contracts import PRODUCER, PRODUCER_VERSION, GenerationRequest, GenerationSettings, RequestError
from engine import check_request

__all__ = ["FakeEngine", "FakeImage", "png_bytes"]

FAKE_DEVICE = "fake"


def png_bytes(width: int, height: int, ink: Callable[[int, int], bool]) -> bytes:
    """A grayscale PNG with `ink(x, y)` black and everything else white."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: none
        raw.extend(0 if ink(x, y) else 255 for x in range(width))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))


class FakeImage:
    """Just enough of a PIL image for the serving adapters: `.save(buffer, format="PNG")`."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def save(self, buffer, format: str = "PNG") -> None:  # noqa: A002 - PIL's spelling
        if format.upper() != "PNG":
            raise ValueError("the fake image is a PNG")
        buffer.write(self.data)


class FakeEngine:
    """`Engine` without the model. `render_hook` runs inside each render, for tests that need to hold one open."""

    def __init__(self, settings: GenerationSettings | None = None, *, device: str = FAKE_DEVICE,
                 delay_s: float = 0.0, render_hook: Callable[[GenerationRequest], None] | None = None) -> None:
        self.settings = settings or GenerationSettings()
        self.device = device
        self.delay_s = float(delay_s)
        self.render_hook = render_hook
        self.calls: list[GenerationRequest] = []
        self._loaded = False
        self._lock = threading.Lock()
        self.concurrent = 0
        self.max_concurrent = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    def identity(self) -> dict:
        return {"model": self.settings.model, "model_revision": self.settings.model_revision,
                "settings": self.settings.as_json(), "device": self.device, "loaded": self.loaded,
                "producer": PRODUCER, "producer_version": PRODUCER_VERSION}

    def pin(self, *, required: bool = False) -> str | None:
        return self.settings.model_revision

    def load(self):
        self._loaded = True
        return self

    def close(self) -> None:
        self._loaded = False

    def check_request(self, request: GenerationRequest) -> None:
        check_request(self.settings, request)

    def render(self, request: GenerationRequest) -> FakeImage:
        if not isinstance(request, GenerationRequest):
            raise RequestError("render takes a normalized GenerationRequest")
        self.check_request(request)
        self.load()
        with self._lock:
            self.calls.append(request)
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.render_hook is not None:
                self.render_hook(request)
            if self.delay_s:
                time.sleep(self.delay_s)
            size = 64
            seed = int(request.seed)
            cx, cy = 8 + seed % 40, 8 + (seed // 40) % 40
            return FakeImage(png_bytes(size, size, lambda x, y: abs(x - cx) <= 6 and abs(y - cy) <= 6))
        finally:
            with self._lock:
                self.concurrent -= 1
