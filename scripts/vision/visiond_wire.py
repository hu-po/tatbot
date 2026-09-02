"""Python client for visiond's bounded synchronized-frame Unix transport.

The Rust producer owns the wire contract in ``rust/visiond/src/transport.rs``.
Keep the small decoder here so every Python vision consumer gets identical
validation and latest-frame-wins behavior without copying socket code.
"""

from __future__ import annotations

import contextlib
import json
import queue
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

WIRE_MAGIC = "tatbot-vision-frame-set"
WIRE_VERSION = 1
MAX_HEADER_BYTES = 4 * 1024 * 1024
MAX_PAYLOAD_BYTES = 128 * 1024 * 1024


def payload_descriptor(payload: dict) -> tuple[str, dict]:
    if len(payload) != 1:
        raise ValueError("wire payload descriptor must have one variant")
    variant, fields = next(iter(payload.items()))
    return variant.lower(), fields


def decode_video(data: bytes, descriptor: dict) -> np.ndarray:
    width, height = int(descriptor["width"]), int(descriptor["height"])
    pixel_format = descriptor["format"].lower()
    if pixel_format not in ("bgr8", "rgb8"):
        raise ValueError(f"vision consumer needs decoded BGR/RGB pixels, got {pixel_format}")
    expected = width * height * 3
    if len(data) != expected:
        raise ValueError(f"video payload is {len(data)} bytes, expected {expected}")
    frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if pixel_format == "rgb8" else frame


def decode_depth(data: bytes, descriptor: dict) -> np.ndarray:
    """Z16 depth as a (height, width) uint16 array in the sensor's raw units.

    The D405 reports 0.1 mm per unit (`depth_units_m` in the frame metadata
    attributes); most other D4xx report 1 mm. Readers scale by that attribute,
    never by an assumed constant (scripts/depth_probe.py learned that the
    expensive way).
    """
    width, height = int(descriptor["width"]), int(descriptor["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"depth descriptor has an empty geometry {width}x{height}")
    expected = width * height * 2
    if len(data) != expected:
        raise ValueError(f"depth payload is {len(data)} bytes, expected {expected}")
    return np.frombuffer(data, dtype="<u2").reshape(height, width)


class UnixWireReader:
    def __init__(self, path: Path, connect_timeout_s: float = 10.0):
        deadline = time.monotonic() + connect_timeout_s
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        while True:
            try:
                self.socket.connect(str(path))
                break
            except (FileNotFoundError, ConnectionRefusedError) as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"visiond socket did not appear at {path}") from error
                time.sleep(0.1)

    def close(self) -> None:
        self.socket.close()

    def _read_exact(self, count: int) -> bytes:
        chunks = []
        remaining = count
        while remaining:
            chunk = self.socket.recv(remaining)
            if not chunk:
                raise EOFError("visiond socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def receive(self) -> dict:
        header_length = struct.unpack(">I", self._read_exact(4))[0]
        if not 0 < header_length <= MAX_HEADER_BYTES:
            raise ValueError(f"invalid wire header length {header_length}")
        header = json.loads(self._read_exact(header_length))
        if header.get("magic") != WIRE_MAGIC or header.get("version") != WIRE_VERSION:
            raise ValueError("unsupported visiond frame-set wire format")
        frames = {}
        for wire_frame in header["frames"]:
            variant, descriptor = payload_descriptor(wire_frame["payload"])
            count = int(descriptor["bytes"])
            if not 0 <= count <= MAX_PAYLOAD_BYTES:
                raise ValueError(f"invalid payload length {count}")
            payload = self._read_exact(count)
            metadata = wire_frame["metadata"]
            # One sensor may deliver both a colour and a depth frame in a set
            # (the D405 pair); keep both under the sensor rather than letting
            # the second overwrite the first.
            entry = frames.setdefault(metadata["sensor_name"], {"metadata": metadata})
            if variant == "video":
                entry["image"] = decode_video(payload, descriptor)
            elif variant == "depth":
                entry["depth"] = decode_depth(payload, descriptor)
                units = (metadata.get("attributes") or {}).get("depth_units_m")
                entry["depth_units_m"] = float(units) if units is not None else None
            else:
                raise ValueError(f"vision consumer needs decoded video or depth, got {variant}")
        return {
            "sequence": int(header["sequence"]),
            "timestamp_basis": header.get("timestamp_basis", "unknown"),
            "timestamp_ns": int(header["timestamp_ns"]),
            "maximum_skew_ns": int(header["maximum_skew_ns"]),
            "frames": frames,
        }


def latest_socket_sets(path: Path, connect_timeout_s: float = 10.0):
    """Drain continuously while yielding only the newest complete frame set."""
    latest: queue.Queue = queue.Queue(maxsize=1)
    stopped = threading.Event()
    active_reader: list[UnixWireReader] = []

    def deliver(item) -> None:
        try:
            latest.put_nowait(item)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                latest.get_nowait()
            latest.put_nowait(item)

    def receive_loop() -> None:
        reader = None
        try:
            reader = UnixWireReader(path, connect_timeout_s=connect_timeout_s)
            active_reader.append(reader)
            while not stopped.is_set():
                deliver(reader.receive())
        except EOFError:
            if not stopped.is_set():
                deliver(None)
        except Exception as error:  # delivered to the processing thread
            if not stopped.is_set():
                deliver(error)
        finally:
            if reader is not None:
                reader.close()

    worker = threading.Thread(target=receive_loop, name="visiond-socket-reader", daemon=True)
    worker.start()
    try:
        while True:
            item = latest.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stopped.set()
        for reader in active_reader:
            with contextlib.suppress(OSError):
                reader.socket.shutdown(socket.SHUT_RDWR)
                reader.close()
        worker.join(timeout=1.0)
