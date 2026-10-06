"""Python client for visiond's bounded synchronized-frame Unix transport.

The Rust producer owns the wire contract in ``rust/visiond/src/transport.rs``.
Keep the small decoder here so every Python vision consumer gets identical
validation and latest-frame-wins behavior without copying socket code.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
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


def decode_video(data: bytes, descriptor: dict, *, preserve_luma: bool = False) -> np.ndarray:
    width, height = int(descriptor["width"]), int(descriptor["height"])
    pixel_format = descriptor["format"].lower()
    if width <= 0 or height <= 0:
        raise ValueError("video descriptor has empty geometry")
    if pixel_format == "jpeg":
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (height, width):
            raise ValueError("invalid JPEG payload geometry")
        return frame
    if pixel_format == "yuyv":
        if width % 2 or len(data) != width * height * 2:
            raise ValueError("invalid YUYV payload geometry")
        packed = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 2)
        return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_YUY2)
    if pixel_format == "y8":
        if len(data) != width * height:
            raise ValueError("invalid Y8 payload length")
        gray = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return gray if preserve_luma else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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
    never by an assumed constant (the retired scripts/depth_probe.py learned
    that the expensive way on 2026-08-22; scripts/il_patch_lerobot.py patch 6).
    """
    width, height = int(descriptor["width"]), int(descriptor["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"depth descriptor has an empty geometry {width}x{height}")
    expected = width * height * 2
    if len(data) != expected:
        raise ValueError(f"depth payload is {len(data)} bytes, expected {expected}")
    return np.frombuffer(data, dtype="<u2").reshape(height, width)


class UnixWireReader:
    def __init__(self, path: Path, connect_timeout_s: float = 10.0, *, stop_event: threading.Event | None = None):
        deadline = time.monotonic() + connect_timeout_s
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    self.socket.connect(str(path))
                    return
                except (FileNotFoundError, ConnectionRefusedError) as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"visiond socket did not appear at {path}") from error
                    if stop_event is None:
                        time.sleep(0.1)
                    else:
                        stop_event.wait(0.1)
            raise RuntimeError("visiond subscription cancelled during connection")
        except BaseException:
            self.close()
            raise

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

    def receive_raw(self) -> tuple[dict, list[bytes]]:
        """One set's validated header and its payload bytes, nothing decoded.

        The owner writes each set to every client with a 20 ms timeout and drops
        a client that stalls it; a luma set from five PoE cameras is ~25 MB, so a
        reader that converts each frame between payload reads stalls that long
        whenever OpenCV is cold or the node is loaded (measured 5 of 6 readers
        dropped under a running tracker). Nothing here touches a pixel.
        """
        header_length = struct.unpack(">I", self._read_exact(4))[0]
        if not 0 < header_length <= MAX_HEADER_BYTES:
            raise ValueError(f"invalid wire header length {header_length}")
        header = json.loads(self._read_exact(header_length))
        if header.get("magic") != WIRE_MAGIC or header.get("version") != WIRE_VERSION:
            raise ValueError("unsupported visiond frame-set wire format")
        payloads = []
        for wire_frame in header["frames"]:
            variant, descriptor = payload_descriptor(wire_frame["payload"])
            if variant not in ("video", "depth"):
                raise ValueError(f"vision consumer needs decoded video or depth, got {variant}")
            count = int(descriptor["bytes"])
            if not 0 <= count <= MAX_PAYLOAD_BYTES:
                raise ValueError(f"invalid payload length {count}")
            payloads.append(self._read_exact(count))
        return header, payloads

    def receive(self) -> dict:
        """The next decoded set; the whole set is drained before decoding starts."""
        return decode_frame_set(*self.receive_raw())


def decode_frame_set(header: dict, payloads: list[bytes], *, retain_payloads: bool = False) -> dict:
    """Decode one raw set (from ``receive_raw``) into images and depth arrays."""
    frames = {}
    for wire_frame, payload in zip(header["frames"], payloads, strict=True):
        variant, descriptor = payload_descriptor(wire_frame["payload"])
        metadata = wire_frame["metadata"]
        # One sensor may deliver both a colour and a depth frame in a set
        # (the D405 pair); keep both under the sensor rather than letting
        # the second overwrite the first.
        entry = frames.setdefault(metadata["sensor_name"], {"metadata": metadata})
        if retain_payloads:
            profile = metadata["profile"]
            payload_format = "z16" if variant == "depth" else descriptor["format"].lower()
            if (any(profile[key] != descriptor[key] for key in ("width", "height"))
                    or profile["format"].lower() != payload_format):
                raise ValueError("original wire payload descriptor differs from its metadata profile")
            entry["raw_payload"] = payload
        if variant == "video":
            entry["image"] = decode_video(payload, descriptor)
            entry["image_metadata"] = metadata
        elif variant == "depth":
            entry["depth"] = decode_depth(payload, descriptor)
            entry["depth_metadata"] = metadata
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


def latest_socket_sets(path: Path, connect_timeout_s: float = 10.0, duration_s: float | None = None, *,
                       stop_event: threading.Event | None = None, retain_payloads: bool = False):
    """Drain continuously while yielding only the newest complete frame set.

    The socket thread only reads bytes; each yielded set is decoded here, on
    the consumer's thread, so a slow or cold decode can never stall the
    owner's writes past its client timeout. Sets the consumer never takes
    are dropped undecoded.
    """
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("duration_s must be finite and positive")
    deadline = None if duration_s is None else time.monotonic() + duration_s
    latest: queue.Queue = queue.Queue(maxsize=1)
    stopped = stop_event if stop_event is not None else threading.Event()
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
            reader = UnixWireReader(path, connect_timeout_s=min(connect_timeout_s, duration_s) if duration_s is not None else connect_timeout_s,
                                    stop_event=stopped)
            active_reader.append(reader)
            while not stopped.is_set():
                deliver(reader.receive_raw())
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
        while not stopped.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            try:
                item = latest.get(timeout=min(0.1, remaining) if remaining is not None else 0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield decode_frame_set(*item, retain_payloads=retain_payloads)
    finally:
        stopped.set()
        for reader in active_reader:
            with contextlib.suppress(OSError):
                reader.socket.shutdown(socket.SHUT_RDWR)
                reader.close()
        worker.join(timeout=1.0)


def read_evidence_frame(camera_dir: Path, entry: dict, *, preserve_luma: bool = False) -> dict:
    path = (camera_dir / entry["payload_file"]).resolve()
    if not path.is_relative_to(camera_dir.resolve()):
        raise ValueError("evidence payload escapes recording directory")
    payload = path.read_bytes()
    if len(payload) != int(entry["payload_bytes"]):
        raise ValueError(f"payload length mismatch for {path}")
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"payload checksum mismatch for {path}")
    metadata = entry["metadata"]
    profile = metadata["profile"]
    descriptor = {
        "format": profile["format"],
        "width": profile["width"],
        "height": profile["height"],
    }
    if profile["format"] == "z16":
        return {"metadata": metadata, "depth": decode_depth(payload, descriptor)}
    return {"metadata": metadata, "image": decode_video(payload, descriptor, preserve_luma=preserve_luma)}
