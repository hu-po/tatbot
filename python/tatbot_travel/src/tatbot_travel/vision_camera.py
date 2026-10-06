"""Latest aligned wrist RGB-D from visiond; the capture owner keeps the USB device."""

from __future__ import annotations

import copy
import json
import multiprocessing
import threading
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tomllib

from tatbot_travel import assets
from tatbot_travel.camera import Intrinsics
from tatbot_travel.shadow import SOCKET, visiond_wire


def wrist_config() -> dict:
    table = tomllib.loads((assets.repo_root() / "rust/visiond/config/vision.toml").read_text())
    cameras = [c for c in table.get("cameras", {}).get("realsense", []) if c.get("arm") == "left"]
    if len(cameras) != 1:
        raise RuntimeError("expected exactly one configured left wrist RealSense")
    return cameras[0]


def active_intrinsics(metadata: dict) -> Intrinsics:
    values = json.loads(metadata["attributes"]["intrinsics"])
    models = {"BrownConradyInverse": "inverse_brown_conrady", "InverseBrownConrady": "inverse_brown_conrady",
              "BrownConrady": "brown_conrady", "None": "brown_conrady"}
    model = models.get(values["distortion_model"])
    coefficients = tuple(values["distortion_coefficients"])
    numeric = [values[name] for name in ("fx", "fy", "ppx", "ppy")] + list(coefficients)
    if (values.get("schema") != "tatbot.camera-intrinsics/1" or model is None or len(coefficients) != 5
            or not np.isfinite(numeric).all() or min(values["fx"], values["fy"]) <= 0
            or (values["distortion_model"] == "None" and any(coefficients))):
        raise ValueError("unsupported or malformed active wrist intrinsics")
    return Intrinsics(int(values["width"]), int(values["height"]), values["fx"], values["fy"],
                      values["ppx"], values["ppy"], coefficients, model)


def decode_wrist(frame_set: dict, config: dict, *, unix_ns: int, monotonic_s: float) -> tuple[tuple, dict]:
    """Validate the configured pair, and preserve its capture age in the local monotonic clock."""
    color_name, depth_name = config["name"] + "_color", config["name"] + "_depth"
    color, depth = frame_set["frames"][color_name], frame_set["frames"][depth_name]
    cm, dm = color["image_metadata"], depth["depth_metadata"]
    intrinsics = active_intrinsics(cm)
    if active_intrinsics(dm) != intrinsics or dm["attributes"].get("aligned_to") != color_name:
        raise ValueError("wrist depth is not aligned to the active color calibration")
    identity = [(m["attributes"].get("device_serial"), m["attributes"].get("physical_arm"),
                 m["attributes"].get("capture_owner_role"), m["attributes"].get("capture_epoch")) for m in (cm, dm)]
    expected = (str(config["serial"]), "left", config["owner_role"])
    if identity[0] != identity[1] or identity[0][:3] != expected or not identity[0][3]:
        raise ValueError("wrist RGB-D identity/owner/epoch differs from the configured left camera")
    image, raw_depth = color["image"], depth["depth"]
    shape = (intrinsics.height, intrinsics.width)
    if (shape != (480, 640) or image.shape != (*shape, 3) or image.dtype != np.uint8
            or raw_depth.shape != shape or raw_depth.dtype != np.uint16):
        raise ValueError("wrist RGB-D must be the configured 640x480 color/aligned Z16 stream")
    units = depth["depth_units_m"]
    if units is None or not np.isfinite(units) or not 0 < units <= 0.01:
        raise ValueError("wrist depth units are missing or invalid")
    stamps = [m["timestamps"]["normalized_unix_ns"] for m in (cm, dm)]
    if (any(type(stamp) is not int or stamp <= 0 or stamp > unix_ns for stamp in stamps)
            or abs(stamps[0] - stamps[1]) > 40_000_000 or cm["sequence"] != dm["sequence"]):
        raise ValueError("wrist RGB-D capture timestamps are invalid or unsynchronized")
    # Use the older exposure: neither network delivery nor decode makes an old capture fresh.
    captured = monotonic_s - (unix_ns - min(stamps)) / 1e9
    rgb = np.ascontiguousarray(image[..., ::-1])  # the shared wire decoder returns BGR
    metres = raw_depth.astype(np.float32) * units
    rgb.flags.writeable = metres.flags.writeable = False
    source = {"camera": color_name, "depth": depth_name, "device_serial": identity[0][0],
              "owner_role": identity[0][2], "capture_epoch": identity[0][3], "depth_units_m": units,
              "intrinsics": asdict(intrinsics), "timestamp_basis": "normalized_unix_ns",
              "source_sequence": cm["sequence"], "capture_unix_ns": min(stamps),
              "color_metadata": copy.deepcopy(cm), "depth_metadata": copy.deepcopy(dm)}
    source.update(_original_payloads(color, depth))
    return (rgb, metres, captured), source


def _original_payloads(color, depth):
    if "raw_payload" not in color and "raw_payload" not in depth:
        return {}
    if any(not isinstance(entry.get("raw_payload"), bytes) for entry in (color, depth)):
        raise ValueError("original wrist capture requires both owner payloads")
    return {"_original_payloads": {"color": color["raw_payload"], "depth": depth["raw_payload"]}}


class _CaptureSlot:
    """One atomic latest RGB-D pair; no image pickling or queued old frames."""

    RGB_BYTES = 480 * 640 * 3
    DEPTH_BYTES = 480 * 640 * 4
    # Both original metadata records fit the shared wire's bounded header.
    SOURCE_BYTES = 8 * 1024 * 1024

    def __init__(self, context, *, retain_original=False):
        self.bytes = context.RawArray("B", self.RGB_BYTES + self.DEPTH_BYTES + self.SOURCE_BYTES)
        self.retain_original = retain_original
        self.original = context.RawArray("B", self.RGB_BYTES + 480 * 640 * 2) if retain_original else None
        self.color_size = context.RawValue("I", 0)
        self.lock, self.stop = context.Lock(), context.RawValue("b", False)
        self.size, self.serial, self.captured = context.RawValue("I", 0), context.RawValue("Q", 0), context.RawValue("d", 0)

    def publish(self, frame, source):
        rgb, depth, captured = frame
        payloads = source.get("_original_payloads")
        metadata = json.dumps({k: v for k, v in source.items() if k != "_original_payloads"}, allow_nan=False).encode()
        if self.retain_original and (payloads is None or not 0 < len(payloads["color"]) <= self.RGB_BYTES
                                     or len(payloads["depth"]) != 480 * 640 * 2):
            raise ValueError("original wrist payloads are missing or exceed the bounded slot")
        if len(metadata) > self.SOURCE_BYTES:
            raise ValueError("wrist capture metadata exceeds the bounded shared slot")
        while not self.stop.value:
            if not self.lock.acquire(timeout=.05):
                continue
            try:
                pixels = np.frombuffer(self.bytes, dtype=np.uint8)
                pixels[:self.RGB_BYTES] = rgb.reshape(-1)
                np.frombuffer(self.bytes, dtype=np.float32, count=480 * 640, offset=self.RGB_BYTES)[:] = depth.reshape(-1)
                start = self.RGB_BYTES + self.DEPTH_BYTES
                pixels[start:start + len(metadata)] = np.frombuffer(metadata, dtype=np.uint8)
                if self.retain_original:
                    raw = np.frombuffer(self.original, dtype=np.uint8)
                    size = len(payloads["color"])
                    raw[:size] = np.frombuffer(payloads["color"], dtype=np.uint8)
                    raw[self.RGB_BYTES:] = np.frombuffer(payloads["depth"], dtype=np.uint8)
                    self.color_size.value = size
                self.size.value, self.captured.value = len(metadata), captured
                self.serial.value += 1
                return
            finally:
                self.lock.release()

    def take(self, previous_serial):
        if not self.lock.acquire(timeout=.05):
            return None
        try:
            if self.serial.value == previous_serial:
                return None
            rgb = np.frombuffer(self.bytes, dtype=np.uint8, count=self.RGB_BYTES).reshape(480, 640, 3).copy()
            depth = np.frombuffer(self.bytes, dtype=np.float32, count=480 * 640, offset=self.RGB_BYTES).reshape(480, 640).copy()
            start = self.RGB_BYTES + self.DEPTH_BYTES
            metadata = bytes(self.bytes[start:start + self.size.value])
            captured, serial = self.captured.value, self.serial.value
            raw = np.frombuffer(self.original, dtype=np.uint8) if self.retain_original else None
            original = ({"color": raw[:self.color_size.value].tobytes(),
                         "depth": raw[self.RGB_BYTES:].tobytes()} if self.retain_original else None)
        finally:
            self.lock.release()
        rgb.flags.writeable = depth.flags.writeable = False
        source = json.loads(metadata)
        source["intrinsics"]["distortion"] = tuple(source["intrinsics"]["distortion"])
        if original is not None:
            source["_original_payloads"] = original
        return serial, (rgb, depth, captured), source


def _capture_progress(previous, source):
    if previous is not None:
        stable = ("capture_epoch", "intrinsics", "depth_units_m", "device_serial")
        if any(source[key] != previous[key] for key in stable) or source["source_sequence"] <= previous["source_sequence"]:
            raise ValueError("wrist capture restarted, changed geometry or failed to advance")


def _subscribe_wrist(path, config, slot, result):
    """Drain/validate in a spawned subscriber, independent of agent IPC's GIL."""
    reader_stop = threading.Event()

    def cancel_reader():
        # Do not leave a multiprocessing.Condition waiter behind when this
        # process exits on EOF; its notify acknowledgement can strand cleanup.
        while not slot.stop.value:
            reader_stop.wait(.05)
        reader_stop.set()

    threading.Thread(target=cancel_reader, name="wrist-subscriber-cancel", daemon=True).start()
    previous = None
    try:
        options = {"retain_payloads": True} if slot.retain_original else {}
        with closing(visiond_wire().latest_socket_sets(path, stop_event=reader_stop, **options)) as sets:
            for frame_set in sets:
                frame, source = decode_wrist(frame_set, config, unix_ns=time.time_ns(), monotonic_s=time.monotonic())
                _capture_progress(previous, source)
                previous = source
                slot.publish(frame, source)
        if not slot.stop.value:
            result.send("visiond wrist subscription ended")
    except Exception as error:
        if not slot.stop.value:
            result.send(f"{type(error).__name__}: {error}")
    finally:
        result.close()


def _capture_frames(path, config, stop_event, *, retain_original=False):
    """A cancellable subscriber, never the USB owner or an arm driver.

    Large agent JSON validation/serialization can hold the parent's GIL past
    the owner's 20-ms client write timeout. A thread in that interpreter cannot
    guarantee continuous draining. Spawn without inheriting driver descriptors;
    publish into a fixed latest slot and keep original capture times/metadata.
    EOF, validation errors and process death stay fatal; there is no reconnect.
    """
    context = multiprocessing.get_context("spawn")
    slot = _CaptureSlot(context, retain_original=retain_original)
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_subscribe_wrist, args=(path, config, slot, sender),
                             name="travel-wrist-subscriber", daemon=True)
    worker.start()
    sender.close()
    serial = 0
    try:
        while not stop_event.is_set():
            if receiver.poll():
                try:
                    message = receiver.recv()
                except EOFError:
                    message = "visiond wrist subscriber exited without a terminal report"
                raise RuntimeError(message)
            if not worker.is_alive():
                raise RuntimeError(f"visiond wrist subscriber exited with code {worker.exitcode}")
            item = slot.take(serial)
            if item is not None:
                serial, frame, source = item
                yield frame, source
            else:
                stop_event.wait(.01)
    finally:
        slot.stop.value = True
        worker.join(timeout=1.2)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=.5)
        receiver.close()
        if worker.is_alive():
            raise RuntimeError("visiond wrist subscriber process did not close within its deadline")
        worker.close()


class VisiondWristCamera:
    """The travel camera contract, backed by the shared cancellable wire subscription."""

    def __init__(self, path: Path = SOCKET, *, config: dict | None = None, retain_original=False):
        self.path, self.config = path, wrist_config() if config is None else config
        self.retain_original = retain_original
        self.stop, self.closing, self.lock = threading.Event(), threading.Event(), threading.Lock()
        self.frame, self.source, self.original, self.error = None, None, None, None
        self.worker = threading.Thread(target=self._receive, name="travel-visiond-camera", daemon=True)
        self.worker.start()

    def _receive(self) -> None:
        try:
            options = {"retain_original": True} if self.retain_original else {}
            with closing(_capture_frames(self.path, self.config, self.stop, **options)) as sets:
                for frame, source in sets:
                    self._accept(frame, source)
            if not self.closing.is_set():
                raise RuntimeError("visiond wrist subscription ended")
        except Exception as error:
            if not self.closing.is_set():
                with self.lock:
                    self.error = error

    def _accept(self, frame, source):
        with self.lock:
            _capture_progress(self.source, source)
            self.frame = frame
            self.source = {k: v for k, v in source.items() if k != "_original_payloads"}
            payloads = source.get("_original_payloads")
            self.original = dict(payloads) if payloads is not None else None

    def newest(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"visiond wrist capture failed: {self.error}") from self.error
            return self.frame

    def newest_with_metadata(self):
        """One atomic sample: later receive callbacks cannot relabel these frame bytes."""
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"visiond wrist capture failed: {self.error}") from self.error
            return self.frame, {"socket": str(self.path), **copy.deepcopy(self.source or {})}

    def newest_with_original(self):
        """One atomic frame/metadata/original-byte sample, only when explicitly retained."""
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"visiond wrist capture failed: {self.error}") from self.error
            original = dict(self.original) if self.original is not None else None
            return self.frame, {"socket": str(self.path), **copy.deepcopy(self.source or {})}, original

    def wait_ready(self, *, timeout_s: float = 30, stale_s: float = 0.3):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame, source = self.newest_with_metadata()
            if frame is not None and 0 <= time.monotonic() - frame[2] <= stale_s:
                return source
            self.stop.wait(0.05)
        raise RuntimeError("fresh aligned visiond wrist capture did not arrive before startup deadline")

    def metadata(self):
        with self.lock:
            return {"socket": str(self.path), **(self.source or {})}

    @property
    def intrinsics(self):
        source = self.metadata()
        if "intrinsics" not in source:
            raise RuntimeError("wrist calibration is unavailable before a validated capture")
        return Intrinsics(**source["intrinsics"])

    def close(self):
        self.closing.set()
        self.stop.set()
        self.worker.join(timeout=2)
        if self.worker.is_alive():
            raise RuntimeError("visiond wrist subscriber did not close within its deadline")
