"""Shared overhead owner capture and optics; no ROS or arm driver is opened."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OVERHEAD_TOPIC = "tatbot/vision/overhead-depth/capture"
COLOR, DEPTH = "overhead_depth_color", "overhead_depth_depth"
WORLD_FRAME = "overhead_depth_color_optical"


def _lib(repo: Path) -> None:
    for sub in ("scripts/lib", "scripts/vision"):
        path = str(repo / sub)
        if path not in sys.path:
            sys.path.insert(0, path)



def _depth_m(frame) -> np.ndarray | None:
    """A capture's aligned z16 depth in metres (its depth_units_m), or None without one."""
    if not frame:
        return None
    from visiond_wire import decode_depth

    metadata, _, data = frame
    return decode_depth(data, metadata["profile"]).astype(np.float32) * float(metadata["attributes"]["depth_units_m"])



def _decode(frame) -> np.ndarray:
    """A capture's colour payload as BGR, whatever encoding the owner published."""
    import cv2

    metadata, descriptor, data = frame
    profile = metadata["profile"]
    kind = str(descriptor.get("format", profile.get("format", ""))).lower()
    width, height = int(profile["width"]), int(profile["height"])
    if kind in ("jpeg", "jpg", "mjpeg"):
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    elif kind == "yuyv":
        image = cv2.cvtColor(np.frombuffer(data, np.uint8).reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUYV)
    elif kind in ("bgr8", "rgb8"):
        image = np.frombuffer(data, np.uint8).reshape(height, width, 3)
        image = image if kind == "bgr8" else image[:, :, ::-1]
    else:
        raise ValueError(f"{COLOR}: unhandled colour payload format {kind!r}")
    if image is None or image.shape[:2] != (height, width):
        raise ValueError(f"{COLOR}: colour payload does not decode to its {width}x{height} profile")
    return np.ascontiguousarray(image)



class Camera:
    """The D555 owner's capture queryable over the tatbot bus."""

    def __init__(self, repo: Path):
        _lib(repo)
        from tatbot_cli import nodes

        from tatbot_bridge.stack import open_bus

        mapping = nodes.load(repo)
        owners = nodes.nodes_with(mapping, "overhead-depth")
        if len(owners) != 1:
            raise RuntimeError(f"expected exactly one overhead-depth node in config/nodes.json, found {owners}")
        self.owner = owners[0]
        self.session = open_bus(nodes.bus_endpoint(mapping, address="lan"))

    def close(self) -> None:
        self.session.close()

    def capture(self, after_ns: int, timeout_s: float = 4.0, *, retain_original: bool = False) -> dict:
        """The newest colour/depth set exposed after `after_ns`, as {image, metadata, stamp_ns, depth_profile,
        depth_m}. With retain_original, original_packet is the exact validated frame-set reply."""
        from board_rgbd_evidence import unpack

        deadline = time.monotonic() + timeout_s
        last = "no reply"
        while time.monotonic() < deadline:
            window = {"after_ns": int(after_ns), "before_ns": time.time_ns() - 30_000_000}
            if window["before_ns"] <= window["after_ns"]:
                time.sleep(0.05)
                continue
            replies = list(self.session.get(OVERHEAD_TOPIC, payload=json.dumps(window), timeout=2.0))
            if not replies:
                last = "no reply"
            elif replies[0].err is not None:
                last = replies[0].err.payload.to_bytes().decode(errors="replace")
            else:
                original = replies[0].ok.payload.to_bytes()
                header, frames = unpack(original, max_bytes=64 * 2**20)
                producer = ((header.get("envelope") or {}).get("producer") or {}).get("node")
                if producer != self.owner:
                    raise RuntimeError(f"{OVERHEAD_TOPIC}: captured by {producer!r}, not the owner {self.owner!r}")
                if COLOR not in frames:
                    raise RuntimeError(f"{OVERHEAD_TOPIC}: the capture carries no {COLOR}")
                metadata = frames[COLOR][0]
                if DEPTH in frames and frames[DEPTH][0]['attributes'].get('aligned_to') != COLOR:
                    raise RuntimeError(f'{DEPTH}: capture depth is not aligned to {COLOR}')
                stamp = int(metadata["timestamps"]["normalized_unix_ns"])
                if stamp >= after_ns:
                    return {"image": _decode(frames[COLOR]), "metadata": metadata, "stamp_ns": stamp,
                            "depth_profile": (frames.get(DEPTH) or ({},))[0].get("profile"),
                            "depth_metadata": (frames.get(DEPTH) or (None,))[0],
                            "depth_m": _depth_m(frames.get(DEPTH)),
                            **({"original_packet": original} if retain_original else {})}
                last = "only older exposures"
            time.sleep(0.1)
        raise RuntimeError(f"{OVERHEAD_TOPIC}: no capture exposed after the hold within {timeout_s:g} s ({last})")



def intrinsics_of(metadata: dict) -> dict:
    """The active colour optics the owner published with the frame (`tatbot.camera-intrinsics/1`)."""
    intrinsics = json.loads(metadata["attributes"]["intrinsics"])
    if intrinsics.get("schema") != "tatbot.camera-intrinsics/1":
        raise ValueError(f"{COLOR}: frame optics are {intrinsics.get('schema')!r}, not tatbot.camera-intrinsics/1")
    return intrinsics



def k_dist(intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    k = np.array([[intrinsics["fx"], 0.0, intrinsics["ppx"]], [0.0, intrinsics["fy"], intrinsics["ppy"]],
                  [0.0, 0.0, 1.0]])
    return k, np.array((list(intrinsics.get("distortion_coefficients") or []) + [0.0] * 5)[:5], float)
