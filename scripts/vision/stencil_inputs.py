"""Recorded image manifests and existing visiond transports for one RGB view."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from stencil_surface import verify_pair
from surface_replay import recording_frames
from visiond_wire import latest_socket_sets


def _image_entry(root, row):
    image_path = (root/row["image"]).resolve()
    if image_path.stat().st_size > 32_000_000:
        raise ValueError("recorded image exceeds size limit")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode recorded image: {image_path}")
    frame = {"image": image, "timestamp_ns": int(row["timestamp_ns"]),
             "source_id": row.get("source_id", "image-manifest"),
             "provenance": {"image": str(image_path), "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()}}
    if "depth_m" in row:
        depth_path = (root/row["depth_m"]).resolve()
        if depth_path.stat().st_size > 128_000_000:
            raise ValueError("recorded depth exceeds size limit")
        frame["depth_m"] = np.load(depth_path, allow_pickle=False)
        if not np.issubdtype(frame["depth_m"].dtype, np.floating):
            raise ValueError("depth_m must contain floating-point meters")
        _provenance(frame, "depth_m", depth_path)
    if "metadata" in row:
        _metadata_entry(frame, root/row["metadata"], row["sensor"])
    if "camera_model" in row:
        model = root/row["camera_model"]
        if model.stat().st_size > 64_000:
            raise ValueError("camera model exceeds size limit")
        frame["camera_model"] = json.loads(model.read_text())
        _provenance(frame, "camera_model", model)
        frame["source_id"] += ":"+frame["provenance"]["camera_model"]["sha256"]
    return frame


def _provenance(frame, name, path):
    frame["provenance"][name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _metadata_entry(frame, path, sensor):
    if path.stat().st_size > 128_000:
        raise ValueError("RGB-D metadata exceeds size limit")
    metadata = [item["metadata"] for item in json.loads(path.read_text())["frames"]]
    frame["color_metadata"] = next(item for item in metadata if item["sensor_name"] == sensor)
    depth_sensor = sensor.removesuffix("_color")+"_depth"
    frame["depth_metadata"] = next(item for item in metadata if item["sensor_name"] == depth_sensor)
    signature = [frame["source_id"]]
    for item in (frame["color_metadata"], frame["depth_metadata"]):
        attrs = item.get("attributes", {})
        signature.append([item["profile"], attrs.get("intrinsics"), attrs.get("capture_epoch")])
    frame["source_id"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    _provenance(frame, "metadata", path)


def image_frames(index):
    """JSONL image paths, normalized timestamp_ns, optional float depth_m NPY.

Annotations/ground truth are deliberately not forwarded to the observer.
Paths may refer to existing evidence outside the manifest directory.
"""
    index = Path(index).expanduser().resolve()
    with index.open() as stream:
        for line in stream:
            if len(line) > 64_000:
                raise ValueError("image manifest row exceeds size limit")
            if line.strip():
                yield _image_entry(index.parent, json.loads(line))


def visiond_frames(index):
    for image, stamp, provenance, _ in recording_frames(Path(index).expanduser().resolve()):
        signature = [provenance[key] for key in ("sensor", "capture_epoch", "profile", "intrinsics")]
        yield {"image": image, "timestamp_ns": stamp, "source_id": json.dumps(signature, sort_keys=True),
               "provenance": provenance}


def aligned_depth(depth, color, sensor):
    if depth is None or depth.get("depth_units_m") is None:
        return None
    units = depth["depth_units_m"]
    depth_metadata = depth.get("depth_metadata", depth["metadata"])
    color_metadata = color.get("image_metadata", color["metadata"])
    attributes = depth_metadata.get("attributes", {})
    if not np.isfinite(units) or units <= 0 or attributes.get("aligned_to") != sensor:
        return None
    try:
        verify_pair(color_metadata, depth_metadata)
    except (ValueError, KeyError, TypeError):
        return None
    values = depth["depth"].astype(float)*units
    values[depth["depth"] == 65535] = 0
    return values


def owner_frames(socket, sensor, duration_s):
    """Subscribe to the existing owner; latest sets replace queued sets."""
    for frame_set in latest_socket_sets(Path(socket), duration_s=duration_s):
        frames = frame_set["frames"]
        color = frames.get(sensor)
        if color is None or "image" not in color:
            continue
        metadata = color.get("image_metadata", color["metadata"])
        attributes = metadata.get("attributes", {})
        signature = json.dumps([sensor, attributes.get("capture_epoch"), metadata["profile"],
                                attributes.get("intrinsics")], sort_keys=True)
        row = {"image": color["image"], "timestamp_ns": int(metadata["timestamps"]["normalized_unix_ns"]),
               "source_id": hashlib.sha256(signature.encode()).hexdigest(), "provenance": metadata}
        depth = frames.get(sensor.removesuffix("_color")+"_depth")
        row["depth_m"] = aligned_depth(depth, color, sensor)
        row["color_metadata"] = metadata
        if depth is not None:
            row["depth_metadata"] = depth.get("depth_metadata", depth["metadata"])
            row["provenance"] = dict(metadata, paired_depth_metadata=row["depth_metadata"])
        yield row
