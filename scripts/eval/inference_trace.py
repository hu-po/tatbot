"""Atomic policy-server evidence for one observation/action-chunk inference."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _safe_key(key: str) -> str:
    return "observation__" + "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in key
    )


def write_inference_trace(
    directory: str | Path,
    *,
    timestep: int,
    observation: dict[str, Any],
    observation_state: Any,
    normalized_action: Any,
    decoded_action: Any,
    fixed_noise_seed: str | None,
) -> dict[str, Any]:
    """Persist exactly what entered inference and what could reach the wire.

    A configured trace is a contract: any conversion or write failure raises
    before the server serializes actions.  The NPZ is intentionally uncompressed
    to keep the synchronous evidence gate bounded.
    """

    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    stem = f"inference-{int(timestep):09d}-{stamp}"
    destination = root / f"{stem}.npz"
    metadata_path = root / f"{stem}.json"

    arrays: dict[str, np.ndarray] = {
        "observation_state": _array(observation_state),
        "normalized_action": _array(normalized_action),
        "decoded_action": _array(decoded_action),
    }
    key_map: dict[str, str] = {}
    scalar_metadata: dict[str, Any] = {}
    for key, value in observation.items():
        if key == "_prev":
            continue
        if isinstance(value, (str, bytes, bool, int, float)):
            scalar_metadata[key] = (
                value.decode(errors="replace") if isinstance(value, bytes) else value
            )
            continue
        stored = _safe_key(key)
        key_map[stored] = key
        arrays[stored] = _array(value)

    handle, temporary = tempfile.mkstemp(dir=root, prefix=f".{stem}.", suffix=".npz")
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "kind": "policy inference evidence",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestep": int(timestep),
        "fixed_noise_seed": fixed_noise_seed,
        "npz": str(destination),
        "npz_sha256": digest,
        "observation_key_map": key_map,
        "observation_metadata": scalar_metadata,
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
    }
    handle, temporary = tempfile.mkstemp(dir=root, prefix=f".{stem}.", suffix=".json")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, metadata_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {**metadata, "metadata": str(metadata_path)}
