"""Shared provenance metadata for Python Rerun producers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def source_commit(repo_root: Path = REPO_ROOT) -> str:
    explicit = os.environ.get("TATBOT_SOURCE_COMMIT")
    if explicit:
        return explicit
    manifest = repo_root / ".tatbot-deploy.json"
    if manifest.is_file():
        try:
            value = json.loads(manifest.read_text()).get("source_commit")
            if value:
                return str(value)
        except (OSError, ValueError):
            pass
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _calibration_id(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return json.loads(path.expanduser().read_text()).get("bundle_id")
    except (OSError, ValueError):
        return None


def _file_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return hashlib.sha256(path.expanduser().read_bytes()).hexdigest()
    except OSError:
        return None


def producer_metadata(
    workflow: str,
    recording_id: str | None,
    calibration: Path | None = None,
    urdf: Path | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "workflow": workflow,
        "recording_id": recording_id,
        "producer_host": socket.gethostname(),
        "producer_pid": os.getpid(),
        "started_unix_ns": time.time_ns(),
        "source_commit": source_commit(),
        "urdf_path": str(urdf.expanduser()) if urdf else None,
        "urdf_sha256": _file_hash(urdf),
        "calibration_id": _calibration_id(calibration),
    }


def log_producer_metadata(
    rerun_module,
    workflow: str,
    recording_id: str | None,
    calibration: Path | None = None,
    urdf: Path | None = None,
) -> dict:
    metadata = producer_metadata(workflow, recording_id, calibration, urdf)
    entity = re.sub(r"[^A-Za-z0-9_.-]+", "_", workflow)
    rerun_module.log(
        f"session/producers/{entity}",
        rerun_module.TextLog(json.dumps(metadata, sort_keys=True)),
        static=True,
    )
    return metadata
