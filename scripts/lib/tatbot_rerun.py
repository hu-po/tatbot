"""The one Rerun contract for every Python producer (docs/vision.md).

Every tatbot workflow streams into ONE persistent viewer (`tatbot viewer`) under
ONE application id; a recording id joins the producers of a session, and the
entity-path prefixes below say what each producer may log. A producer never
spawns a viewer of its own: it connects to the proxy it was given (or writes an
.rrd), states its rate cap, and stamps the shared wall-clock timeline.

    import tatbot_rerun as tr
    ap = argparse.ArgumentParser(); tr.add_sink_args(ap)
    rr = tr.start("stencil_observe", args, calibration=..., urdf=...)
    tr.set_capture_time(t_ns)
    rr.log("surface/status", rr.TextLog("..."))

Import it with `scripts/lib` on sys.path (scripts/vision/stencil_rerun.py is one
producer that does); launchers that copy a producer to another node copy this file too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

APP_ID = "tatbot"
"""One application id for the whole project. Rerun keys blueprints by application
id, so one id means one display-owned blueprint per viewer."""

TIMELINE = "capture_time"
"""Wall-clock nanoseconds since the epoch, on every producer on every node, so
multi-node data lines up (only as well as their NTP does)."""

STANDING_RECORDING = "live-cameras"
"""The standing recording on the rerun-server node: the fixed blueprint is sent
into it at every server start (`rerun_viewer::STANDING_RECORDING`)."""

_STAMP = re.compile(r"(?:^|(?<=-))(\d{8})[T_-](\d{2})(\d{2})\d{2}Z?(?=-|$)")
"""The `YYYYMMDD[T_-]HHMMSS[Z]` stamp every minted recording id carries: run ids
lead with it, workflow ids follow a prefix. Same rule as `rerun_viewer::recording_stamp`."""

PORT = 9876
"""The gRPC proxy port of the persistent viewer (scripts/vision/rerun_session.sh)."""

ENTITY_PREFIXES = {
    "cameras": "camera images: cameras/NN_<name>/{image|color|depth} (visiond)",
    "robot": "URDF links and animated transforms (visiond stream-teleop / replay)",
    "world": "calibration frame, frustums, zones, reconstructed surface",
    "teleop": "teleop scalars: leader/follower joints, efforts, timing",
    "surface": "surface images and status: stencil tracking, surface replay, RGB-D surfaces",
    "calibration": "calibration evidence: the stencil camera's measured depth and candidate geometry",
    "draw": "drawing: the ROS 2 stack's plan and progress (draw/ros, ros/tatbot_rerun)",
    "session": "producer provenance (session/producers/<workflow>) and status",
}


# --- addressing ---------------------------------------------------------------------


def lan_ip() -> str | None:
    """This host's address on the rig subnet (config/nodes.json `__rig__`), or
    TATBOT_RERUN_LAN_IP; None without one. One implementation, in
    tatbot_cli.nodes, so producers and the CLI publish the same address."""
    from tatbot_cli import nodes  # scripts/lib is on sys.path wherever this module is importable
    return nodes.lan_ip(Path(__file__).resolve().parents[2])


def proxy_url(host: str | None = None, port: int = PORT) -> str:
    """The gRPC proxy URL of the persistent viewer on `host` (default: this host on the LAN)."""
    return f"rerun+http://{host or lan_ip() or '127.0.0.1'}:{port}/proxy"


def default_connect() -> str | None:
    """Where a producer streams when the launcher did not say: TATBOT_RERUN_CONNECT."""
    return os.environ.get("TATBOT_RERUN_CONNECT") or None


def recording_id(prefix: str) -> str:
    """A session recording id: <prefix>-<utc stamp>; every producer of one session shares it.

    The stamp is UTC like a run id's, so the name the viewer shows for it
    reads on the same clock as its time panel.
    """
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def recording_name(recording_id: str | None) -> str | None:
    """What the viewer's source list shows for a recording, from its id alone.

    Every producer joining an id sends the same value, so the static name never
    flips between them (`rerun_viewer::recording_name` is the Rust twin; the
    tests hold both to one vector). A bare run id names nothing here: the
    session that owns it sends the session's name. Without this Rerun shows
    `<unknown>`, because `rr.init(recording_id=...)` sends no default properties.
    """
    if not recording_id:
        return None
    if recording_id == STANDING_RECORDING:
        return "Rig preview"
    stamp = _STAMP.search(recording_id)
    if stamp and stamp.start() == 0:
        return None
    prefix = recording_id[:stamp.start() - 1] if stamp else recording_id
    words = prefix.replace("-", " ").strip()
    if not words:
        return None
    words = words[0].upper() + words[1:]
    return f"{words} {stamp.group(2)}:{stamp.group(3)}" if stamp else words


# --- argparse -----------------------------------------------------------------------


def add_sink_args(parser: argparse.ArgumentParser, *, require: bool = False) -> None:
    """--connect URL | --output FILE, plus --recording-id. `require`: one sink must be stated."""
    group = parser.add_mutually_exclusive_group(required=require)
    group.add_argument("--connect", metavar="URL",
                       help=f"stream to the persistent viewer's proxy (rerun+http://HOST:{PORT}/proxy); "
                            "default TATBOT_RERUN_CONNECT")
    group.add_argument("--output", metavar="RRD", help="write an .rrd instead of streaming")
    parser.add_argument("--recording-id", metavar="ID",
                        help="join this session's shared recording (other producers use the same id)")


# --- provenance -----------------------------------------------------------------


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
        return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _calibration_id(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return json.loads(Path(path).expanduser().read_text()).get("bundle_id")
    except (OSError, ValueError):
        return None


def _file_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()
    except OSError:
        return None


def producer_metadata(workflow: str, recording_id: str | None, calibration: Path | None = None,
                      urdf: Path | None = None) -> dict:
    return {
        "schema_version": 1,
        "workflow": workflow,
        "recording_id": recording_id,
        "application_id": APP_ID,
        "producer_host": socket.gethostname(),
        "producer_pid": os.getpid(),
        "started_unix_ns": time.time_ns(),
        "source_commit": source_commit(),
        "urdf_path": str(Path(urdf).expanduser()) if urdf else None,
        "urdf_sha256": _file_hash(urdf),
        "calibration_id": _calibration_id(calibration),
    }


def log_producer_metadata(rerun_module, workflow: str, recording_id: str | None,
                          calibration: Path | None = None, urdf: Path | None = None) -> dict:
    """Static TextLog at session/producers/<workflow>: host, pid, commit, URDF hash, calibration id."""
    metadata = producer_metadata(workflow, recording_id, calibration, urdf)
    entity = re.sub(r"[^A-Za-z0-9_.-]+", "_", workflow)
    rerun_module.log(f"session/producers/{entity}", rerun_module.TextLog(json.dumps(metadata, sort_keys=True)),
                     static=True)
    return metadata


# --- the stream ----------------------------------------------------------------------


class SinkError(RuntimeError):
    """No sink: neither --connect/--output nor TATBOT_RERUN_CONNECT. A producer never spawns a viewer."""


def start(workflow: str, args: argparse.Namespace | None = None, *, connect: str | None = None,
          output: str | os.PathLike | None = None, recording_id: str | None = None,
          calibration: Path | None = None, urdf: Path | None = None, view_coordinates: bool = True):
    """Init the shared recording, attach the sink(s), log provenance; returns the `rerun` module.

    Sinks: --connect streams to the persistent viewer, --output writes an .rrd, both at once
    write and stream (`set_sinks`). Nothing here ever spawns a viewer.
    """
    import rerun as rr  # noqa: PLC0415  (after argparse so --help needs no SDK)

    if args is not None:
        connect = connect or getattr(args, "connect", None)
        output = output or getattr(args, "output", None)
        recording_id = recording_id or getattr(args, "recording_id", None)
    connect = connect or (default_connect() if not output else None)
    if not connect and not output:
        raise SinkError(f"{workflow}: no Rerun sink — pass --connect rerun+http://HOST:{PORT}/proxy "
                        "(the persistent viewer; `tatbot viewer status`) or --output FILE.rrd")

    rr.init(APP_ID, recording_id=recording_id)
    if connect and output:
        rr.set_sinks(rr.GrpcSink(url=connect), rr.FileSink(path=str(output)))
    elif connect:
        rr.connect_grpc(connect)
    else:
        rr.save(str(output))
    # An explicit recording id switches the SDK's default properties off, so
    # the shared name goes explicitly. A fresh id, or a file no other producer
    # joins, is named after this producer; a run id streamed to the fleet is
    # left to the session that owns it.
    name = recording_name(recording_id)
    if name is None and (not recording_id or not connect):
        name = workflow
    if name:
        rr.send_recording_name(name)
    log_producer_metadata(rr, workflow, recording_id, calibration, urdf)
    if view_coordinates:
        try:
            # Which way is up for the 3D view. Some rerun/numpy ABI pairs reject
            # this one batch type; nothing else depends on it.
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        except Exception as error:  # noqa: BLE001
            print(f"  (view-coordinates hint skipped: {error})")
    return rr


def set_capture_time(t_ns: int) -> None:
    """Stamp the shared wall-clock timeline (nanoseconds since the epoch)."""
    import numpy as np
    import rerun as rr  # noqa: PLC0415

    rr.set_time(TIMELINE, timestamp=np.datetime64(int(t_ns), "ns"))


class RateLimiter:
    """`ready()` is true at most `fps` times a second; 0 means never (the pane stays idle)."""

    def __init__(self, fps: float):
        self.interval = 1.0 / fps if fps > 0 else None
        self._last = 0.0

    def ready(self) -> bool:
        if self.interval is None:
            return False
        now = time.monotonic()
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False
