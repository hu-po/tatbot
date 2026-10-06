"""Per-arm identities, the native arm owner's build and telemetry, and the retained per-arm captures.

The physical arm behind an operator label (pink, blue), its controller role
and wrist tag target resolve here for `tatbot calib pose` and `tatbot calib
joint-measure`, with the native owner's build inputs and its seven-axis
telemetry record. The hand-guided per-arm recipe that captured tip, wrist and
palette against the three-tip fixture is retired with the fixture, and the rig
bundle solve that read its retained captures went with the overhead cameras
(2026-09-29).

Stdlib only: the CLI imports it to plan and refuse before any network or
driver activity.
"""
from __future__ import annotations

import bisect
import ipaddress
import json
import math
import struct
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from tatbot_cli import arms as arm_registry
from tatbot_digest import sha256_file as digest

CONTROLLER_ROLE = {"config/trossen/leader.yaml": "leader", "config/trossen/follower.yaml": "follower"}
WRIST_TARGET = {"right": "wrist", "left": "wrist_left"}

# Seven-axis telemetry record shared with rust/tatbot-arm/src/guide.rs:
# tick, wall ns, monotonic ns, positions[7], velocities[7], external efforts[7],
# controller mode (0 idle, 1 position, 2 fault, 3 hand-guiding), e-stop word, flags.
TELEMETRY_RECORD = struct.Struct("<QQQ21dBiB")
TELEMETRY_MAGIC = b"TBGUIDE1"


def native_estop_device(repo: Path, device: str) -> str:
    """Normalize a native relay URI using the execution owner's canonical role map."""
    if not device.startswith('udp://'):
        return device
    from tatbot_cli import nodes

    url = urllib.parse.urlsplit(device)
    query = urllib.parse.parse_qs(url.query, keep_blank_values=True, strict_parsing=True)
    if (set(query) != {'from'} or len(query['from']) != 1 or not query['from'][0]
            or not url.port or url.path or url.fragment or url.username is not None):
        raise ValueError('UDP e-stop needs udp://[IPv4]:PORT?from=<relay IPv4 or role>')
    listen = ipaddress.IPv4Address(url.hostname or '0.0.0.0')
    source = query['from'][0]
    try:
        relay = ipaddress.IPv4Address(source)
    except ipaddress.AddressValueError:
        mapping = nodes.load(repo)
        lan = mapping[nodes.require_role(mapping, source)].get('lan')
        if not lan:
            raise ValueError('UDP e-stop relay role has no LAN address') from None
        relay = ipaddress.IPv4Address(lan)
    return f'udp://{listen}:{url.port}?from={relay}'


def native_source_digests(repo: Path) -> dict:
    """Exact build inputs shared with rust/tatbot-arm/build.rs; target independent."""
    paths = ("rust/Cargo.toml", "rust/Cargo.lock", "rust/tatbot-arm/Cargo.toml", "rust/tatbot-arm/build.rs",
             "rust/tatbot-arm/src", "rust/trossen-arm-sys/Cargo.toml", "rust/trossen-arm-sys/build.rs",
             "rust/trossen-arm-sys/src", "rust/trossen-arm-sys/include")
    files = []
    for name in paths:
        path = repo / name
        files.extend(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else files.append(path)
    return {str(path.relative_to(repo)): digest(path) for path in sorted(files)}


class RecipeError(ValueError):
    """A selection or retained capture that must be refused before anything uses it."""


@dataclass(frozen=True)
class SelectedArm:
    label: str
    arm_id: str
    controller_role: str
    controller_config: str
    profile_ip_field: str
    sdk_end_effector: str
    base_link: str
    tool_mount: str
    wrist_target: str
    tag_parent: str
    wrist_layout: str
    other_arm_id: str


def write_json(path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def arm_labels(repo: Path) -> dict[str, str]:
    document = json.loads((Path(repo) / "config/arm-labels.json").read_text())
    if document.get("schema") != "tatbot.arm-labels/1" or set(document.get("aliases", {}).values()) != set(arm_registry.ARM_IDS):
        raise RecipeError("config/arm-labels.json must alias exactly left and right")
    return dict(document["aliases"])


def selected_arm(repo: Path, label: str) -> SelectedArm:
    """The physical arm behind an operator label; nothing here opens a controller."""
    aliases = arm_labels(repo)
    if label not in aliases:
        raise RecipeError(f"unknown physical arm {label!r}; choose from {', '.join(sorted(aliases))}")
    arm_id = aliases[label]
    physical = arm_registry.load(Path(repo))[arm_id]
    role = CONTROLLER_ROLE.get(physical.controller_config)
    if role is None:
        raise RecipeError(f"{arm_id}: controller config {physical.controller_config} has no known controller role")
    target = fiducial_target(repo, WRIST_TARGET[arm_id])
    parent = target.get("parent_frame") or ""
    if not parent.startswith(physical.urdf_prefix + "/"):
        raise RecipeError(f"{WRIST_TARGET[arm_id]}: parent frame {parent!r} is not on the {arm_id} arm")
    return SelectedArm(label=label, arm_id=arm_id, controller_role=role,
                       controller_config=physical.controller_config,
                       profile_ip_field=physical.profile_ip_field,
                       sdk_end_effector=physical.sdk_end_effector,
                       base_link=f"{physical.urdf_prefix}/base_link",
                       tool_mount=f"{physical.urdf_prefix}/tool_mount",
                       wrist_target=WRIST_TARGET[arm_id], tag_parent=parent,
                       wrist_layout=target.get("layout") or "",
                       other_arm_id="left" if arm_id == "right" else "right")


def fiducial_target(repo: Path, name: str) -> dict:
    inventory = json.loads((Path(repo) / "config/fiducials.json").read_text())
    try:
        return inventory["targets"][name]
    except KeyError as error:
        raise RecipeError(f"config/fiducials.json has no target {name!r}") from error


def read_telemetry(path: Path) -> list[dict]:
    """Every retained tick, in order, checked for finiteness and monotonic time."""
    data = Path(path).read_bytes()
    if len(data) < 16 or data[:8] != TELEMETRY_MAGIC or struct.unpack_from("<Q", data, 8)[0] != TELEMETRY_RECORD.size:
        raise RecipeError("unknown telemetry format")
    body = data[16:]
    if len(body) % TELEMETRY_RECORD.size:
        raise RecipeError("truncated telemetry record")
    rows = []
    previous = None
    for values in TELEMETRY_RECORD.iter_unpack(body):
        tick, wall_ns, mono_ns = values[:3]
        axes = values[3:24]
        if any(not math.isfinite(v) for v in axes):
            raise RecipeError("nonfinite telemetry")
        if previous is not None and (tick <= previous[0] or mono_ns <= previous[1]):
            raise RecipeError("telemetry ticks reversed or duplicated")
        previous = (tick, mono_ns)
        rows.append({"tick": tick, "wall_ns": wall_ns, "mono_ns": mono_ns,
                     "positions": list(axes[0:7]), "velocities": list(axes[7:14]), "efforts": list(axes[14:21]),
                     "mode": values[24], "estop": values[25], "flags": values[26]})
    if not rows:
        raise RecipeError("no retained telemetry")
    return rows


def unchanged_feedback_ns(rows: list[dict]) -> int:
    """Longest run of an entirely unchanged seven-axis tuple; the freshness screen."""
    longest, run_start = 0, rows[0]["mono_ns"]
    previous = None
    for row in rows:
        current = (tuple(row["positions"]), tuple(row["velocities"]), tuple(row["efforts"]))
        if current != previous:
            previous, run_start = current, row["mono_ns"]
        longest = max(longest, row["mono_ns"] - run_start)
    return longest


def bind_wrist_capture(capture: Path, telemetry: Path) -> dict:
    """Retain nominal exposure/telemetry timing without adopting a calibration."""
    import numpy as np

    rows = read_telemetry(telemetry)
    with np.load(capture, allow_pickle=False) as data:
        roles = json.loads(str(data['camera_roles']))
        stamps = []
        for role in roles:
            frames = json.loads(str(data[f'owner_frames_{role}']))
            color = json.loads(str(data[f'owner_color_metadata_{role}']))
            for kind, frame in [*[('depth', f['metadata']) for f in frames], ('color', color)]:
                stamp = frame['timestamps']['normalized_unix_ns']
                if type(stamp) is not int or stamp <= 0:
                    raise RecipeError('capture lacks a normalized exposure timestamp')
                stamps.append((role, kind, frame['sequence'], stamp))
    return {'schema': 'tatbot.current-pose-wrist-binding/1',
            'source_capture_sha256': digest(capture), 'source_telemetry_sha256': digest(telemetry),
            **_bind_stamps(rows, stamps)}


def bind_owner_packet(capture: Path, telemetry: Path) -> dict:
    """Bind the original frame-set exposures, never a decoded image's save time."""
    from board_rgbd_evidence import unpack

    header, frames = unpack(Path(capture).read_bytes(), max_bytes=64 * 2**20)
    stamps = [(sensor, metadata['profile']['stream'], metadata['sequence'],
               metadata['timestamps']['normalized_unix_ns'])
              for sensor, (metadata, _, _) in frames.items()]
    return {'schema': 'tatbot.current-pose-owner-binding/1',
            'source_capture_sha256': digest(capture), 'source_telemetry_sha256': digest(telemetry),
            'producer': (header.get('envelope') or {}).get('producer'),
            **_bind_stamps(read_telemetry(telemetry), stamps)}


def _bind_stamps(rows, stamps):
    if not stamps or any(type(stamp) is not int or stamp <= 0 for _, _, _, stamp in stamps):
        raise RecipeError('capture lacks retained frames with normalized exposure timestamps')
    wall = [row['wall_ns'] for row in rows]
    if any(b <= a for a, b in zip(wall, wall[1:], strict=False)):
        raise RecipeError('telemetry wall clock is not strictly ordered')
    pairs = [_bind_exposure(rows, wall, role, kind, seq, stamp) for role, kind, seq, stamp in stamps]
    lo, hi = min(stamp for _, _, _, stamp in stamps), max(stamp for _, _, _, stamp in stamps)
    interval = rows[max(0, bisect.bisect_right(wall, lo)-1):bisect.bisect_left(wall, hi)+1]
    return {'motion_authority': False,
            'basis': 'nearest native host-time telemetry to normalized owner exposure',
            'physical_accuracy_bound_m': None, 'world_registration': 'unmeasured',
            'pairs': pairs, 'maximum_skew_ms': max(pair['skew_ms'] for pair in pairs),
            'capture_joint_span_rad': [max(r['positions'][i] for r in interval)-min(r['positions'][i] for r in interval)
                                       for i in range(6)],
            'capture_max_measured_velocity_rad_s': max(abs(v) for r in interval for v in r['velocities'][:6])}


def _bind_exposure(rows, wall, role, kind, sequence, stamp):
    index = bisect.bisect_left(wall, stamp)
    if index == 0 or index == len(rows):
        raise RecipeError('owner exposure is not bracketed by retained arm telemetry')
    before, after = rows[index-1], rows[index]
    row = min((before, after), key=lambda r: abs(r['wall_ns']-stamp))
    return {'camera_role': role, 'stream': kind, 'source_sequence': sequence, 'exposure_wall_ns': stamp,
            'bracket_wall_ns': [before['wall_ns'], after['wall_ns']],
            'bracket_gap_ms': (after['wall_ns']-before['wall_ns'])/1e6,
            'telemetry_tick': row['tick'], 'measured_wall_ns': row['wall_ns'],
            'skew_ms': abs(row['wall_ns']-stamp)/1e6, 'joints_rad': row['positions'][:6],
            'carriage_m': row['positions'][6], 'mode': row['mode'], 'estop': row['estop']}
