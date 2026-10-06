"""Resolve physical wrist cameras from the vision registry without opening devices."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import tomllib
from tatbot_cli import arms as arm_registry
from tatbot_cli import nodes

SENSOR_PROFILES = ('deployment', 'legacy-two-view')

REPO = Path(__file__).resolve().parents[2]


def capture_roles(repo: Path, config: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Configured wrist roles, including arms with no mounted wrist view."""
    result = {arm: [] for arm in arm_registry.load(repo)}
    for camera in registry(repo, config):
        if camera.get('arm') is not None:
            result[camera['arm']].append(camera['role'])
    return {arm: tuple(roles) for arm, roles in result.items()}


def capture_arm(roles, *, repo: Path = REPO, config: Path | None = None) -> str:
    """The one physical arm a capture's declared roles belong to."""
    roles = list(roles)
    owners = {role: arm for arm, views in capture_roles(repo, config).items() for role in views}
    if not roles or len(set(roles)) != len(roles) or any(role not in owners for role in roles):
        raise ValueError('invalid capture camera roles')
    mounted = {owners[role] for role in roles}
    if len(mounted) != 1:
        raise ValueError("invalid capture camera roles: one arm's wrist cameras only")
    return mounted.pop()


@dataclass(frozen=True)
class CameraDescription:
    """Portable camera geometry and features; contains no device credentials."""

    role: str
    arm: str
    optical_frame: str
    depth_optical_frame: str
    width: int
    height: int
    fps: float
    intrinsic: tuple[float, float, float, float]
    intrinsic_basis: str
    geometry_basis: str = 'nominal-urdf'

    def as_dict(self) -> dict:
        return asdict(self)


def describe(repo: Path, *, arms: tuple[str, ...] = ('right',),
             profile: str = 'deployment', config: Path | None = None) -> tuple[CameraDescription, ...]:
    """Resolve render/feature descriptions without checking device ownership.

    Live capture still uses ``for_arm`` to enforce ownership. Streams here are
    aligned RGB-D; differing stream dimensions/rates need an explicit alignment
    adapter. When intrinsics are absent, the nominal vertical FOV is recorded
    rather than being represented as measured calibration.
    """
    configured = arm_registry.load(repo)
    if not arms or len(set(arms)) != len(arms) or any(a not in configured for a in arms):
        raise ValueError('camera description requires distinct configured arms')
    if profile not in SENSOR_PROFILES:
        raise ValueError(f'unknown sensor profile {profile!r}')
    if profile == 'legacy-two-view':
        if arms != ('right',) or config is not None:
            raise ValueError('legacy-two-view describes the historical follower only')
        # Frozen historical observation geometry, independent of today's rig.
        focal = 480 / (2 * math.tan(0.96 / 2))
        return tuple(CameraDescription(
            role=role, arm='right', optical_frame=f'right/{stem}_color_optical_frame',
            depth_optical_frame=f'right/{stem}_color_optical_frame',
            width=640, height=480, fps=30.0,
            intrinsic=(focal, focal, 320.0, 240.0), intrinsic_basis='nominal-fov',
            geometry_basis='historical-two-view',
        ) for role, stem in (('wrist_upper', 'camera'), ('wrist_lower', 'camera_lower')))
    cameras = registry(repo, config)
    if any(camera.get('arm') is None for camera in cameras):
        raise ValueError('D405 physical-arm assignment is incomplete')
    result = []
    for arm in arms:
        selected = [camera for camera in cameras if camera['arm'] == arm]
        if selected:
            color_frames = optical_frames(repo, arm=arm, stream='color', config=config)
            depth_frames = optical_frames(repo, arm=arm, stream='depth', config=config)
            result.extend(_describe_camera(camera, arm, color_frames[camera['role']],
                                           depth_frames[camera['role']]) for camera in selected)
    return tuple(result)


def _describe_camera(camera: dict, arm: str, optical: str, depth_optical: str) -> CameraDescription:
    color, depth = camera['color'], camera['depth']
    fields = ('width', 'height', 'fps_num', 'fps_den')
    if any(type(color.get(k)) is not int or color[k] <= 0 for k in fields):
        raise ValueError(f"{camera['role']}: invalid RGB profile")
    if any(color[k] != depth.get(k) for k in fields):
        raise ValueError(f"{camera['role']}: aligned RGB-D dimensions and rates must match")
    intrinsic, basis = _intrinsics(camera)
    return CameraDescription(
        role=camera['role'], arm=arm, optical_frame=optical,
        depth_optical_frame=depth_optical, width=color['width'], height=color['height'],
        fps=color['fps_num'] / color['fps_den'],
        intrinsic=intrinsic, intrinsic_basis=basis,
    )


def _intrinsics(camera: dict) -> tuple[tuple[float, ...], str]:
    color = camera['color']
    intrinsic = color.get('intrinsics')
    basis = 'declared-intrinsics'
    if intrinsic is None:
        fovy = float(color.get('nominal_fov_y_rad', 0.96))
        if not math.isfinite(fovy) or not 0 < fovy < math.pi:
            raise ValueError(f"{camera['role']}: invalid nominal FOV")
        focal = color['height'] / (2 * math.tan(fovy / 2))
        intrinsic = (focal, focal, color['width'] / 2, color['height'] / 2)
        basis = 'nominal-fov'
    if (not isinstance(intrinsic, (list, tuple)) or len(intrinsic) != 4
            or not all(math.isfinite(float(x)) for x in intrinsic)
            or min(float(intrinsic[0]), float(intrinsic[1])) <= 0):
        raise ValueError(f"{camera['role']}: invalid camera intrinsics")
    return tuple(float(x) for x in intrinsic), basis


def common_stream_profile(cameras: tuple[CameraDescription, ...], *, control_freq: float) -> tuple[int, int]:
    """Dimensions for a synchronous dataset writer; reject implicit resampling."""
    sizes = {(camera.width, camera.height) for camera in cameras}
    if len(sizes) != 1 or any(camera.fps != control_freq for camera in cameras):
        raise ValueError("dataset cameras require common dimensions and the control cadence")
    return sizes.pop()


def validate_checkpoint_views(checkpoint: dict, roles: tuple[str, ...], *, use_depth: bool,
                              image_shapes: dict[str, tuple[int, int, int]] | None = None) -> None:
    """Compare exact feature identities; never synthesize a missing view."""
    features = checkpoint.get('policy', checkpoint).get('input_features', {})
    expected = {name.removeprefix('observation.images.') for name in features
                if name.startswith('observation.images.')}
    actual = set(roles)
    if use_depth:
        actual |= {f'{role}_depth' for role in roles}
    if expected != actual:
        raise ValueError(f'checkpoint camera keys {sorted(expected)} do not match '
                         f'the selected views {sorted(actual)}; no view substitution is allowed')
    if image_shapes is not None:
        if set(image_shapes) != actual:
            raise ValueError('image shape declarations differ from selected views')
        for role, shape in image_shapes.items():
            declared = features[f'observation.images.{role}'].get('shape')
            if declared != list(shape):
                raise ValueError(f'{role}: checkpoint camera shape {declared} differs from {shape}')


def read_checkpoint_config(policy: str, config: Path | None = None) -> tuple[Path, dict]:
    """Read a local config, including an explicit copy for a server-side model.

    LeRobot's policy setup response does not expose the loaded checkpoint's
    input features. Require its config locally so a mismatch refuses before
    the first action query. The evaluation records these exact config bytes.
    """
    path = Path(config or policy).expanduser()
    if path.is_dir():
        path = path / 'config.json'
        if not path.is_file():
            path = path.parent / 'train_config.json'
    if not path.is_file():
        raise ValueError('checkpoint config unavailable locally; provide --checkpoint-config '
                         'with the config.json for the selected server-side checkpoint')
    return path.resolve(), json.loads(path.read_text())


def fixed_chain(repo: Path, parent: str, child: str, *, urdf: Path | None = None) -> tuple:
    """Parent-to-child fixed joint origins, in order, without a physics engine."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf or repo / 'urdf/tatbot.urdf').getroot()
    parents = {joint.find('child').get('link'): joint for joint in tree.findall('joint')}
    result, seen = [], set()
    cursor = child
    while cursor != parent:
        if cursor in seen or cursor not in parents or parents[cursor].get('type') != 'fixed':
            raise ValueError(f'{child}: no fixed camera chain to {parent}')
        seen.add(cursor)
        joint = parents[cursor]
        origin = joint.find('origin')
        xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
        rpy = origin.get('rpy', '0 0 0') if origin is not None else '0 0 0'
        result.append((tuple(map(float, xyz.split())), tuple(map(float, rpy.split()))))
        cursor = joint.find('parent').get('link')
    return tuple(reversed(result))


def registry_path(repo: Path) -> Path:
    """The deployment's vision registry, or the checked-in example when the
    checkout has none: vision.toml is the rig's profile and stays out of the
    public export, whose copy of this module (and arm_kinematics, which reads
    the wrist frames at import) must still resolve the stock right-arm camera.
    The example names no real device (serial 000000000000), so nothing that
    opens a camera can mistake it for the rig."""
    path = repo / "rust/visiond/config/vision.toml"
    return path if path.is_file() else repo / "rust/visiond/config/vision.example.toml"


def registry(repo: Path, config: Path | None = None) -> list[dict]:
    path = config or registry_path(repo)
    cameras = tomllib.loads(path.read_text()).get("cameras", {}).get("realsense", [])
    selected = [c for c in cameras if c.get("group") == "d405"]
    configured = arm_registry.load(repo)
    for field in ("name", "serial", "role"):
        values = [c.get(field) for c in selected]
        if (any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"D405 registry requires distinct nonempty {field} values")
    for camera in selected:
        for field in ("name", "role", "owner_role"):
            value = camera.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ValueError(f"{camera['name']}: missing or invalid {field}")
        if camera.get("arm") is not None and camera["arm"] not in configured:
            raise ValueError(f"{camera['name']}: camera arm is not configured")
    return selected


def mounted_for_arm(repo: Path, arm: str, config: Path | None = None) -> list[dict]:
    """The selected configured arm's physical wrist cameras, possibly none.

    A missing mount is an acquisition capability deficit for a caller that
    needs wrist depth; it does not make the arm ID itself invalid.
    """
    if arm not in arm_registry.load(repo):
        raise ValueError(f"physical camera arm {arm!r} is not configured")
    cameras = registry(repo, config)
    if any(camera.get('arm') is None for camera in cameras):
        raise ValueError('D405 physical-arm assignment is incomplete; identify the moved camera before recording')
    return [camera for camera in cameras if camera['arm'] == arm]


def owner(camera: dict, mapping: dict) -> str:
    owners = nodes.nodes_with(mapping, camera["owner_role"])
    if len(owners) != 1:
        raise ValueError(f"{camera['name']}: capture role {camera['owner_role']} needs exactly one owner")
    return owners[0]


def owned_names(repo: Path, node: str, config: Path | None = None) -> list[str]:
    """Select only this capture host's devices; no first-node or first-camera fallback."""
    mapping = nodes.load(repo)
    selected = [c["name"] for c in registry(repo, config) if owner(c, mapping) == node]
    if not selected:
        raise ValueError("this node owns no manifested D405 cameras")
    return selected


def for_arm(repo: Path, arm: str, node: str, config: Path | None = None) -> list[dict]:
    selected = mounted_for_arm(repo, arm, config)
    if not selected:
        raise ValueError(f"no D405 camera is assigned to the {arm} arm")
    mapping = nodes.load(repo)
    for camera in selected:
        if owner(camera, mapping) != node:
            raise ValueError(f"{camera['name']}: {arm} wrist camera belongs to another capture host; "
                             "local LeRobot capture cannot open it")
    return selected


def lerobot_config(repo: Path, arm: str, node: str, *, use_depth: bool,
                   checkpoint: dict | None = None) -> str:
    """Use local arm views only; refuse a checkpoint trained with different RGB keys."""
    cameras = for_arm(repo, arm, node)
    result = {}
    for camera in cameras:
        color, depth = camera["color"], camera["depth"]
        if (color["fps_den"] != 1 or (use_depth and any(
                color[key] != depth[key] for key in ("width", "height", "fps_num", "fps_den")))):
            raise ValueError(f"{camera['name']}: unsupported LeRobot RGBD profile")
        result[camera["role"]] = {
            "type": "intelrealsense", "serial_number_or_name": camera["serial"],
            "width": color["width"], "height": color["height"],
            "fps": color["fps_num"], "use_depth": use_depth,
        }
    if checkpoint is not None:
        validate_checkpoint_views(checkpoint, tuple(result), use_depth=use_depth)
    return json.dumps(result)


def optical_frames(repo: Path, *, arm: str, stream: str, config: Path | None = None,
                   urdf: Path | None = None) -> dict[str, str]:
    """Bind each installed wrist view to its configured fixed optical frame.

    A camera on a different arm cannot be registered with this arm's joint
    readings. The original bracket supplies nominal CAD geometry; observed
    agreement, joint timing and robot-world registration remain separate.
    """
    import xml.etree.ElementTree as ET

    configured = arm_registry.load(repo)
    if arm not in configured or stream not in ('color', 'depth'):
        raise ValueError('wrist optical frame requires an explicit arm and color/depth stream')
    selected = [c for c in registry(repo, config) if c.get('arm') == arm]
    prefix = configured[arm].urdf_prefix
    key = 'optical_frame' if stream == 'color' else 'depth_optical_frame'
    if len(selected) > 1 and any(key not in c for c in selected):
        raise ValueError(f'{arm}: multiple cameras require explicit optical frames')
    tree = ET.parse(urdf or repo / 'urdf/tatbot.urdf').getroot()
    parents = {j.find('child').get('link'): j for j in tree.findall('joint')}
    result = {}
    for camera in selected:
        frame = camera.get(key, f'{prefix}/realsense_{stream}_optical_frame')
        if not isinstance(frame, str) or not frame.startswith(f'{prefix}/'):
            raise ValueError(f'{camera["role"]}: optical frame belongs to another arm')
        cursor, seen = frame, set()
        while cursor != f'{prefix}/link_6':
            if cursor in seen or cursor not in parents or parents[cursor].get('type') != 'fixed':
                raise ValueError(f'{frame}: stock camera must have a fixed chain to its own arm link_6')
            seen.add(cursor)
            cursor = parents[cursor].find('parent').get('link')
        result[camera['role']] = frame
    if len(set(result.values())) != len(result):
        raise ValueError(f'{arm}: wrist cameras require distinct optical frames')
    return result


# Existing callers use this as a view of this checkout's registry. Operations
# that load another fixture/installation use capture_roles(repo) directly.
CAPTURE_ROLES = capture_roles(REPO)
