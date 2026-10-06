"""The simulator's palette: the installed palette's URDF, without dip authority.

One path for every checkout. ``config/palette_geometry.json`` names the palette
(revision, URDF, body frame and a synthetic scene pose), ``config/palette.yaml``
names its slots and cap sizes, and the fiducial inventory names its tag. The
URDF's ``inkcap_*`` frames are cap SUPPORT FLOORS, so each rim is its floor plus
the cap's outside height -- the arithmetic ``ink_spec.palette_rim_layout`` gives
hardware -- unless ``rim_z_m`` records a measured rim. The ``palette_tag`` frame
places the sticker, its in-plane turn included.

The scene pose is ``simulation_root_*``: a placement for rendering and planning,
never the rack's measured pose, so the provenance stays visible to callers.
"""

from __future__ import annotations

import json
import math
import posixpath
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import ink_spec

GEOMETRY_RELPATH = 'config/palette_geometry.json'
GEOMETRY_SCHEMA = 2
FIDUCIAL_INVENTORY_RELPATH = 'config/fiducials.json'
EXAMPLE_FIDUCIAL_INVENTORY_RELPATH = 'config/examples/fiducials.json'
TAG_MESH_DIR = 'urdf/meshes/tags'
TAG_FRAME = 'palette_tag'
CAP_PREFIX = 'inkcap_'


@dataclass(frozen=True)
class PaletteScene:
    revision: str
    urdf: str
    mesh: str
    collision_mesh: str | None
    mesh_scale: tuple[float, float, float]
    tag_mesh: str
    tag_inventory: str
    """The fiducial inventory whose palette target names ``tag_mesh``."""
    tag_xyz_m: tuple[float, float, float]
    tag_rpy: tuple[float, float, float]
    """URDF fixed-axis roll, pitch, yaw of the tag in the palette body frame."""
    slots: tuple[tuple[str, ink_spec.PaletteSlot], ...]
    rim_layout_m: tuple[tuple[str, tuple[float, float, float]], ...]
    rim_basis: str

    @property
    def palette(self) -> dict[str, ink_spec.PaletteSlot]:
        return dict(self.slots)

    @property
    def rims(self) -> dict[str, tuple[float, float, float]]:
        return dict(self.rim_layout_m)


def _floats(text: str | None, count: int, what: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in (text or '').split())
    except ValueError:
        values = ()
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(f'simulation palette {what} is not {count} finite numbers')
    return values


def _fixed_origin(root: ET.Element, frame: str, child: str):
    """(xyz, rpy) of the fixed joint that hangs ``child`` directly off ``frame``."""
    joints = [joint for joint in root.findall('joint')
              if (joint.find('child') is not None
                  and joint.find('child').get('link') == child)]
    if len(joints) != 1 or joints[0].get('type') != 'fixed':
        raise ValueError(f'simulation palette has no single fixed joint for {child}')
    parent = joints[0].find('parent')
    if parent is None or parent.get('link') != frame:
        raise ValueError(f'simulation palette {child} does not hang off {frame}')
    origin = joints[0].find('origin')
    xyz = _floats(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', 3, f'{child} xyz')
    rpy = _floats(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', 3, f'{child} rpy')
    return xyz, rpy


def _body_meshes(repo: Path, urdf: str, root: ET.Element, frame: str):
    """Repo-relative visual and collision meshes of the body link, and their scale."""
    found = {}
    for role in ('visual', 'collision'):
        mesh = root.find(f"link[@name='{frame}']/{role}/geometry/mesh")
        if mesh is None:
            continue
        path = posixpath.normpath(posixpath.join(posixpath.dirname(urdf), mesh.get('filename', '')))
        if path.startswith(('/', '../')) or not (repo / path).is_file():
            raise ValueError(f'simulation palette {role} mesh {mesh.get("filename")!r} is missing')
        found[role] = (path, _floats(mesh.get('scale', '1 1 1'), 3, f'{role} mesh scale'))
    if 'visual' not in found:
        raise ValueError(f'simulation palette {frame} has no visual mesh')
    mesh, scale = found['visual']
    collision = found.get('collision')
    if collision is not None and collision[1] != scale:
        raise ValueError('simulation palette collision mesh scale differs from its visual mesh')
    return mesh, collision[0] if collision else None, scale


def fiducial_inventory_path(repo: Path) -> Path:
    """The live inventory, else the public example -- as tatbot_sim.urdf resolves it."""
    live = Path(repo) / FIDUCIAL_INVENTORY_RELPATH
    return live if live.is_file() else Path(repo) / EXAMPLE_FIDUCIAL_INVENTORY_RELPATH


def _tag_mesh(repo: Path) -> tuple[str, str]:
    """(mesh, inventory): the palette target's rendered tag, named by its identity."""
    inventory_path = fiducial_inventory_path(repo)
    inventory = json.loads(inventory_path.read_text())
    target = (inventory.get('targets') or {}).get('palette')
    if target is None:
        raise ValueError(f'{inventory_path.name} has no palette target')
    family = target.get('family', inventory.get('family') if inventory.get('schema_version') == 1 else None)
    ids = target.get('ids') or []
    edge_mm = float(target.get('edge_m', 0)) * 1000
    if not str(family).startswith('apriltag_') or len(ids) != 1 or abs(edge_mm - round(edge_mm)) > 1e-6:
        raise ValueError('simulation palette needs one inventory palette tag with an integer-mm edge')
    name = f"{family.removeprefix('apriltag_')}_{int(ids[0]):03d}_{round(edge_mm)}mm"
    mesh = f'{TAG_MESH_DIR}/{name}/tag.glb'
    if not (repo / mesh).is_file():
        raise ValueError(f'simulation palette tag mesh {mesh} is missing for the inventory palette target')
    return mesh, str(inventory_path.relative_to(repo))


def _rims(root: ET.Element, frame: str, slots: dict, geometry: dict):
    """Rim centres in the body frame, in palette.yaml order, and their basis."""
    caps = {joint.find('child').get('link') for joint in root.findall('joint')
            if joint.find('child') is not None
            and joint.find('child').get('link', '').startswith(CAP_PREFIX)}
    if caps != set(slots):
        raise ValueError(f'simulation palette caps {sorted(caps)} differ from palette.yaml '
                         f'slots {sorted(slots)}')
    measured = geometry.get('rim_z_m') or {}
    rims = {}
    for name, slot in slots.items():
        (x, y, floor), _rpy = _fixed_origin(root, frame, name)
        z = float(measured.get(name, floor + slot.size.height_m))
        # the same bound ink_spec.palette_rim_layout holds a rim to
        if not math.isfinite(z) or not floor < z < floor + 0.05:
            raise ValueError(f'simulation palette {name} has an invalid rim height {z}')
        rims[name] = (x, y, z)
    measured_caps = set(measured) & set(slots)
    if not measured_caps:
        basis = 'cad-estimate-unmeasured-rims'
    elif measured_caps == set(slots):
        basis = 'measured-rims'
    else:
        basis = 'partly-measured-rims'
    return rims, basis


def load(repo: Path) -> PaletteScene:
    """The palette palette_geometry.json names, as the simulator places it."""
    repo = Path(repo)
    geometry = json.loads((repo / GEOMETRY_RELPATH).read_text())
    if geometry.get('schema_version') != GEOMETRY_SCHEMA:
        raise ValueError(f'{GEOMETRY_RELPATH} schema {geometry.get("schema_version")!r} '
                         f'is not {GEOMETRY_SCHEMA}')
    urdf, frame = str(geometry['urdf']), str(geometry['frame'])
    root = ET.parse(repo / urdf).getroot()
    if root.find(f"link[@name='{frame}']") is None:
        raise ValueError(f'{urdf} has no {frame} link')
    mesh, collision_mesh, scale = _body_meshes(repo, urdf, root, frame)
    tag_mesh, tag_inventory = _tag_mesh(repo)
    tag_xyz, tag_rpy = _fixed_origin(root, frame, TAG_FRAME)
    slots = ink_spec.load_palette(repo)
    rims, basis = _rims(root, frame, slots, geometry)
    return PaletteScene(str(geometry['revision']), urdf, mesh, collision_mesh, scale,
                        tag_mesh, tag_inventory, tag_xyz, tag_rpy,
                        tuple(slots.items()), tuple(rims.items()), basis)


def base_transform(repo: Path, scene: PaletteScene, arm: str = 'right'):
    """(source, 4x4 arm base <- palette body) at the synthetic scene pose.

    Scene pose only: an accepted vision registration grants no dip authority
    in simulation, and a simulator never claims to be where the rack is.
    """
    import numpy as np
    from transforms3d.quaternions import quat2mat

    geometry = json.loads((Path(repo) / GEOMETRY_RELPATH).read_text())
    if geometry.get('revision') != scene.revision:
        raise ValueError(f'{GEOMETRY_RELPATH} changed revision after the scene was loaded')
    point = np.asarray(geometry['simulation_root_xyz_m'], dtype=float)
    quaternion = np.asarray(geometry['simulation_root_quaternion_wxyz'], dtype=float)
    if (point.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(point).all() or not np.isfinite(quaternion).all()
            or abs(float(quaternion @ quaternion) - 1.0) > 1e-8):
        raise ValueError('simulation palette requires a finite unit scene pose')
    root_from_palette = np.eye(4)
    root_from_palette[:3, :3] = quat2mat(quaternion)
    root_from_palette[:3, 3] = point
    return 'synthetic-installed-cad', ink_spec.base_from_root_matrix(repo, arm) @ root_from_palette
