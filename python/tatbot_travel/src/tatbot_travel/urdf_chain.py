"""One arm of the rig URDF, lifted into MJCF.

``urdf/tatbot.urdf`` states both arms, their tools and their wrist cameras.
The travel demo carries one arm, so this module takes the subtree below
``<arm>/base_link`` and emits MJCF bodies for it: revolute joints become
hinges, the carriage and its mimic are frozen at a chosen value, and fixed
joints become nested bodies, so frames such as ``left/tattoo_needle`` and
``left/realsense_color_optical_frame`` keep their URDF names and poses.
Only visual geometry is kept -- the generator measures clearances itself.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARM_JOINTS = tuple(f"joint_{i}" for i in range(6))


@dataclass(frozen=True)
class Visual:
    link: str
    index: int
    kind: str  # mesh | cylinder | sphere | box
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    params: dict


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str  # revolute | continuous | prismatic | fixed
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None


@dataclass
class Chain:
    arm: str
    base: str
    joints: dict[str, Joint]  # by child link
    children: dict[str, list[Joint]]  # by parent link
    visuals: dict[str, list[Visual]]
    urdf_dir: Path

    def hinge_names(self) -> list[str]:
        return [f"{self.arm}/{j}" for j in ARM_JOINTS]

    def limits(self) -> np.ndarray:
        """(6, 2) lower/upper bounds of the arm joints, in hinge order."""
        by_name = {j.name: j for j in self.joints.values()}
        return np.array([[by_name[n].lower, by_name[n].upper] for n in self.hinge_names()], dtype=float)


def _floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split()) if text else default


def rpy_to_quat(rpy) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw (R = Rz Ry Rx) as a MuJoCo (w, x, y, z) quaternion."""
    r, p, y = (0.5 * float(a) for a in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = (float(a) for a in rpy)
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _parse_joint(el: ET.Element) -> Joint:
    origin, limit = el.find("origin"), el.find("limit")
    axis = el.find("axis")
    return Joint(
        name=el.get("name"),
        kind=el.get("type"),
        parent=el.find("parent").get("link"),
        child=el.find("child").get("link"),
        xyz=_floats(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
        rpy=_floats(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
        axis=_floats(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
        lower=float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
        upper=float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
    )


def _parse_visual(link: str, index: int, el: ET.Element) -> Visual | None:
    origin, geometry = el.find("origin"), el.find("geometry")
    shape = geometry[0]
    params: dict = {}
    if shape.tag == "mesh":
        params = {"file": shape.get("filename"), "scale": _floats(shape.get("scale"), (1.0, 1.0, 1.0))}
    elif shape.tag == "cylinder":
        params = {"radius": float(shape.get("radius")), "length": float(shape.get("length"))}
    elif shape.tag == "sphere":
        params = {"radius": float(shape.get("radius"))}
    elif shape.tag == "box":
        params = {"size": _floats(shape.get("size"), (0.0, 0.0, 0.0))}
    else:
        return None
    return Visual(
        link=link,
        index=index,
        kind=shape.tag,
        xyz=_floats(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
        rpy=_floats(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
        params=params,
    )


def load_chain(urdf: Path, arm: str = "left") -> Chain:
    """The subtree of ``<arm>/base_link``: every joint and visual below it."""
    root = ET.parse(urdf).getroot()
    joints = [_parse_joint(el) for el in root.iter("joint")]
    children: dict[str, list[Joint]] = {}
    for joint in joints:
        children.setdefault(joint.parent, []).append(joint)
    visuals: dict[str, list[Visual]] = {}
    for link in root.iter("link"):
        parsed = [_parse_visual(link.get("name"), i, v) for i, v in enumerate(link.findall("visual"))]
        visuals[link.get("name")] = [v for v in parsed if v is not None]
    base = f"{arm}/base_link"
    keep: dict[str, Joint] = {}
    frontier = [base]
    while frontier:
        link = frontier.pop()
        for joint in children.get(link, []):
            keep[joint.child] = joint
            frontier.append(joint.child)
    if not keep:
        raise ValueError(f"{urdf} has no subtree below {base}")
    sub_children = {p: [j for j in js if j.child in keep] for p, js in children.items()}
    sub_visuals = {name: visuals.get(name, []) for name in [base, *keep]}
    return Chain(arm=arm, base=base, joints=keep, children=sub_children, visuals=sub_visuals,
                 urdf_dir=urdf.parent)


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


@dataclass
class MjcfParts:
    """MJCF text for the arm plus the mesh files it references."""

    body: str
    meshes: str
    mesh_files: dict[str, Path]


class _Emitter:
    def __init__(self, chain: Chain, carriage: float,
                 material_for: Callable[[Visual], str | tuple[str, int] | None],
                 extra: dict[str, str], skip_mesh: Callable[[str], bool]):
        self.chain, self.carriage, self.material_for = chain, carriage, material_for
        self.extra, self.skip_mesh = extra, skip_mesh
        self.mesh_files: dict[str, Path] = {}
        self.mesh_xml: list[str] = []

    def mesh_name(self, visual: Visual) -> str:
        file = self.chain.urdf_dir / visual.params["file"]
        scale = visual.params["scale"]
        name = f"{file.parent.name}__{file.stem}__{_fmt(scale).replace(' ', '_')}"
        if name not in self.mesh_files:
            self.mesh_files[name] = file
            self.mesh_xml.append(f'<mesh name="{name}" file="{name}.stl" scale="{_fmt(scale)}"/>')
        return name

    def geom(self, visual: Visual) -> str:
        material = self.material_for(visual)
        if material is None:
            return ""
        material, group = material if isinstance(material, tuple) else (material, 1)
        pose = f'pos="{_fmt(visual.xyz)}" quat="{_fmt(rpy_to_quat(visual.rpy))}"'
        common = f'{pose} material="{material}" contype="0" conaffinity="0" group="{group}"'
        p = visual.params
        if visual.kind == "mesh":
            if self.skip_mesh(p["file"]):
                return ""
            return f'<geom type="mesh" mesh="{self.mesh_name(visual)}" {common}/>'
        if visual.kind == "cylinder":
            return f'<geom type="cylinder" size="{p["radius"]:.9g} {0.5 * p["length"]:.9g}" {common}/>'
        if visual.kind == "sphere":
            return f'<geom type="sphere" size="{p["radius"]:.9g}" {common}/>'
        return f'<geom type="box" size="{_fmt(0.5 * np.asarray(p["size"]))}" {common}/>'

    def joint_xml(self, joint: Joint) -> tuple[str, str]:
        """(pose attributes, joint element) for the body this joint creates."""
        xyz = np.asarray(joint.xyz, dtype=float)
        rot = rpy_to_matrix(joint.rpy)
        if joint.kind == "prismatic":
            xyz = xyz + rot @ (np.asarray(joint.axis) * self.carriage)
        pose = f'pos="{_fmt(xyz)}" quat="{_fmt(rpy_to_quat(joint.rpy))}"'
        if joint.kind in ("revolute", "continuous"):
            limited = joint.lower is not None and joint.upper is not None
            rng = f' range="{joint.lower:.9g} {joint.upper:.9g}"' if limited else ""
            element = f'<joint name="{joint.name}" type="hinge" axis="{_fmt(joint.axis)}"{rng}/>'
            return pose, element
        return pose, ""

    def body(self, link: str, pose: str, joint_element: str, indent: str) -> str:
        lines = [f'{indent}<body name="{link}" {pose}>']
        if joint_element:
            lines.append(f"{indent}  {joint_element}")
        lines += [f"{indent}  {g}" for g in (self.geom(v) for v in self.chain.visuals.get(link, [])) if g]
        if link in self.extra:
            lines.append(f"{indent}  {self.extra[link]}")
        for joint in self.chain.children.get(link, []):
            child_pose, child_joint = self.joint_xml(joint)
            lines.append(self.body(joint.child, child_pose, child_joint, indent + "  "))
        lines.append(f"{indent}</body>")
        return "\n".join(lines)


def to_mjcf(chain: Chain, *, base_pos=(0.0, 0.0, 0.0), base_quat=(1.0, 0.0, 0.0, 0.0),
            carriage: float = 0.0,
            material_for: Callable[[Visual], str | tuple[str, int] | None] = lambda v: "default",
            extra: dict[str, str] | None = None,
            skip_mesh: Callable[[str], bool] = lambda f: not f.lower().endswith(".stl")) -> MjcfParts:
    """MJCF for the arm, its base placed at ``base_pos``/``base_quat`` in the parent frame.

    ``material_for`` names the MJCF material of each visual, or (material, geom group) to draw it only
    for renders that show that group (``None`` drops it);
    ``extra`` injects raw MJCF (cameras, sites) into the named bodies.
    """
    emitter = _Emitter(chain, carriage, material_for, extra or {}, skip_mesh)
    body = emitter.body(chain.base, f'pos="{_fmt(base_pos)}" quat="{_fmt(base_quat)}"', "", "    ")
    return MjcfParts(body=body, meshes="\n    ".join(emitter.mesh_xml), mesh_files=emitter.mesh_files)
