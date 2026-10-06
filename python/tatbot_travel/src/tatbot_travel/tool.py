"""The laser pen as the datasheet states it.

The generated URDF block already places the pen body and its measured TCP
(``<arm>/tattoo_needle``: lens face plus the datasheet's standoff) in
``<arm>/tattoo_pen``. What the block does not carry is which profile station
is the lens face and which segments are chrome, so both come from the
datasheet here rather than from a copied number.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import yaml

from tatbot_travel import assets

LASER_PEN = "picosecond-laser-pen"


@dataclass(frozen=True)
class PenGeometry:
    profile: np.ndarray  # (N, 2): z along the pen axis, radius; z = 0 at the clamp datum
    segment_rgba: tuple[tuple[float, float, float, float], ...]
    tcp_z: float  # datasheet TCP: lens face plus its stated standoff

    @property
    def lens_z(self) -> float:
        """The lens face: the profile's last station."""
        return float(self.profile[-1, 0])

    def is_chrome(self, segment: int) -> bool:
        """Chrome segments are the dark metallic ones; the shell is near-white."""
        return max(self.segment_rgba[segment][:3]) < 0.8


def _obj(vertices: np.ndarray, normals: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> str:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in normals]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uvs]
    lines += ["f " + " ".join(f"{i + 1}/{i + 1}/{i + 1}" for i in tri) for tri in faces]
    return "\n".join(lines) + "\n"


def lathe_obj(stations: np.ndarray, radii: np.ndarray, n_theta: int = 48) -> str:
    """A surface of revolution about +z as OBJ text, with a u-around / v-along texture map."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    dz = np.gradient(stations)
    dr = np.gradient(radii)
    slope = np.arctan2(-dr, dz)  # outward normal tilts back where the radius grows
    vertices, normals, uvs = [], [], []
    span = max(stations[-1] - stations[0], 1e-9)
    for z, r, s in zip(stations, radii, slope, strict=True):
        for t in theta:
            vertices.append((r * np.cos(t), r * np.sin(t), z))
            normals.append((np.cos(s) * np.cos(t), np.cos(s) * np.sin(t), np.sin(s)))
            uvs.append((t / (2.0 * np.pi), (z - stations[0]) / span))
    faces = []
    ring = n_theta + 1
    for i in range(len(stations) - 1):
        for j in range(n_theta):
            a, b = i * ring + j, i * ring + j + 1
            c, d = a + ring, b + ring
            faces += [(a, b, d), (a, d, c)]
    return _obj(np.array(vertices), np.array(normals), np.array(uvs), np.array(faces))


def disk_obj(z: float, radius: float, n_theta: int = 48) -> str:
    """A disk facing +z at station ``z`` (the lens face)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rim = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z)], axis=1)
    vertices = np.vstack([[0.0, 0.0, z], rim])
    normals = np.tile([0.0, 0.0, 1.0], (len(vertices), 1))
    uvs = 0.5 + 0.5 * np.vstack([[0.0, 0.0], rim[:, :2] / radius])
    faces = np.array([(0, 1 + j, 1 + (j + 1) % n_theta) for j in range(n_theta)])
    return _obj(vertices, normals, uvs, faces)


def pen_meshes(pen: PenGeometry, samples_per_segment: int = 6) -> dict[str, str]:
    """OBJ text per appearance class: ``shell``, ``chrome`` and the ``lens`` face."""
    parts: dict[str, list[str]] = {"shell": [], "chrome": []}
    last = len(pen.segment_rgba) - 1
    for k in range(last + 1):
        z0, r0 = pen.profile[k]
        z1, r1 = pen.profile[k + 1]
        stations = np.linspace(z0, z1, samples_per_segment)
        radii = np.linspace(r0, r1, samples_per_segment)
        cls = "chrome" if (k == last or pen.is_chrome(k)) else "shell"
        parts[cls].append(lathe_obj(stations, radii))
    meshes = {f"pen_{cls}_{i}": text for cls, texts in parts.items() for i, text in enumerate(texts)}
    meshes["pen_lens_0"] = disk_obj(pen.lens_z, float(pen.profile[-1, 1]))
    return meshes


@lru_cache(maxsize=4)
def load_pen(tool_id: str = LASER_PEN) -> PenGeometry:
    spec = yaml.safe_load(assets.tool_datasheet(tool_id).read_text())
    profile = np.asarray(spec["profile"], dtype=float)
    colors = tuple(tuple(float(c) for c in s.split()) for s in spec["segment_colors"])
    if len(colors) != len(profile) - 1:
        raise ValueError(f"{tool_id}: one segment colour per profile segment expected")
    return PenGeometry(profile=profile, segment_rgba=colors, tcp_z=float(spec["tcp_z_m"]))
