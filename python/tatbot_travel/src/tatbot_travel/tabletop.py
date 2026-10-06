"""Seeded table mats and paper, independent of which cameras are enabled."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from tatbot_travel import textures
from tatbot_travel.appearance import SurfaceConfig, sample_surface
from tatbot_travel.scene import Material, SceneBuilder


@dataclass(frozen=True)
class PaperPlacementConfig:
    x_m: tuple[float, float] | None = None  # inherit the tabletop range when omitted
    y_m: tuple[float, float] | None = None
    width_m: tuple[float, float] = (0.14, 0.24)
    height_m: tuple[float, float] = (0.20, 0.32)
    yaw_deg: tuple[float, float] = (-180, 180)
    layouts: dict[str, float] = field(default_factory=lambda: {"scatter": 0.55, "cluster": 0.30, "stack": 0.15})
    scatter_separation_m: float = 0.14
    cluster_spread_m: float = 0.09
    stack_spread_m: float = 0.012
    stack_yaw_std_deg: float = 6

    def __post_init__(self):
        for name in ("x_m", "y_m", "width_m", "height_m", "yaw_deg"):
            bounds = getattr(self, name)
            if bounds is not None and (len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]):
                raise ValueError(f"invalid paper placement range: {name}")
        if min(self.width_m[0], self.height_m[0]) <= 0:
            raise ValueError("paper dimensions must be positive")
        spread = [self.scatter_separation_m, self.cluster_spread_m, self.stack_spread_m, self.stack_yaw_std_deg]
        if not np.isfinite(spread).all() or min(spread) < 0:
            raise ValueError("paper placement spreads must be finite and non-negative")
        weights = np.asarray(list(self.layouts.values()), dtype=float)
        if (not self.layouts or set(self.layouts) - {"scatter", "cluster", "stack"}
                or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError("invalid paper layout weights")


@dataclass(frozen=True)
class TabletopConfig:
    mat_count: tuple[int, int] = (0, 2)
    paper_count: tuple[int, int] | None = None  # default: 0..WorldConfig.max_sheets
    x_m: tuple[float, float] = (0.12, 0.65)
    y_m: tuple[float, float] = (-0.35, 0.35)
    mat_surface: SurfaceConfig = field(default_factory=lambda: SurfaceConfig(
        styles={"grid": 0.4, "rubber": 0.35, "cloth": 0.25}, value=(0.15, 0.65), neutral_prob=0.2))
    paper_styles: dict[str, float] = field(default_factory=lambda: {
        "plain": 0.3, "lined": 0.25, "grid": 0.2, "printed": 0.25})
    paper_placement: PaperPlacementConfig = field(default_factory=PaperPlacementConfig)

    def __post_init__(self):
        for count in (self.mat_count, self.paper_count):
            if count is not None and (len(count) != 2 or any(int(x) != x for x in count)
                                      or not 0 <= count[0] <= count[1] <= 8):
                raise ValueError("tabletop counts must be ordered integer ranges in [0, 8]")
        for bounds in (self.x_m, self.y_m):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                raise ValueError("invalid tabletop position range")
        weights = np.asarray(list(self.paper_styles.values()), dtype=float)
        if (not self.paper_styles or set(self.paper_styles) - {"plain", "lined", "grid", "printed"}
                or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError("invalid paper style weights")


@dataclass(frozen=True)
class TabletopItem:
    name: str
    kind: str
    xy: tuple[float, float]
    half: tuple[float, float]
    yaw: float
    bottom: float
    thickness: float

    @property
    def top(self) -> float:
        return self.bottom + self.thickness

    def contains(self, xy: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        local = (np.asarray(xy) - self.xy) @ np.array([[c, -s], [s, c]])
        return np.all(np.abs(local) <= np.asarray(self.half) + 1e-6, axis=-1)

    def overlaps(self, other: TabletopItem) -> bool:
        """Rectangle intersection by separating axes, including edge-only overlap."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        own = np.array([[c, s], [-s, c]])
        c, s = np.cos(other.yaw), np.sin(other.yaw)
        theirs = np.array([[c, s], [-s, c]])
        axes = np.concatenate([own, theirs])
        radii = np.abs(axes @ own.T) @ self.half + np.abs(axes @ theirs.T) @ other.half
        return bool(np.all(np.abs(axes @ (np.asarray(other.xy) - self.xy)) <= radii + 1e-9))

    def xml(self) -> str:
        return (f'<geom name="{self.name}" type="mesh" mesh="{self.name}" '
                f'pos="{self.xy[0]} {self.xy[1]} {self.bottom + self.thickness / 2}" euler="0 0 {self.yaw}" '
                f'material="{self.name}" contype="0" conaffinity="0"/>')

    def mesh_obj(self) -> str:
        """A thin solid with explicit top-face UVs, so one sheet/mat gets one texture."""
        x, y = self.half
        z = self.thickness / 2
        vertices = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
                    (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
        header = "".join(f"v {a} {b} {c}\n" for a, b, c in vertices)
        header += "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        faces = [(5, 6, 7), (5, 7, 8), (1, 3, 2), (1, 4, 3),
                 (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
                 (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]
        return header + "".join("f " + " ".join(f"{v}/{(v - 1) % 4 + 1}" for v in face) + "\n" for face in faces)


def _surface_minimum(points: np.ndarray, faces: np.ndarray, half: np.ndarray) -> float | None:
    """Lowest skin point over a rectangle, including triangle/edge intersections."""
    triangles = points[faces]
    overlaps = np.all((triangles[..., :2].min(axis=1) <= half)
                      & (triangles[..., :2].max(axis=1) >= -half), axis=1)
    triangles = triangles[overlaps]
    if not len(triangles):
        return None
    minima = []
    inside = np.all(np.abs(triangles[..., :2]) <= half + 1e-9, axis=-1)
    if inside.any():
        minima.append(float(triangles[..., 2][inside].min()))
    a, b = triangles.reshape(-1, 3), np.roll(triangles, -1, axis=1).reshape(-1, 3)
    delta = b - a
    for axis, boundary in ((0, -half[0]), (0, half[0]), (1, -half[1]), (1, half[1])):
        t = np.divide(boundary - a[:, axis], delta[:, axis], out=np.full(len(a), np.nan),
                      where=np.abs(delta[:, axis]) > 1e-12)
        crossing = a + t[:, None] * delta
        valid = (t >= 0) & (t <= 1) & (np.abs(crossing[:, 1 - axis]) <= half[1 - axis] + 1e-9)
        if valid.any():
            minima.append(float(crossing[valid, 2].min()))
    origin = triangles[:, 0]
    u, v = triangles[:, 1] - origin, triangles[:, 2] - origin
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    for corner in ((-half[0], -half[1]), (-half[0], half[1]), (half[0], -half[1]), (half[0], half[1])):
        offset = np.asarray(corner) - origin[:, :2]
        alpha = np.divide(offset[:, 0] * v[:, 1] - offset[:, 1] * v[:, 0], det,
                          out=np.full(len(det), np.nan), where=np.abs(det) > 1e-12)
        beta = np.divide(u[:, 0] * offset[:, 1] - u[:, 1] * offset[:, 0], det,
                         out=np.full(len(det), np.nan), where=np.abs(det) > 1e-12)
        inside = (alpha >= -1e-9) & (beta >= -1e-9) & (alpha + beta <= 1 + 1e-9)
        if inside.any():
            z = origin[:, 2] + alpha * u[:, 2] + beta * v[:, 2]
            minima.append(float(z[inside].min()))
    return min(minima) if minima else None


def support_height(vertices: np.ndarray, faces: np.ndarray, xy: np.ndarray, items: list[TabletopItem]) -> float:
    """Translate the actual skin triangles onto the table or a mat/paper beneath them."""
    height = -float(vertices[:, 2].min())
    for item in items:
        points = vertices.copy()
        c, s = np.cos(item.yaw), np.sin(item.yaw)
        points[:, :2] = (points[:, :2] + xy - item.xy) @ np.array([[c, -s], [s, c]])
        lowest = _surface_minimum(points, faces, np.asarray(item.half))
        if lowest is not None:
            height = max(height, item.top - lowest)
    return height


def _paper_pose(rng: np.random.Generator, cfg: TabletopConfig, layout: str,
                previous: list[TabletopItem]) -> tuple[tuple[float, float], float]:
    paper = cfg.paper_placement
    lo = np.array([(paper.x_m or cfg.x_m)[0], (paper.y_m or cfg.y_m)[0]])
    hi = np.array([(paper.x_m or cfg.x_m)[1], (paper.y_m or cfg.y_m)[1]])
    xy, yaw = rng.uniform(lo, hi), np.radians(rng.uniform(*paper.yaw_deg))
    sheets = [item for item in previous if item.kind == "paper"]
    if sheets and layout in ("cluster", "stack"):
        spread = paper.stack_spread_m if layout == "stack" else paper.cluster_spread_m
        xy = np.clip(rng.normal(sheets[0].xy, spread), lo, hi)
        if layout == "stack":
            yaw = sheets[0].yaw + np.radians(rng.normal(0, paper.stack_yaw_std_deg))
    if sheets and layout == "scatter":
        centers = np.asarray([item.xy for item in sheets])
        candidates = np.vstack([xy, rng.uniform(lo, hi, (24, 2))])
        distance = np.linalg.norm(candidates[:, None] - centers, axis=-1).min(axis=1)
        valid = np.flatnonzero(distance >= paper.scatter_separation_m)
        xy = candidates[valid[0] if len(valid) else distance.argmax()]
    return tuple(float(v) for v in xy), float((yaw + np.pi) % (2 * np.pi) - np.pi)


def _sample_item(rng: np.random.Generator, cfg: TabletopConfig, kind: str, index: int,
                 previous: list[TabletopItem], paper_layout: str) -> TabletopItem:
    if kind == "mat":
        xy = (float(rng.uniform(*cfg.x_m)), float(rng.uniform(*cfg.y_m)))
        yaw = float(rng.uniform(-np.pi, np.pi))
        half = (rng.uniform(0.14, 0.25), rng.uniform(0.10, 0.20))
    else:
        xy, yaw = _paper_pose(rng, cfg, paper_layout, previous)
        paper = cfg.paper_placement
        half = (rng.uniform(*paper.width_m) / 2, rng.uniform(*paper.height_m) / 2)
    thickness = float(rng.uniform(0.001, 0.003)) if kind == "mat" else 0.0002
    footprint = TabletopItem("footprint", kind, xy, tuple(half), yaw, 0, thickness)
    bottom = max([0.0, *(item.top for item in previous if footprint.overlaps(item))])
    return TabletopItem(f"tabletop_{kind}{index}", kind, xy, tuple(half), yaw, bottom, thickness)


def add_tabletop(sb: SceneBuilder, geometry_rng: np.random.Generator, appearance_rng: np.random.Generator,
                 cfg: TabletopConfig, max_sheets: int) -> tuple[list[TabletopItem], list[dict]]:
    items, metadata = [], []
    layouts = list(cfg.paper_placement.layouts)
    weights = np.asarray(list(cfg.paper_placement.layouts.values()))
    paper_layout = str(geometry_rng.choice(layouts, p=weights / weights.sum()))
    paper_count = cfg.paper_count or (0, max_sheets)
    counts = (("mat", cfg.mat_count), ("paper", paper_count))
    for kind, bounds in counts:
        for index in range(int(geometry_rng.integers(bounds[0], bounds[1] + 1))):
            item = _sample_item(geometry_rng, cfg, kind, index, items, paper_layout)
            if kind == "mat":
                look = sample_surface(appearance_rng, cfg.mat_surface)
                rgb = textures.table_texture(appearance_rng, kind=look["style"], colour=np.asarray(look["rgb"]) * 255)
            else:
                styles = list(cfg.paper_styles)
                weights = np.asarray(list(cfg.paper_styles.values()))
                look = {"style": str(appearance_rng.choice(styles, p=weights / weights.sum())),
                        "specular": 0.02, "shininess": 0.1, "layout": paper_layout}
                rgb = textures.paper_texture(appearance_rng, look["style"])
            sb.add_texture(item.name, rgb)
            sb.add_mesh_obj(item.name, item.mesh_obj())
            sb.materials[item.name] = Material((1, 1, 1, 1), look["specular"], look["shininess"],
                                               texture=item.name)
            sb.world_xml.append(item.xml())
            items.append(item)
            metadata.append({**asdict(item), **look})
    return items, metadata
