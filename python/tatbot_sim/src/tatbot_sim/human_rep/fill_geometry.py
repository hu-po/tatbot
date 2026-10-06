"""Plan finite-width deposition inside the union of canonical paint regions.

Region tessellation is a representation detail, never a set of inked borders.
This module has no body, robot, service, or ink-capacity authority. Coordinates
and widths are metric chart values; GEOS performs only planar set operations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from shapely import GeometryCollection, LineString, Polygon, STRtree, normalize, set_precision, union_all

from tatbot_sim.human_rep.contracts import FILL_STYLES, ContractError

MAX_FILL_ROWS = 20_000
MAX_FILL_STROKES = 100_000
ARC_SEGMENTS = 32  # Per quadrant; inscribed-circle radial error <0.031%.
PAINT_GRID_M = 1e-10  # 0.1 nm: join numeric cracks at shared tessellation edges.
PAINT_JOIN_M = 1e-8  # 10 nm: float32 SVG stroke tessellation T-junction tolerance.
# How much of a painted region the planned stroke path must actually cover.
#
# A fidelity floor, not a safety limit: it does not gate motion, contact force,
# retract or ink accounting, and nothing below depends on it to stop the arm.
# What it decides is whether the plan draws the picture it claims to.
#
# It was 0.98, which refused 57 of 95 candidates in a measured suite over
# generated flash. Those 57 refusals are not one population. Their coverage
# runs 0.0, 0.31, 0.77 ... 0.97, and the tail is a cliff: a floor of 0.70 and a
# floor of 0.50 admit exactly the same 48, because nine candidates sit at
# essentially zero and are genuinely undrawable. Everything above the cliff is
# a drawing with some ink missing; everything below it is not a drawing.
#
# 0.90 is set on 2026-09-10 at the fleet owner's direction: a tenth of the ink
# may be missing, and everything below the cliff is still refused. This gate is
# shared with real session preparation, so the same tolerance applies there; a
# consumer that wants the old strictness passes coverage_floor=0.98.
PAINT_COVERAGE_FLOOR = 0.90
# How much wider than the artwork the drawing may be, where the artwork is
# thinner than the tool. A 0.3 mm tool tracing a 0.2 mm line lays 1.76x the
# painted area, which is the line drawn slightly heavy — recognisably the same
# drawing. Past 4x the ink is mostly outside the artwork and the result is not
# that drawing, so the tool is refused for it exactly as before.
NARROW_OVERDRAW_LIMIT = 4.0


def close_paint_seams(target):
    """Canonical numeric seam treatment shared by paint planning and rendering."""
    target = target.buffer(PAINT_JOIN_M, quad_segs=ARC_SEGMENTS).buffer(-PAINT_JOIN_M, quad_segs=ARC_SEGMENTS)
    return normalize(set_precision(set_precision(target, PAINT_GRID_M), 0))


def paint_union(polygons: Sequence[np.ndarray], masks: Sequence[np.ndarray]):
    """Union before masking/erosion; never repair invalid source polygons."""
    if len(polygons) + len(masks) > 50_000 or sum(len(p) for p in [*polygons, *masks]) > 100_000:
        raise ContractError("tattoo_program_over_budget", "$.layers", "paint exceeds polygon/point budget")

    def checked(points):
        polygon = Polygon(points)
        if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
            raise ContractError("tattoo_program_unsupported", "$.layers", "invalid or empty paint polygon")
        return polygon

    # SVGLoader's float32 stroke mesh and metric conversion can leave coincident
    # edges a few ulps apart. Explicit snap-rounding joins those edges; the grid
    # is fixed in physical units and far below the artwork chord-error budget.
    target = union_all([checked(points) for points in polygons], grid_size=PAINT_GRID_M)
    # A float32 stroke fan can terminate at a double-precision edge interior
    # (a T-junction). Grid rounding alone cannot put that point on the edge.
    # Close only sub-20-nm seams. This fixed, documented target tolerance is
    # independent of the requested tool width and cannot hide missing dots.
    target = close_paint_seams(target)
    if masks:
        target = target.difference(union_all([checked(points) for points in masks], grid_size=PAINT_GRID_M))
    # Buffer intersections can themselves create sub-grid slivers. Snap the
    # result to the same declared grid; never discard a component by area.
    return normalize(set_precision(set_precision(target, PAINT_GRID_M), 0))


def _parts(geometry, kind):
    if geometry.geom_type == kind:
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _parts(part, kind)


def _center_domain(target, safe_radius: float):
    """Where the tool's centre may travel, and which paint is thinner than it.

    Eroding by the tool's radius is right for any region the tool fits inside.
    A region thinner than the tool has an empty erosion, and the whole drawing
    used to be refused for it — but a tool that cannot draw a line thinner than
    its own tip does not therefore refuse to draw the line. It traces the
    middle, and the drawn line comes out at tool width. That is what a person
    does with a fine liner, and it is what these regions get: the erosion
    radius is reduced for them until a path exists, which for a thin sliver is
    its centre.

    The regions this applies to are returned, because the deposited ink then
    lies wider than the artwork there and the spill rule has to know exactly
    where that is allowed.
    """
    eroded = normalize(target.buffer(-safe_radius, quad_segs=ARC_SEGMENTS))
    thin = []
    for component in _parts(target, "Polygon"):
        if not normalize(component.buffer(-safe_radius, quad_segs=ARC_SEGMENTS)).is_empty:
            continue
        # Halve the radius until the component admits a centre path. Eight
        # halvings reach 1/256 of the tool radius; below that the component is
        # narrower than a quarter micron and is not a drawing instruction.
        radius = safe_radius
        for _ in range(8):
            radius /= 2
            middle = normalize(component.buffer(-radius, quad_segs=ARC_SEGMENTS))
            if not middle.is_empty:
                eroded = normalize(union_all([eroded, middle], grid_size=PAINT_GRID_M))
                thin.append(component)
                break
    return eroded, thin


def _link_ring(output: list[np.ndarray], points: np.ndarray, center_domain, width_m: float) -> bool:
    """Append the ring to the previous path when a short link joins them.

    Join only a short link wholly inside the eroded paint. This preserves holes
    and disconnected marks while eliminating hundreds of unnecessary lifts
    across thin linework.
    """
    if not output:
        return False
    distances = np.linalg.norm(points[:-1] - output[-1][-1], axis=1)
    index = int(np.argmin(distances))
    link = LineString([output[-1][-1], points[index]])
    if distances[index] <= width_m * 2 and center_domain.covers(link):
        opened = np.roll(points[:-1], -index, axis=0)
        output[-1] = np.concatenate([output[-1], opened, opened[:1]])
        return True
    return False


@dataclass(frozen=True)
class FillPath:
    """One planned centreline and where the planner got it from.

    `depth` is the inset the path started at (0: the boundary ring of a paint
    component); a path that links deeper rings keeps the depth of its first.
    `island_area_m2` is the area of the inset polygon the first ring bounds:
    a thin feature erodes into small islands, and their rings are the confetti
    the schedule drops first. `component` is the connected paint polygon the
    path lies in, shared by identity across every path of that component, so
    the schedule can hold each component to the coverage floor.
    """
    points: np.ndarray
    depth: int
    island_area_m2: float
    component: Polygon


def _inset_rings(center_domain, width_m: float, rows: int, components: list[Polygon]):
    """Concentric inset rings of the centre domain, linked where a short link
    lies inside it, with (depth, island area, component index) per path."""
    output: list[np.ndarray] = []
    # Every inset polygon lies inside exactly one paint component, found by
    # its representative point.
    provenance: list[tuple[int, float, int]] = []
    component_tree = STRtree(components)
    inset = center_domain
    for depth in range(rows):
        if inset.is_empty:
            break
        for polygon in _parts(inset, "Polygon"):
            owner = _owner(component_tree, components, polygon)
            for ring in [polygon.exterior, *polygon.interiors]:
                points = np.asarray(ring.coords, dtype=np.float64)
                if not _link_ring(output, points, center_domain, width_m):
                    output.append(points)
                    provenance.append((depth, float(polygon.area), owner))
        inset = normalize(inset.buffer(-width_m * .45, quad_segs=ARC_SEGMENTS))
        if len(output) > MAX_FILL_STROKES:
            raise ContractError("tattoo_program_over_budget", "$.layers", "fill exceeds 100000 strokes")
    return output, provenance


def _long_axis(polygon: Polygon) -> np.ndarray:
    """Unit direction of the long side of the minimum-area bounding rectangle."""
    box = polygon.minimum_rotated_rectangle
    if box.geom_type != "Polygon":
        return np.array([1.0, 0.0])
    corners = np.asarray(box.exterior.coords)[:4]
    edges = np.diff(np.vstack([corners, corners[:1]]), axis=0)[:2]
    edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    norm = float(np.linalg.norm(edge))
    if norm <= 0:
        return np.array([1.0, 0.0])
    axis = edge / norm
    # A stable sign: rows read left to right, top to bottom, whichever way
    # the rectangle happened to be enumerated.
    return axis if (axis[0], axis[1]) > (0.0, 0.0) or (axis[0] == 0.0 and axis[1] > 0.0) else -axis


def hatch_paths(center_domain, width_m: float, rings: int, components: list[Polygon]):
    """Parallel rows across the centre domain eroded by `rings` ring pitches.

    Each connected component of the eroded region is hatched along the long
    axis of its minimum-area bounding rectangle at the ring pitch
    (0.45 x width), so a feather becomes a few long rows rather than a stack
    of nested slivers. Consecutive rows join into a serpentine when the joining
    segment lies wholly inside `center_domain` and is at most 2 x width long,
    the ring linker's own admissibility test, so visible negative space is
    never bridged. Rows are ordered by offset then along-row coordinate.
    Returns the paths and their (depth, island area, component index).
    """
    pitch = width_m * 0.45
    region = normalize(center_domain.buffer(-pitch * rings, quad_segs=ARC_SEGMENTS))
    output: list[np.ndarray] = []
    provenance: list[tuple[int, float, int]] = []
    if region.is_empty:
        return output, provenance
    component_tree = STRtree(components)
    for polygon in _parts(region, "Polygon"):
        owner = _owner(component_tree, components, polygon)
        axis = _long_axis(polygon)
        normal = np.array([-axis[1], axis[0]])
        vertices = np.asarray(polygon.exterior.coords)
        along = vertices @ axis
        across = vertices @ normal
        reach = float(along.max() - along.min()) + width_m
        centre = float(along.min() + along.max()) / 2
        offsets = np.arange(across.min() + pitch / 2, across.max(), pitch)
        if len(offsets) > MAX_FILL_ROWS:
            raise ContractError("tattoo_program_over_budget", "$.layers", "hatch exceeds 20000 rows")
        # Serpentine chains: each row piece, in offset then along-row order,
        # joins the nearest open chain whose link is admissible (a hole splits
        # a row into pieces that continue separate chains), else starts one.
        chains: list[np.ndarray] = []
        for offset in offsets:
            base = offset * normal + centre * axis
            line = LineString([base - reach * axis, base + reach * axis]).intersection(polygon)
            pieces = sorted((np.asarray(piece.coords) for piece in _parts(line, "LineString")),
                            key=lambda points: float(np.min(points @ axis)))
            for points in pieces:
                if len(points) < 2 or _polyline_length(points) <= 1e-12:
                    continue
                joined = _join_chain(chains, points, center_domain, width_m)
                if not joined:
                    chains.append(points)
        output.extend(chains)
        provenance.extend((HATCH_DEPTH, float(polygon.area), owner) for _ in chains)
        if len(output) > MAX_FILL_STROKES:
            raise ContractError("tattoo_program_over_budget", "$.layers", "fill exceeds 100000 strokes")
    return output, provenance


def _join_chain(chains: list[np.ndarray], points: np.ndarray, center_domain, width_m: float) -> bool:
    """Append a row piece to the nearest open chain reachable by a short link
    inside the centre domain, reversing the piece when its far end is nearer."""
    if not chains:
        return False
    ends = np.array([chain[-1] for chain in chains])
    forward = np.linalg.norm(ends - points[0], axis=1)
    backward = np.linalg.norm(ends - points[-1], axis=1)
    order = np.argsort(np.minimum(forward, backward), kind="stable")
    for index in order[:4]:
        reverse = backward[index] < forward[index]
        distance = backward[index] if reverse else forward[index]
        if distance > width_m * 2:
            break
        candidate = points[::-1] if reverse else points
        if center_domain.covers(LineString([ends[index], candidate[0]])):
            chains[index] = np.concatenate([chains[index], candidate])
            return True
    return False


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _owner(tree: STRtree, components: list[Polygon], polygon: Polygon) -> int:
    """Index of the paint component an inset polygon lies in."""
    owners = tree.query(polygon.representative_point(), predicate="within")
    return int(owners[0]) if len(owners) else int(np.argmin([c.distance(polygon) for c in components]))


# Contour-then-hatch: this many boundary rings (the depth-0 outline and one
# inner ring at 0.45 x width) before the centre is hatched.
HATCH_RINGS = 2
# Hatch rows are reported at this inset depth, so the schedule tiers them as
# detail (tier 2) behind the rings' own depth tiers.
HATCH_DEPTH = 3


def fill_paths(target, width_m: float, *, coverage_floor: float | None = None,
               fill_style: str = "concentric") -> list[np.ndarray]:
    """Planned centrelines; no boundary-centered overpainting."""
    return [path.points for path in fill_plan(target, width_m, coverage_floor=coverage_floor, fill_style=fill_style)]


def fill_plan(target, width_m: float, *, coverage_floor: float | None = None,
              fill_style: str = "concentric") -> list[FillPath]:
    """Planned centrelines with their inset depth and paint component.

    `concentric` (the default) insets the centre domain ring by ring;
    `hatch` draws `HATCH_RINGS` boundary rings and fills what remains with
    parallel rows along each component's long axis. Both are qualified the
    same way: the radius guard compensates for the polygonal approximation of
    round buffers, and ideal coverage is checked for every connected
    component, so a small lost dot cannot hide in the area of a large fill.
    This is a geometric check at the declared planning width, not a
    calibrated tool/deposition model.
    """
    if not math.isfinite(width_m) or width_m <= 0:
        raise ContractError("out_of_range", "$.width_m", "expected positive finite planning width")
    if fill_style not in FILL_STYLES:
        raise ContractError("wrong_enum", "$.fill_style", f"expected one of {FILL_STYLES}, got {fill_style!r}")
    if target.is_empty:
        return []  # Explicit negative space can remove the complete target.
    radius = width_m / 2
    safe_radius = radius / math.cos(math.pi / (4 * ARC_SEGMENTS)) + 1e-7
    center_domain, narrow = _center_domain(target, safe_radius)
    if center_domain.is_empty:
        raise ContractError("tattoo_program_unsupported", "$.layers", "paint is narrower than the planning width")
    xmin, ymin, xmax, ymax = center_domain.bounds
    requested_rows = max(ymax - ymin, xmax - xmin) / (width_m * 0.45)
    if not math.isfinite(requested_rows) or requested_rows > MAX_FILL_ROWS:
        raise ContractError("tattoo_program_over_budget", "$.layers", "fill exceeds 20000 rows")
    components = list(_parts(target, "Polygon"))
    if fill_style == "hatch":
        output, provenance = _inset_rings(center_domain, width_m, HATCH_RINGS, components)
        rows, row_provenance = hatch_paths(center_domain, width_m, HATCH_RINGS, components)
        output += rows
        provenance += row_provenance
    else:
        output, provenance = _inset_rings(center_domain, width_m, max(1, math.ceil(requested_rows)), components)
    # Repeated insets of tessellated curves create thousands of sub-micron
    # vertices. Remove at most 50 nm of chord detail, below the 100 nm radius
    # guard above; the unchanged footprint/each-component gates below still
    # qualify the resulting paths, including the boundaries of holes.
    simplified = []
    for points in output:
        candidate = np.asarray(LineString(points).simplify(5e-8, preserve_topology=False).coords)
        simplified.append(candidate if len(np.unique(candidate, axis=0)) >= 2 else points)
    output = simplified
    _qualify(target, components, output, width_m, narrow, safe_radius, coverage_floor)
    return [FillPath(points, depth, island_area, components[owner])
            for points, (depth, island_area, owner) in zip(output, provenance, strict=True)]


def _qualify(target, components, output, width_m, narrow, safe_radius, coverage_floor) -> None:
    """Refuse a plan that spills outside the paint or loses a component."""
    coverage = deposited_coverage(output, width_m)
    # Ink may land outside the artwork only in the halo the tool needs to trace
    # paint thinner than itself. Everywhere else the old rule stands: not one
    # part in a hundred million outside the painted region.
    allowed = target
    if narrow:
        thin = union_all(narrow, grid_size=PAINT_GRID_M)
        halo = normalize(thin.buffer(safe_radius, quad_segs=ARC_SEGMENTS))
        drawn = coverage.intersection(halo).area
        if drawn > NARROW_OVERDRAW_LIMIT * thin.area:
            raise ContractError(
                "tattoo_program_unsupported", "$.layers",
                f"paint is narrower than the planning width, and tracing it would lay "
                f"{drawn / thin.area:.2f}x its area in ink (limit {NARROW_OVERDRAW_LIMIT})")
        allowed = normalize(union_all([target, halo], grid_size=PAINT_GRID_M))
    spill = coverage.difference(allowed).area
    if spill > max(1e-16, target.area * 1e-8):
        raise ContractError("tattoo_program_unsupported", "$.layers", f"finite-width fill spills outside paint ({spill:.9g} m2 / {target.area:.9g} m2)")
    floor = PAINT_COVERAGE_FLOOR if coverage_floor is None else float(coverage_floor)
    for component in components:
        fraction = coverage.intersection(component).area / component.area
        if fraction < floor:
            raise ContractError("tattoo_program_unsupported", "$.layers",
                                f"planning width loses paint component coverage ({fraction:.6f} < {floor}; "
                                f"area={component.area:.9g} m2, bounds={component.bounds})")


def deposited_coverage(paths: Sequence[np.ndarray], width_m: float):
    """Ideal round-footprint coverage for geometric qualification, not physics."""
    if not paths:
        return GeometryCollection()
    return union_all([LineString(points).buffer(width_m / 2, quad_segs=ARC_SEGMENTS) for points in paths])
