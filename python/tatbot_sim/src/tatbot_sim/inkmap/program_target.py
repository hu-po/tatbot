"""Deterministic canonical target rasters for typed Inkmap scenarios.

This is reference intent, not deposited ink.  It rasterizes the admitted
``TattooProgram`` in the placement chart at a documented metric resolution;
runtime pigment remains exclusively in :class:`tatbot_sim.inkfield.InkField`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely import contains_xy, prepare, union_all
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from tatbot_sim.human_rep.contracts import validate_contract
from tatbot_sim.human_rep.fill_geometry import PAINT_GRID_M, close_paint_seams
from tatbot_sim.human_rep.ink_program import canvas_to_chart, cubic_points, material_width

TARGET_RASTER_VERSION = 3
DEFAULT_PIXELS_PER_M = 4_000
SUPERSAMPLE = 4


@dataclass(frozen=True)
class ProgramTarget:
    """Canonical chart-space appearance and labels.

    Array row zero is the chart's negative-y edge and column zero its
    negative-x edge. ``coverage`` is effective deposited opacity in [0, 1].
    ``color_srgb`` is unassociated source color (zero where coverage is zero).
    IDs are discrete and must only be sampled with nearest-neighbour rules.
    """

    coverage: np.ndarray
    color_srgb: np.ndarray
    layer_id: np.ndarray
    placement_id: np.ndarray
    width_m: float
    height_m: float
    pixels_per_m: int
    supersample: int
    tattoo_program_sha256: str
    surface_placement_sha256: str
    tool_width_m: float | None = None

    @property
    def rows(self) -> int:
        return int(self.coverage.shape[0])

    @property
    def cols(self) -> int:
        return int(self.coverage.shape[1])

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"tatbot.program-target/{TARGET_RASTER_VERSION}\0".encode())
        for value in (self.coverage, self.color_srgb, self.layer_id, self.placement_id):
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(
            json.dumps(
                {
                    "width_m": self.width_m,
                    "height_m": self.height_m,
                    "pixels_per_m": self.pixels_per_m,
                    "supersample": self.supersample,
                    "tool_width_m": self.tool_width_m,
                    "tattoo_program_sha256": self.tattoo_program_sha256,
                    "surface_placement_sha256": self.surface_placement_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        return digest.hexdigest()


def _placed_points(points: Any, program: dict[str, Any], placement: dict[str, Any]) -> np.ndarray:
    # The patch's metric frame already carries placement rotation (the same
    # split used by browser buildDecal). The target is a texture in that frame,
    # so baking rotation here would apply it twice on the body. Mirror remains
    # a texture operation and is intentionally preserved.
    texture_placement = {**placement, "rotation_rad": 0.0}
    return canvas_to_chart(np.asarray(points, dtype=np.float64), program, texture_placement)


def _element_geometry(
    element: dict[str, Any], program: dict[str, Any], placement: dict[str, Any], tool_width_m: float | None
) -> BaseGeometry:
    kind = element["kind"]
    width = material_width(element["width_m"], tool_width_m)
    if kind == "cubic_bezier":
        points = _placed_points(cubic_points(element["control_points_m"]), program, placement)
        return LineString(points).buffer(width / 2.0, cap_style="round", join_style="round")
    points = _placed_points(element["points_m"], program, placement)
    if kind in {"dots", "stipple"}:
        return unary_union([Point(point).buffer(width / 2.0) for point in points])
    if element["fill"]:
        return Polygon(points)
    if element["closed"] and not np.array_equal(points[0], points[-1]):
        points = np.concatenate([points, points[:1]])
    return LineString(points).buffer(width / 2.0, cap_style="round", join_style="round")


def _raster_geometry(
    geometry: BaseGeometry, rows: int, cols: int, width_m: float, height_m: float,
    *, row_start: int = 0, total_rows: int | None = None,
) -> np.ndarray:
    mask = np.zeros((rows, cols), dtype=np.uint8)
    # Sample pixel centers. OpenCV fillPoly includes both rounded boundary
    # pixels, which expands narrow strokes and erodes holes even after
    # supersampling. Prepared geometry keeps this exact point test bounded.
    prepare(geometry)
    x = (np.arange(cols) + .5) * (width_m / cols) - width_m / 2
    for start in range(0, rows, 128):
        y = (np.arange(start, min(rows, start + 128)) + row_start + .5) * (height_m / (total_rows or rows)) - height_m / 2
        mask[start:start + len(y)] = contains_xy(geometry, x[None, :], y[:, None]).astype(np.uint8) * 255
    return mask


def render_program_target(
    tattoo_program: dict[str, Any],
    placement: dict[str, Any],
    *,
    pixels_per_m: int = DEFAULT_PIXELS_PER_M,
    supersample: int = SUPERSAMPLE,
    tool_width_m: float | None = None,
) -> ProgramTarget:
    """Rasterize metric intent; line widths stay fixed as geometry is resized.

    A known tool width overrides conversion widths, as in material compilation.
    Filled regions still scale as artwork; only their deposition paths depend
    on pen width, not the region boundary used as the target.
    """

    program = validate_contract(tattoo_program, expected_schema="tatbot.tattoo-program/1")
    if placement.get("schema") not in ("tatbot.surface-placement/1", "tatbot.surface-placement/2"):
        raise ValueError("target requires a surface placement")
    placed = validate_contract(placement)
    if placed["tattoo_program_sha256"] != program["content_sha256"]:
        raise ValueError("target program and surface placement hashes differ")
    if pixels_per_m < DEFAULT_PIXELS_PER_M:
        raise ValueError(f"program target requires at least {DEFAULT_PIXELS_PER_M} pixels/m")
    if supersample < 1:
        raise ValueError("supersample must be positive")

    width_m, height_m = (float(value) for value in placed["physical_scale_m"])
    rows = max(1, math.ceil(height_m * pixels_per_m - 1e-9))
    cols = max(1, math.ceil(width_m * pixels_per_m - 1e-9))
    high_rows, high_cols = rows * supersample, cols * supersample
    paint_runs = []
    colors = {ink["id"]: np.asarray(ink["color_srgb"], dtype=np.float32) for ink in program["inks"]}
    negative = unary_union(
        [Polygon(_placed_points(mask["points_m"], program, placed)) for mask in program["negative_space_masks"]]
    )
    for layer_index, layer in enumerate(program["layers"], start=1):
        color = colors[layer["ink_id"]]
        # SVG paint regions arrive as many adjacent triangles. Rasterizing and
        # alpha-compositing every triangle independently creates false seams at
        # shared edges and is quadratic in raster area. A layer is an ordered
        # set of paint runs: union consecutive elements with the same
        # deposition, then composite each run once. Deposition changes retain
        # their declared order.
        runs: list[tuple[float, list[BaseGeometry]]] = []
        for element in layer["elements"]:
            geometry = _element_geometry(element, program, placed, tool_width_m)
            deposition = float(element["deposition"])
            if runs and runs[-1][0] == deposition:
                runs[-1][1].append(geometry)
            else:
                runs.append((deposition, [geometry]))
        for deposition, geometries in runs:
            geometry = close_paint_seams(union_all(geometries, grid_size=PAINT_GRID_M))
            if not negative.is_empty:
                geometry = geometry.difference(negative)
            paint_runs.append((layer_index, color, deposition, geometry))

    # Tile the high-resolution composition: memory scales with one row band,
    # not with supersample squared times the entire color canvas. Compose
    # before downsampling so overlapping layers retain exact alpha semantics.
    coverage = np.zeros((rows, cols), dtype=np.float32)
    premultiplied = np.zeros((rows, cols, 3), dtype=np.float32)
    layer_id = np.zeros((rows, cols), dtype=np.int32)
    for start in range(0, rows, 32):
        count = min(32, rows - start)
        shape = (count * supersample, high_cols)
        alpha = np.zeros(shape, dtype=np.float32)
        color_tile = np.zeros((*shape, 3), dtype=np.float32)
        ids = np.zeros(shape, dtype=np.int32)
        for layer_index, color, deposition, geometry in paint_runs:
            raster = _raster_geometry(geometry, shape[0], high_cols, width_m, height_m,
                                     row_start=start * supersample, total_rows=high_rows)
            source_alpha = raster.astype(np.float32) * (deposition / 255.0)
            color_tile = source_alpha[..., None] * color + color_tile * (1 - source_alpha[..., None])
            alpha = source_alpha + alpha * (1 - source_alpha)
            ids[source_alpha > 0] = layer_index
        coverage[start:start + count] = cv2.resize(alpha, (cols, count), interpolation=cv2.INTER_AREA)
        premultiplied[start:start + count] = cv2.resize(color_tile, (cols, count), interpolation=cv2.INTER_AREA)
        offset = supersample // 2
        layer_id[start:start + count] = ids[offset::supersample, offset::supersample][:count, :cols]
    color_srgb = np.zeros_like(premultiplied)
    np.divide(premultiplied, coverage[..., None], out=color_srgb, where=coverage[..., None] > 1e-8)
    placement_id = (layer_id > 0).astype(np.int32)
    return ProgramTarget(
        coverage=np.clip(coverage, 0.0, 1.0),
        color_srgb=np.clip(color_srgb, 0.0, 1.0),
        layer_id=layer_id,
        placement_id=placement_id,
        width_m=width_m,
        height_m=height_m,
        pixels_per_m=pixels_per_m,
        supersample=supersample,
        tattoo_program_sha256=program["content_sha256"],
        surface_placement_sha256=placed["content_sha256"],
        tool_width_m=tool_width_m,
    )


def write_program_target(root: Path, target: ProgramTarget, skin_tone: str) -> dict[str, Any]:
    """Write review/label artifacts; never return a texture for runtime ink state."""

    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "target-labels.npz",
        coverage=target.coverage,
        color_srgb=target.color_srgb,
        layer_id=target.layer_id,
        placement_id=target.placement_id,
    )
    rgb = np.asarray([int(skin_tone[index : index + 2], 16) for index in (1, 3, 5)], dtype=np.float32) / 255.0
    preview = rgb[None, None, :] * (1.0 - target.coverage[..., None]) + target.color_srgb * target.coverage[..., None]
    cv2.imwrite(str(root / "target-reference.png"), np.rint(preview[..., ::-1] * 255.0).astype(np.uint8))
    cv2.imwrite(str(root / "target-coverage.png"), np.rint(target.coverage * 255.0).astype(np.uint8))
    manifest = {
        "schema": "tatbot.program-target-artifacts/1",
        "renderer_version": TARGET_RASTER_VERSION,
        "sha256": target.digest(),
        "tattoo_program_sha256": target.tattoo_program_sha256,
        "surface_placement_sha256": target.surface_placement_sha256,
        "width_m": target.width_m,
        "height_m": target.height_m,
        "pixels_per_m": target.pixels_per_m,
        "supersample": target.supersample,
        "tool_width_m": target.tool_width_m,
        "shape": [target.rows, target.cols],
        "coverage": f"float32 effective opacity; area-downsampled from {target.supersample}x binary geometry",
        "color_srgb": "float32 unassociated sRGB; zero where coverage is zero",
        "integer_ids": "int32; 0 background, nearest sampled; layer is 1-based source order, placement is 1",
        "image_origin": "row 0 is chart negative-y; column 0 is chart negative-x",
        "placement_mapping": "rotation is carried by the surface chart; mirror is baked into the texture",
        "runtime_ink": "separate and blank at episode reset",
    }
    (root / "target-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
