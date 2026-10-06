"""Offline source/fixture conversion helpers and metric placement utilities.

The SVG/raster paint adapter remains available to source tooling and contract
fixtures. Its records are not acquired DBV3 artwork: production CLI, editor,
simulator ingestion and ROS preparation refuse them. Native acquisition owns
finished drawing paths; the placement helpers below can place its unchanged
artwork record on a plane or cylinder.

Nominal chart dimensions are not measured pad dimensions, and a design carries
no tool, pose, or authority to move anything; `tatbot ros compile` binds the
fitted tool and the registered page separately.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from tatbot_contracts.artwork import canvas_m as artwork_canvas_m
from tatbot_contracts.artwork import max_width_m

from tatbot_sim.human_rep.ink_program import material_strokes, target_stroke_points
from tatbot_sim.human_rep.placement import make_target_placement
from tatbot_sim.inkmap.artwork import make_artwork_record
from tatbot_sim.inkmap.design import make_design

# Source/fixture paint conversion defaults, also used by the inkgen materializer.
# These do not govern the acquired artwork collection or production preparation.
ADAPTER = "tatbot-svg-paint/1"
DEFAULT_WIDTH_MM = 0.3
# How a stroked SVG path is drawn: `outline` regionises it at its stroke-width
# and the fill planner rings it (the pen draws both edges); `centerline` keeps
# the line as one open path at the planning width, drawn once. A record made
# in the default mode carries no `strokes` field, so its digest is unchanged.
STROKE_MODES = ("outline", "centerline")
CHORD_ERROR_M = 0.000005
TRACE_MODEL = "inkmap-vtracer-otsu-v2"
# The browser's own limits (web/inkmap/tools/trace-raster.ts, svg-program.ts).
MAX_PIXELS = 4_194_304
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_SVG_BYTES = 2 * 1024 * 1024
EPOCH = "2026-09-06T00:00:00Z"
# PNG and JPEG only: the browser's tracer is fed RGBA, and anything else here
# would be a decoder difference between the two callers rather than a feature.
IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpeg"))


class DesignBuildError(ValueError):
    """An input could not become a design, with the reason a caller can act on."""


def image_kind(data: bytes) -> str:
    for magic, kind in IMAGE_MAGIC:
        if data.startswith(magic):
            return kind
    raise DesignBuildError("expected a PNG or JPEG image (the tracer reads no other format)")


def decode_image(data: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes to BGR, refusing anything the tracer cannot take."""
    import cv2

    if not data or len(data) > MAX_IMAGE_BYTES:
        raise DesignBuildError(f"image size must be 1-{MAX_IMAGE_BYTES} bytes; got {len(data)}")
    image_kind(data)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise DesignBuildError("image is not decodable as a colour raster")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise DesignBuildError(f"raster exceeds the tracer's {MAX_PIXELS} pixel budget")
    return image


def trace_raster_svg(image_bgr: np.ndarray) -> tuple[str, dict[str, Any]]:
    """Inkmap's exact Otsu/vtracer paint conversion, including holes."""
    import cv2

    from tatbot_sim.human_rep.artwork_client import artwork_request

    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)
    if rgba.shape[0] * rgba.shape[1] > MAX_PIXELS:
        raise DesignBuildError("raster exceeds pixel budget")
    try:
        result = artwork_request({"operation": "trace_raster", "width": rgba.shape[1],
                                  "height": rgba.shape[0],
                                  "rgba_base64": base64.b64encode(rgba.tobytes()).decode()})
    except ValueError as exc:
        raise DesignBuildError(str(exc)) from exc
    return result["svg"], result["trace"]


def trace_image(data: bytes) -> tuple[str, dict[str, Any]]:
    """PNG/JPEG bytes to painted SVG, through the browser's own tracer."""
    return trace_raster_svg(decode_image(data))


def fit_size_mm(size_px: tuple[float, float], box_mm: tuple[float, float]) -> tuple[float, float]:
    """The largest box_mm-bounded size with the traced raster's aspect ratio."""
    if min(size_px) <= 0 or not all(np.isfinite(size_px)):
        raise DesignBuildError(f"traced size must be positive and finite, got {size_px!r}")
    if min(box_mm) <= 0 or not all(np.isfinite(box_mm)):
        raise DesignBuildError(f"design size must be positive and finite, got {box_mm!r}")
    scale = min(box_mm[0] / size_px[0], box_mm[1] / size_px[1])
    return (float(size_px[0] * scale), float(size_px[1] * scale))


def _source(kind: str, *, identifier: str | None, license: str | None = None,
            attribution: str | None = None, generation: dict[str, Any] | None = None) -> dict[str, Any]:
    # Unknown provenance stays null: the schema calls that unknown, not permission.
    return {"kind": kind, "identifier": identifier, "license": license,
            "attribution": attribution, "generation": generation}


def file_source(svg: str, *, path: str | None = None, trace: dict[str, Any] | None = None,
                image_sha256: str | None = None) -> dict[str, Any]:
    """Provenance for artwork that came off this machine, traced or hand-drawn.

    `kind` is `imported` because that is the schema's word for it; a local path
    is descriptive only and is never resolved or fetched by any reader.
    """
    identifier = json.dumps({"path": path, "svg_sha256": hashlib.sha256(svg.encode()).hexdigest(),
                             **({"image_sha256": image_sha256} if image_sha256 else {}),
                             **({"trace": trace} if trace else {})}, sort_keys=True)
    return _source("imported", identifier=identifier)


def generated_source(*, prompt: str, model: str | None, seed: int, model_revision: str | None = None,
                     identifier: str | None = None) -> dict[str, Any]:
    return _source("generated", identifier=identifier,
                   generation={"prompt": prompt, "model": model, "model_revision": model_revision,
                               "seed": int(seed), "tracing": TRACE_MODEL})


def artwork_from_svg(svg: str, *, name: str, size_mm: tuple[float, float], source: dict[str, Any],
                     width_mm: float = DEFAULT_WIDTH_MM, strokes: str = "outline") -> dict[str, Any]:
    """A source/fixture paint record, excluded from acquired-artwork admission."""
    if len(svg.encode()) > MAX_SVG_BYTES:
        raise DesignBuildError(f"SVG exceeds {MAX_SVG_BYTES} bytes")
    if not name.strip():
        raise DesignBuildError("artwork needs a name")
    if min(size_mm) <= 0 or not all(np.isfinite(size_mm)):
        raise DesignBuildError(f"artwork size must be positive and finite, got {size_mm!r}")
    if not 0 < width_mm <= 2:
        raise DesignBuildError(f"planning width must be in (0, 2] mm, got {width_mm}")
    if strokes not in STROKE_MODES:
        raise DesignBuildError(f"stroke mode must be one of {', '.join(STROKE_MODES)}, got {strokes!r}")
    conversion: dict[str, Any] = {
        "adapter": ADAPTER, "canvas_m": [float(v) / 1000 for v in size_mm],
        "semantic_intent": name.strip(), "width_m": width_mm / 1000, "deposition": 1,
        "chord_error_m": CHORD_ERROR_M,
    }
    if strokes != "outline":
        conversion["strokes"] = strokes
    try:
        return make_artwork_record(name=name.strip(), original_svg=svg, source=source, conversion=conversion)
    except ValueError as exc:
        raise DesignBuildError(str(exc)) from exc


def enclosing_canvas_m(record: dict[str, Any], *, rotation_rad: float = 0.0,
                       anchor_uv_m: tuple[float, float] = (0.0, 0.0),
                       margin_m: float = 0.0) -> list[float]:
    """The smallest chart that holds the rotated artwork, its offset and its margin.

    The default when a caller states no canvas. It is nominal geometry, not a
    measured pad: `--canvas-mm` is how a caller says what the real target is.
    """
    size = np.asarray(artwork_canvas_m(record), dtype=float)
    cosine, sine = abs(math.cos(rotation_rad)), abs(math.sin(rotation_rad))
    extent = np.asarray([[cosine, sine], [sine, cosine]]) @ size
    border = 2 * float(max_width_m(record)) + 2 * float(margin_m)
    return (extent + border + 2 * np.abs(np.asarray(anchor_uv_m, dtype=float))).tolist()


def _target(kind: str, *, canvas_m: list[float], anchor_uv_m: tuple[float, float], margin_m: float,
            radius_m: float | None) -> dict[str, Any]:
    if min(canvas_m) <= 0 or not all(np.isfinite(canvas_m)):
        raise DesignBuildError(f"canvas must be positive and finite, got {canvas_m!r}")
    if margin_m < 0 or not math.isfinite(margin_m):
        raise DesignBuildError(f"margin must not be negative, got {margin_m}")
    target = {"kind": kind, "canvas_m": [float(v) for v in canvas_m],
              "anchor_uv_m": [float(v) for v in anchor_uv_m], "margin_m": float(margin_m)}
    if kind == "cylinder":
        if radius_m is None or not math.isfinite(radius_m) or radius_m <= 0:
            raise DesignBuildError(f"a cylinder needs a positive finite radius, got {radius_m!r}")
        # v is arc length. Past half the circumference the chart's own surface
        # normal has swung more than 90 degrees from the top of the cylinder, so
        # the far side of the wrap is not a face a tool approaches along the
        # normal. Refuse the chart rather than compile artwork onto it.
        if target["canvas_m"][1] >= 2 * math.pi * radius_m - 1e-12:
            raise DesignBuildError(
                f"cylinder chart wraps {target['canvas_m'][1] * 1000:.3f} mm of arc, the full "
                f"{2 * math.pi * radius_m * 1000:.3f} mm circumference of a {radius_m * 1000:.3f} mm radius")
        target["radius_m"] = float(radius_m)
    return target


def analytic_placement(record: dict[str, Any], *, kind: str = "plane", radius_m: float | None = None,
                       canvas_m: tuple[float, float] | None = None,
                       anchor_uv_m: tuple[float, float] = (0.0, 0.0), margin_m: float = 0.0,
                       rotation_rad: float = 0.0, mirrored: bool = False,
                       producer: str = "tatbot-design-build") -> dict[str, Any]:
    """One placement of one artwork on a plane or cylinder chart.

    The material itself is the check: every stroke is compiled and passed
    through the target's own margin rule, so artwork that leaves the canvas is
    refused here rather than at preparation.
    """
    if kind not in {"plane", "cylinder"}:
        raise DesignBuildError(f"analytic target must be plane or cylinder, got {kind!r}")
    if not math.isfinite(rotation_rad):
        raise DesignBuildError("rotation must be finite")
    canvas = list(canvas_m) if canvas_m is not None else enclosing_canvas_m(
        record, rotation_rad=rotation_rad, anchor_uv_m=anchor_uv_m, margin_m=margin_m)
    target = _target(kind, canvas_m=canvas, anchor_uv_m=anchor_uv_m, margin_m=margin_m, radius_m=radius_m)
    placement = make_target_placement(
        tattoo_program_sha256=record["program"]["content_sha256"], target=target,
        physical_scale_m=[float(v) for v in artwork_canvas_m(record)],
        rotation_rad=float(rotation_rad), mirrored=bool(mirrored),
        review={"status": "pending", "reviewer": f"offline {kind} artwork compiler",
                "evidence_sha256": "0" * 64},
        provenance={"producer": producer, "version": "1", "created_utc": EPOCH,
                    "source_sha256": record["content_sha256"]},
    )
    for stroke in material_strokes(record["program"], placement):
        target_stroke_points(stroke, placement["target"])  # refuses outside the margin
    return placement


def plane_placement(record: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return analytic_placement(record, kind="plane", **kwargs)


def cylinder_placement(record: dict[str, Any], *, radius_m: float, **kwargs: Any) -> dict[str, Any]:
    return analytic_placement(record, kind="cylinder", radius_m=radius_m, **kwargs)


def design_from_placements(name: str, entries: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """`tatbot.inkmap-design/1` from (placement id, artwork record, placement) triples."""
    if not entries:
        raise DesignBuildError("a design needs at least one placement")
    artworks: dict[str, dict[str, Any]] = {}
    placements = []
    for index, (placement_id, record, placement) in enumerate(entries):
        artwork_id = f"{placement_id}-art" if placement_id in artworks else placement_id
        artworks[artwork_id] = record
        placements.append({"id": placement_id or f"placement-{index}", "artwork_id": artwork_id,
                           "placement": placement})
    try:
        return make_design(name=name, artworks=artworks, placements=placements)
    except ValueError as exc:
        raise DesignBuildError(str(exc)) from exc


def design_from_artwork(record: dict[str, Any], *, name: str | None = None, placement_id: str = "placement-1",
                        **placement_kwargs: Any) -> dict[str, Any]:
    """The common case: one artwork, one placement, one design."""
    placement = analytic_placement(record, **placement_kwargs)
    return design_from_placements(name or record["name"], [(placement_id, record, placement)])


def write_json(path: Path, document: dict[str, Any]) -> Path:
    """Write a document exactly once; a design is evidence, never overwritten."""
    path = Path(path)
    if path.exists():
        raise DesignBuildError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path
