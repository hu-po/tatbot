"""TattooProgram and target placement -> strict, versioned InkProgram.

The compiler is deterministic and body/session separated: it materializes
canonical body/metric-chart curves and ink-state intent, but it never resolves cap
poses, calibration, measured geometry, robot paths, or samples.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import shapely

from tatbot_sim import tools
from tatbot_sim.human_rep.contracts import (
    ContractError,
    canonical_bytes,
    canonical_digest,
    validate_contract,
)
from tatbot_sim.human_rep.fill_geometry import fill_plan, paint_union
from tatbot_sim.human_rep.placement import make_surface_curve
from tatbot_sim.human_rep.stroke_schedule import schedule as schedule_strokes
from tatbot_sim.inkmap.rig import BodyRig, load_body_rig
from tatbot_sim.inkmap.surface_trace import compile_surface_trace

SCHEMA = "tatbot.ink-program/1"
COMPILER_NAME = "tatbot-human-representation-ink-program"
COMPILER_VERSION = "3"
DEFAULT_INK_ID_MAP = {
    "black": "nighthawk_black",
    "blue": "true_blue",
    "red": "bright_red",
    "green": "lime_green",
    "magenta": "fuchsia",
    "white": "snow_white",
}


@dataclass(frozen=True)
class MaterialStroke:
    points_m: np.ndarray
    ink_id: str
    width_m: float
    deposition: float
    source_primitive_sha256: str
    ordering_rationale: str
    # Schedule provenance (never part of the compiled document): the source
    # layer; for a fill ring its inset depth, the area of the inset island it
    # bounds and the paint component it lies in (all None for an explicit
    # element); the tier the schedule ranks it in and, once scheduled, its
    # final position. See stroke_schedule.
    layer_index: int = 0
    inset_depth: int | None = None
    component_area_m2: float | None = None
    component: Any = None
    tier: int = 0
    schedule_index: int | None = None


def stroke_tier(inset_depth: int | None) -> int:
    """Tier 0: explicit elements and boundary rings; 1: insets 1-2; 2: deeper."""
    if inset_depth is None or inset_depth == 0:
        return 0
    return 1 if inset_depth <= 2 else 2


class InkProgramError(ContractError):
    """Contract-shaped refusal raised before any session state is consulted."""


def _compiler_digest() -> str:
    root = Path(__file__).resolve().parents[5]
    paths = (
        Path(__file__),
        Path(__file__).with_name("fill_geometry.py"),
        Path(__file__).with_name("stroke_schedule.py"),
        root / "scripts" / "lib" / "ink_program.py",
        root / "scripts" / "lib" / "ink_spec.py",
        root / "python" / "tatbot_sim" / "src" / "tatbot_sim" / "inkmap" / "surface_trace.py",
    )
    digest = hashlib.sha256()
    digest.update(f"shapely:{shapely.__version__};geos:{shapely.geos_version_string}\0".encode())
    for path in paths:
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _deduplicate(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise InkProgramError("tattoo_program_invalid", "$.layers", "material stroke is not finite (N,2)")
    if len(value) < 2:
        return value
    keep = np.concatenate([[True], np.linalg.norm(np.diff(value, axis=0), axis=1) > 1e-12])
    return value[keep]


def cubic_points(controls: Sequence[Sequence[float]]) -> np.ndarray:
    control = np.asarray(controls, dtype=np.float64)
    t = np.linspace(0.0, 1.0, 257)[:, None]
    return (
        (1 - t) ** 3 * control[0]
        + 3 * (1 - t) ** 2 * t * control[1]
        + 3 * (1 - t) * t**2 * control[2]
        + t**3 * control[3]
    )


def _dot_stroke(point: Sequence[float], width_m: float) -> np.ndarray:
    radius = max(float(width_m) / 2.0, 5e-5)
    angle = np.linspace(0.0, 2.0 * math.pi, 13)
    center = np.asarray(point, dtype=np.float64)
    return center[None] + radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def target_stroke_points(stroke: MaterialStroke, target: dict[str, Any]) -> np.ndarray:
    """Place material in an analytic target chart and enforce its physical margin."""
    if target["kind"] not in {"plane", "cylinder"}:
        raise InkProgramError("target_invalid", "$.target", "material needs an analytic target chart")
    points = stroke.points_m + np.asarray(target["anchor_uv_m"], dtype=float)
    half_canvas = np.asarray(target["canvas_m"], dtype=float) / 2
    if np.any(np.abs(points) + stroke.width_m / 2 + target["margin_m"] > half_canvas + 1e-12):
        raise InkProgramError("anchor_outside_domain", "$.target", "material footprint exceeds target margin")
    return points


def canvas_to_chart(points: np.ndarray, program: dict[str, Any], placement: dict[str, Any]) -> np.ndarray:
    width = float(program["canvas_m"]["width"])
    height = float(program["canvas_m"]["height"])
    target_width, target_height = (float(item) for item in placement["physical_scale_m"])
    value = np.asarray(points, dtype=np.float64) - np.asarray([width / 2.0, height / 2.0])
    value *= np.asarray([target_width / width, target_height / height])
    if placement["mirrored"]:
        value[:, 0] *= -1.0
    angle = float(placement["rotation_rad"])
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return value @ rotation.T


def material_width(element_width_m: float, tool_width_m: float | None = None) -> float:
    """A metric line width, independent of the artwork's placement scale."""
    width = float(element_width_m if tool_width_m is None else tool_width_m)
    if not math.isfinite(width) or width <= 0:
        raise InkProgramError("out_of_range", "$.tool_width_m", "expected a positive finite line width")
    return width


def _element_chart_pieces(element, program, placed, width):
    kind = element["kind"]
    if kind in {"dots", "stipple"}:
        # Move dot centres with the artwork, then draw circles at pen size in
        # chart space. Scaling a prebuilt circle would scale the pen as well.
        return [_dot_stroke(point, width) for point in canvas_to_chart(element["points_m"], program, placed)]
    if kind == "cubic_bezier":
        polygon = cubic_points(element["control_points_m"])
    else:
        polygon = np.asarray(element["points_m"], dtype=np.float64)
        if element["closed"] and not np.array_equal(polygon[0], polygon[-1]):
            polygon = np.concatenate([polygon, polygon[:1]])
    return [canvas_to_chart(polygon, program, placed)]


def material_strokes(
    tattoo_program: dict[str, Any],
    placement: dict[str, Any],
    *,
    ink_id_map: dict[str, str] | None = None,
    fill_style: str = "concentric",
    tool_width_m: float | None = None,
) -> list[MaterialStroke]:
    """Deterministically decompose line, fill, stipple, and color layers.

    `fill_style` selects the paint planner: `concentric` inset rings, or
    `hatch` (boundary rings, then rows along each region's long axis).
    Widths are physical metres: a known tool width overrides the artwork's
    conversion width for lines, dots and fill spacing; neither scales with size.
    """

    program = validate_contract(tattoo_program, expected_schema="tatbot.tattoo-program/1")
    if placement.get("schema") not in {"tatbot.surface-placement/1", "tatbot.surface-placement/2"}:
        raise InkProgramError("wrong_schema", "$.schema", "expected surface placement")
    placed = validate_contract(placement)
    if placed["tattoo_program_sha256"] != program["content_sha256"]:
        raise InkProgramError(
            "wrong_hash",
            "$.tattoo_program_sha256",
            "placement does not bind the supplied TattooProgram",
        )
    mapping = dict(DEFAULT_INK_ID_MAP)
    mapping.update(ink_id_map or {})
    masks = [np.asarray(mask["points_m"], dtype=np.float64) for mask in program["negative_space_masks"]]
    output: list[MaterialStroke] = []
    for layer_index, layer in enumerate(program["layers"]):
        source_ink = layer["ink_id"]
        ink_id = mapping.get(source_ink, source_ink)
        # Schema validation above guarantees ink_id is a string (never None).
        assert isinstance(ink_id, str)
        # The preview paints the union of filled polygons, not their tessellation
        # edges. Preserve differing width/deposition groups instead of flattening
        # them to one ink/material setting.
        fills: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for element_index, element in enumerate(layer["elements"]):
            digest = hashlib.sha256(canonical_bytes(element)).hexdigest()
            kind = element["kind"]
            width = material_width(element["width_m"], tool_width_m)
            if element["fill"] and kind in {"region", "path"}:
                fills.setdefault((width, element["deposition"]), []).append(element)
                continue
            for piece_index, piece in enumerate(_element_chart_pieces(element, program, placed, width)):
                chart = _deduplicate(piece)
                if len(chart) < 2 or _polyline_length(chart) <= 1e-9:
                    raise InkProgramError(
                        "tattoo_program_unsupported",
                        f"$.layers[{layer_index}].elements[{element_index}]",
                        "element materializes to a zero-length stroke",
                    )
                output.append(
                    MaterialStroke(
                        points_m=chart,
                        ink_id=ink_id,
                        width_m=width,
                        deposition=float(element["deposition"]),
                        source_primitive_sha256=digest,
                        ordering_rationale=(
                            f"source layer {layer_index}, element {element_index}, deterministic piece {piece_index}"
                            "; tier 0"
                        ),
                        layer_index=layer_index,
                    )
                )
        for (width, deposition), elements in fills.items():
            polygons = [canvas_to_chart(np.asarray(element["points_m"]), program, placed) for element in elements]
            chart_masks = [canvas_to_chart(mask, program, placed) for mask in masks]
            target = paint_union(polygons, chart_masks)
            digest = hashlib.sha256(canonical_bytes(elements)).hexdigest()
            for piece_index, piece in enumerate(fill_plan(target, width, fill_style=fill_style)):
                output.append(MaterialStroke(
                    points_m=_deduplicate(piece.points), ink_id=ink_id,
                    width_m=width, deposition=deposition,
                    source_primitive_sha256=digest,
                    ordering_rationale=(f"source layer {layer_index}, union of {len(elements)} filled primitives; "
                                        f"finite-width inset contour/hatch {piece_index}"
                                        f"; tier {stroke_tier(piece.depth)} depth {piece.depth}"),
                    layer_index=layer_index, inset_depth=piece.depth,
                    component_area_m2=piece.island_area_m2, component=piece.component,
                    tier=stroke_tier(piece.depth),
                ))
    if not output:
        raise InkProgramError("tattoo_program_unsupported", "$.layers", "no material strokes")
    return output


def scheduled_material_strokes(
    tattoo_program: dict[str, Any],
    placement: dict[str, Any],
    *,
    ink_id_map: dict[str, str] | None = None,
    footprint_width_m: float | None = None,
    report: dict | None = None,
    fill_style: str = "concentric",
) -> list[MaterialStroke]:
    """The strokes every consumer draws: materialized, tiered, dedup'd, ordered.

    `design check` and scan coverage go through here, so they agree on count
    and order. `footprint_width_m` is
    the fitted tool's measured line width for material and dedup when recorded; `report`
    receives the schedule audit (see stroke_schedule); `fill_style` is the
    paint planner (see fill_geometry.FILL_STYLES).
    """
    return schedule_strokes(material_strokes(tattoo_program, placement, ink_id_map=ink_id_map,
                                             fill_style=fill_style, tool_width_m=footprint_width_m),
                            footprint_width_m=footprint_width_m, report=report,
                            preserve_order=tattoo_program["provenance"]["producer"] == "dbv3-batik-paths/1")


def _split_polyline(points: np.ndarray, maximum_length_m: float) -> list[np.ndarray]:
    """Split exactly at arc-length limits, retaining each shared cut point."""

    if maximum_length_m <= 0.0:
        raise InkProgramError("stroke_over_capacity", "$.events", "non-positive capacity-derived length")
    points = _deduplicate(points)
    output: list[np.ndarray] = []
    current = [points[0].copy()]
    remaining = maximum_length_m
    for original_start, original_stop in zip(points[:-1], points[1:], strict=True):
        start = original_start.copy()
        stop = original_stop.copy()
        segment_length = float(np.linalg.norm(stop - start))
        while segment_length > remaining + 1e-12:
            cut = start + (stop - start) * (remaining / segment_length)
            current.append(cut)
            output.append(np.asarray(current))
            current = [cut]
            start = cut
            segment_length = float(np.linalg.norm(stop - start))
            remaining = maximum_length_m
        current.append(stop)
        remaining -= segment_length
        if remaining <= 1e-12:
            output.append(np.asarray(current))
            current = [stop]
            remaining = maximum_length_m
    if len(current) >= 2 and _polyline_length(np.asarray(current)) > 1e-12:
        output.append(np.asarray(current))
    return output


def _capacity_split(strokes: Sequence[MaterialStroke], policy: Any, speed_m_s: float) -> list[MaterialStroke]:
    if policy.mode == "none":
        raise InkProgramError("ink_supply_unavailable", "$.tool", "an ink-producing program cannot use ink.mode none")
    cost_per_m = float(policy.deposit_ul_per_mm) * 1000.0 + float(policy.bleed_ul_per_s) / speed_m_s
    if policy.dips and (policy.uptake_ul <= 0.0 or cost_per_m <= 0.0):
        raise InkProgramError("ink_policy_invalid", "$.tool", "dip policy cannot bound a stroke")
    maximum = math.inf if not policy.dips or cost_per_m == 0.0 else float(policy.uptake_ul) / cost_per_m
    output = []
    for stroke in strokes:
        pieces = _split_polyline(stroke.points_m, maximum) if _polyline_length(stroke.points_m) > maximum else [stroke.points_m]
        for piece_index, piece in enumerate(pieces):
            output.append(
                replace(
                    stroke,
                    points_m=piece,
                    ordering_rationale=(
                        stroke.ordering_rationale
                        if len(pieces) == 1
                        else f"{stroke.ordering_rationale}; capacity split {piece_index + 1}/{len(pieces)}"
                    ),
                )
            )
    return output


def _scheduled_split(program, placed, tool, policy, speed_m_s, ink_id_map, report,
                     fill_style="concentric") -> list[MaterialStroke]:
    """Schedule, then capacity-split; the report gains one tier/depth per split piece."""
    strokes = _capacity_split(
        scheduled_material_strokes(program, placed, ink_id_map=ink_id_map,
                                   footprint_width_m=getattr(tool, "line_width_m", None), report=report,
                                   fill_style=fill_style),
        policy,
        speed_m_s,
    )
    if report is not None:
        report["stroke_tiers"] = [stroke.tier for stroke in strokes]
        report["stroke_depths"] = [stroke.inset_depth for stroke in strokes]
    return strokes


def _nested_curve(curve: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in curve.items() if key not in {"schema", "content_sha256"}}


def compile_ink_program(
    tattoo_program: dict[str, Any],
    placement: dict[str, Any],
    *,
    tool: Any,
    policy: Any | None = None,
    rig: BodyRig | None = None,
    ink_id_map: dict[str, str] | None = None,
    speed_m_s: float = 0.004,
    orientation_tolerance_rad: float = 0.12,
    contact_envelope_m: tuple[float, float] = (-0.0002, 0.0002),
    initial_load_fraction: float = 0.0,
    operating_budget_s: float | None = None,
    provenance: dict[str, Any],
    schedule_report: dict | None = None,
    fill_style: str = "concentric",
) -> dict[str, Any]:
    """Compile target-bound design intent with one material/supply planner.

    `schedule_report`, when given, receives the stroke schedule audit plus
    `stroke_tiers` / `stroke_depths`, one entry per stroke event in the
    compiled order (capacity splits share their source stroke's entry). It is
    evidence beside the document, never part of its bytes. `fill_style` is
    the paint planner; a program compiled with a style other than the
    contract's default (`concentric`) carries it as `fill_style`, following
    the `operating_budget_s` precedent, so the document names what it needs
    to be reproduced.
    """

    if not math.isfinite(speed_m_s) or speed_m_s <= 0.0:
        raise InkProgramError("out_of_range", "$.speed_m_s", "must be positive")
    if not 0.0 <= initial_load_fraction <= 1.0:
        raise InkProgramError("out_of_range", "$.initial_load_fraction", "expected [0,1]")
    if placement.get("schema") not in {"tatbot.surface-placement/1", "tatbot.surface-placement/2"}:
        raise InkProgramError("wrong_schema", "$.schema", "expected surface placement")
    placed = validate_contract(placement)
    target = placed.get("target", placed)
    body = None
    if target.get("kind", "body") == "body":
        body = rig or load_body_rig()
        if body.part_names != ("SOMA",):
            raise InkProgramError("body_model_unsupported", "$.body", "expected the sole SOMA surface")
        validate_contract(placed, expected_topology_sha256=body.topology_sha256)
    program = validate_contract(tattoo_program, expected_schema="tatbot.tattoo-program/1")
    ink_policy = policy or tools.ink_registry().policy_for(tool)
    continuous = ink_policy.mode == "cartridge"
    if continuous:
        initial_load_fraction = 1.0
    exact = tools.ink_registry()
    try:
        required_tool_class = exact_program_module().tool_class(tool)
    except ValueError as exc:
        raise InkProgramError("tool_class_unsupported", "$.tool", str(exc)) from exc
    strokes = _scheduled_split(program, placed, tool, ink_policy, speed_m_s, ink_id_map, schedule_report,
                               fill_style)
    compiler_sha256 = _compiler_digest()
    curves: list[tuple[MaterialStroke, dict[str, Any]]] = []
    if body is None:
        for stroke in strokes:
            points = target_stroke_points(stroke, target)
            curves.append((stroke, {
                "coordinates": [{"target_sha256": placed["content_sha256"], "chart_uv_m": point.tolist()}
                                for point in points],
                "rest_surface_arc_length_m": _polyline_length(points),
                "width_m": stroke.width_m, "deposition": stroke.deposition, "direction": "forward",
                "source_primitive_sha256": stroke.source_primitive_sha256, "compiler_sha256": compiler_sha256,
            }))
    else:
        supported = {int(value) for value in target["supported_domain"]["face_indices"]}
        trace_placement = {
            "anchor": {
                "face": target["anchor"]["face_index"],
                "barycentric": target["anchor"]["barycentric"],
            },
            # Points have already received the placement rotation.
            "rotation_rad": 0.0,
            "size_mm": [float(value) * 1000 for value in placed["physical_scale_m"]],
        }
        trace = compile_surface_trace(
            body,
            trace_placement,
            [stroke.points_m for stroke in strokes],
        )
        for stroke_index, (stroke, anchors) in enumerate(zip(strokes, trace.strokes, strict=True)):
            unexpected = sorted({anchor.face for anchor in anchors} - supported)
            if unexpected:
                raise InkProgramError(
                    "anchor_outside_domain",
                    f"$.events[{stroke_index}]",
                    f"stroke reaches unsupported face {unexpected[0]}",
                )
            rest_points = [np.asarray(anchor.barycentric) @ body.rest_vertices[anchor.face] for anchor in anchors]
            curve = make_surface_curve(
                topology_sha256=body.topology_sha256,
                addresses=[(anchor.face, anchor.barycentric) for anchor in anchors],
                rest_points_m=rest_points,
                width_m=stroke.width_m,
                deposition=stroke.deposition,
                direction="forward",
                source_primitive_sha256=stroke.source_primitive_sha256,
                compiler_sha256=compiler_sha256,
            )
            curves.append((stroke, _nested_curve(curve)))

    first_ink = curves[0][0].ink_id
    remaining_fraction = float(initial_load_fraction)
    charge = exact.Charge(
        ul=float(initial_load_fraction) * float(ink_policy.charge_capacity_ul),
        capacity_ul=float(ink_policy.charge_capacity_ul),
        ink_id=first_ink,
    )
    events: list[dict[str, Any]] = [
        {
            "kind": "tool_change",
            "required_tool_class": required_tool_class,
            "transition_intent": "bind the declared tool class before material contact",
        }
    ]
    for stroke_index, (stroke, curve) in enumerate(curves):
        length_m = float(curve["rest_surface_arc_length_m"])
        contact_s = length_m / speed_m_s
        need_ul = float(ink_policy.stroke_ul(length_m * 1000.0, contact_s))
        reason = None
        if continuous:
            if stroke.ink_id != first_ink:
                raise InkProgramError("ink_supply_unavailable", "$.tool", "cartridge cannot change ink by dipping")
        elif stroke_index == 0 and charge.ul <= 0.0:
            reason = "session_start"
        elif charge.ink_id != stroke.ink_id:
            reason = "color_change"
        elif need_ul > charge.ul + 1e-12:
            reason = "low_charge"
        if reason is not None:
            if reason == "color_change":
                charge.ul = 0.0
            charge.credit(float(ink_policy.uptake_ul), stroke.ink_id)
            expected = charge.frac
            events.append(
                {
                    "kind": "dip",
                    "ink_id": stroke.ink_id,
                    "target_load": [expected, min(1.0, expected + 0.05)],
                    "trigger_reason": reason,
                    "dependency": f"before-stroke-{stroke_index}",
                    "expected_load_after": expected,
                }
            )
        if need_ul > charge.ul + 1e-9:
            raise InkProgramError(
                "stroke_over_capacity",
                f"$.events[{stroke_index}]",
                f"stroke needs {need_ul:.6g} uL after deterministic splitting; charge has {charge.ul:.6g} uL",
            )
        events.extend(
            [
                {"kind": "pen_transition", "state": "approach"},
                {"kind": "pen_transition", "state": "contact_intent"},
                {
                    "kind": "stroke",
                    "curve": curve,
                    "start_choice": "start",
                    "allowed_tool_class": required_tool_class,
                    "speed_m_s": float(speed_m_s),
                    "orientation_tolerance_rad": float(orientation_tolerance_rad),
                    "contact_envelope_m": [float(contact_envelope_m[0]), float(contact_envelope_m[1])],
                    "research_depth_band_m": None,
                    "ordering_rationale": stroke.ordering_rationale,
                },
                {"kind": "pen_transition", "state": "retract"},
            ]
        )
        charge.debit(need_ul)
        if stroke_index + 1 < len(curves) and curves[stroke_index + 1][0].ink_id != stroke.ink_id:
            events.append({"kind": "barrier", "intent": f"wipe before color change from {stroke.ink_id}"})

    document = {
        "schema": "tatbot.ink-program/2" if placed["schema"].endswith("/2") else SCHEMA,
        "content_sha256": "0" * 64,
        "tattoo_program_sha256": program["content_sha256"],
        "surface_placement_sha256": placed["content_sha256"],
        "compiler_sha256": compiler_sha256,
        "initial_ink_state": {"ink_id": first_ink, "load_fraction": float(initial_load_fraction)},
        "predicted_ink_state": {"ink_id": charge.ink_id or first_ink,
                                "load_fraction": max(0.0, remaining_fraction) if continuous else charge.frac},
        # Old scenario budgets remain metadata; they no longer stop a run.
        **({"operating_budget_s": float(operating_budget_s)} if operating_budget_s is not None else {}),
        # Likewise the paint planner, only when it is not the contract's
        # default: absent means concentric, and every existing program's
        # bytes are unchanged.
        **({"fill_style": fill_style} if fill_style != "concentric" else {}),
        "events": events,
        "total_material_path_length_m": sum(
            float(curve["rest_surface_arc_length_m"]) for _, curve in curves
        ),
        "uncertainty": {"length_sigma_m": 0.0, "load_sigma": 0.0},
        "provenance": deepcopy(provenance),
    }
    document["content_sha256"] = canonical_digest(document)
    return validate_contract(document, expected_schema=document["schema"],
                             expected_topology_sha256=body.topology_sha256 if body else None)


def exact_program_module():
    """Return the script-side exact planner without duplicating it here."""

    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[5]
    library = root / "scripts" / "lib"
    for name in ("ink_spec", "ink_program"):
        module_name = f"tatbot_exact_{name}"
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, library / f"{name}.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load exact {name} module")
            module = importlib.util.module_from_spec(spec)
            # ink_program imports ink_spec by its ordinary script name.
            if name == "ink_spec":
                sys.modules["ink_spec"] = module
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    return sys.modules["tatbot_exact_ink_program"]
