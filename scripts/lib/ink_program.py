"""Exact InkProgram intent arithmetic shared by offline compilers.

This module deliberately owns no geometry, robot, or session state.  It maps
the durable ``tatbot.ink-program/1`` events onto the existing ``ink_spec``
charge model and resolves caps only when an execution session is bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ink_spec

TOOL_CLASS_BY_ID = {
    "lutin-3rl-bugpin": "tattoo-needle",
    "lutin-ballpoint-dot": "ballpoint",
    "picosecond-laser-pen": "non-contact-laser",
}

# One reviewable mapping table between established runtime names and the new
# contract.  Values name the authority that remains unchanged.
FIELD_MAPPING = (
    ("InkPolicy.mode", "tool.class", "tool datasheet"),
    ("StrokeNeed.contact_mm", "stroke.curve.rest_surface_arc_length_m", "InkProgram"),
    ("StrokeNeed.contact_s", "stroke.speed_m_s", "InkProgram"),
    ("StrokeNeed.ink_id", "active ink state", "InkProgram event order"),
    ("DipPlan.before_stroke", "dip dependency/event position", "InkProgram"),
    ("DipPlan.reason", "dip.trigger_reason", "ink_spec.plan_dips"),
    ("DipPlan.charge_after_ul", "dip.expected_load_after", "ink_spec.Charge"),
    ("DipPlan.slot_id", "palette.resolved_caps.slot", "ExecutionProgram"),
    ("SlotLoad.fill_ul", "palette load snapshot", "ExecutionProgram"),
    ("dip marker", "samples sample range", "tatbot.draw-samples/1"),
)


class InkProgramRefusal(ValueError):  # noqa: N818 - named for the contract refusal taxonomy
    """A structured, source-bound lowering refusal."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        event_index: int | None = None,
        input_hashes: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.event_index = event_index
        self.input_hashes = dict(input_hashes or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "refused",
            "code": self.code,
            "detail": self.detail,
            "event_index": self.event_index,
            "input_hashes": self.input_hashes,
        }


@dataclass(frozen=True)
class ProgramStroke:
    event_index: int
    stroke_index: int
    ink_id: str
    need: ink_spec.StrokeNeed


@dataclass(frozen=True)
class ResolvedDip:
    event_index: int
    before_stroke: int
    slot_id: str
    ink_id: str
    reason: str
    charge_before_ul: float
    charge_after_ul: float
    cap_fill_ul: float
    why_slot: str


def tool_class(tool: Any) -> str:
    """Return the stable intent class for one existing tool datasheet."""

    tool_id = getattr(tool, "tool_id", tool)
    try:
        return TOOL_CLASS_BY_ID[str(tool_id)]
    except KeyError as exc:
        raise InkProgramRefusal("tool_class_unsupported", f"no InkProgram class for {tool_id!r}") from exc


def _hashes(program: dict[str, Any]) -> dict[str, str]:
    result = {}
    if isinstance(program.get("content_sha256"), str):
        result["ink_program_sha256"] = program["content_sha256"]
    for name in ("tattoo_program_sha256", "surface_placement_sha256", "compiler_sha256"):
        if isinstance(program.get(name), str):
            result[name] = program[name]
    return result


def program_strokes(program: dict[str, Any]) -> list[ProgramStroke]:
    """Derive exact ``StrokeNeed`` records from ordered durable events.

    Ink identity is a state machine: it starts at ``initial_ink_state`` and a
    dip changes it.  This makes multi-ink intent explicit without adding a cap
    or other session fact to a stroke.
    """

    if program.get("schema") not in {"tatbot.ink-program/1", "tatbot.ink-program/2"}:
        raise InkProgramRefusal("ink_program_invalid", "expected versioned InkProgram", input_hashes=_hashes(program))
    state = program.get("initial_ink_state") or {}
    current_ink = state.get("ink_id")
    if not isinstance(current_ink, str) or not current_ink:
        raise InkProgramRefusal("ink_program_invalid", "initial ink ID is absent", input_hashes=_hashes(program))
    strokes: list[ProgramStroke] = []
    for event_index, event in enumerate(program.get("events") or []):
        kind = event.get("kind") if isinstance(event, dict) else None
        if kind == "dip":
            current_ink = event.get("ink_id")
            if not isinstance(current_ink, str) or not current_ink:
                raise InkProgramRefusal(
                    "ink_program_invalid",
                    "dip has no ink ID",
                    event_index=event_index,
                    input_hashes=_hashes(program),
                )
        elif kind == "stroke":
            curve = event.get("curve") or {}
            length_m = float(curve.get("rest_surface_arc_length_m", 0.0))
            speed_m_s = float(event.get("speed_m_s", 0.0))
            if length_m <= 0.0 or speed_m_s <= 0.0:
                raise InkProgramRefusal(
                    "ink_program_invalid",
                    "stroke length and speed must be positive",
                    event_index=event_index,
                    input_hashes=_hashes(program),
                )
            need = ink_spec.StrokeNeed(length_m * 1000.0, length_m / speed_m_s, current_ink)
            strokes.append(ProgramStroke(event_index, len(strokes), current_ink, need))
    if not strokes:
        raise InkProgramRefusal("ink_program_invalid", "program has no stroke", input_hashes=_hashes(program))
    return strokes


def intended_dips(program: dict[str, Any]) -> list[tuple[int, int, dict[str, Any]]]:
    """Return ``(event index, next stroke index, event)`` in program order."""

    result = []
    stroke_index = 0
    for event_index, event in enumerate(program.get("events") or []):
        if not isinstance(event, dict):
            continue
        if event.get("kind") == "dip":
            result.append((event_index, stroke_index, event))
        elif event.get("kind") == "stroke":
            stroke_index += 1
    return result


def resolve_dips(
    program: dict[str, Any],
    policy: ink_spec.InkPolicy,
    palette: dict[str, ink_spec.PaletteSlot],
    load: dict[str, ink_spec.SlotLoad],
    *,
    arm: str = "right",
    tool_id: str = "tool",
    inks: dict[str, ink_spec.Ink] | None = None,
) -> list[ResolvedDip]:
    """Resolve cap slots while proving intent equals ``ink_spec.plan_dips``."""

    strokes = program_strokes(program)
    if policy.mode == "none":
        raise InkProgramRefusal(
            "ink_supply_unavailable",
            f"{tool_id!r} has ink.mode none",
            input_hashes=_hashes(program),
        )
    initial = program["initial_ink_state"]
    initial_charge = float(initial["load_fraction"]) * policy.charge_capacity_ul
    for stroke in strokes:
        required = policy.stroke_ul(stroke.need.contact_mm, stroke.need.contact_s)
        if policy.dips and required > policy.uptake_ul + 1e-12:
            raise InkProgramRefusal(
                "stroke_over_capacity",
                f"stroke {stroke.stroke_index} needs {required:.6g} uL; one uptake is {policy.uptake_ul:.6g} uL",
                event_index=stroke.event_index,
                input_hashes=_hashes(program),
            )
    try:
        planned = ink_spec.plan_dips(
            [stroke.need for stroke in strokes],
            policy,
            palette,
            load,
            arm=arm,
            initial_charge_ul=initial_charge,
            initial_ink=initial["ink_id"],
            tool_id=tool_id,
            inks=inks,
        )
    except ink_spec.InkSupplyError as exc:
        raise InkProgramRefusal(
            "ink_supply_unavailable",
            str(exc),
            input_hashes=_hashes(program),
        ) from exc
    intent = intended_dips(program)
    if len(planned) != len(intent):
        raise InkProgramRefusal(
            "dip_intent_mismatch",
            f"InkProgram declares {len(intent)} dips; exact charge model requires {len(planned)}",
            input_hashes=_hashes(program),
        )
    output: list[ResolvedDip] = []
    for plan, (event_index, before_stroke, event) in zip(planned, intent, strict=True):
        expected_load = 0.0 if policy.charge_capacity_ul <= 0 else plan.charge_after_ul / policy.charge_capacity_ul
        mismatch = []
        if plan.before_stroke != before_stroke:
            mismatch.append(f"position {before_stroke} != {plan.before_stroke}")
        if event.get("trigger_reason") != plan.reason:
            mismatch.append(f"reason {event.get('trigger_reason')!r} != {plan.reason!r}")
        if event.get("ink_id") != (plan.ink_id or strokes[plan.before_stroke].ink_id):
            mismatch.append("ink ID differs")
        if abs(float(event.get("expected_load_after", -1.0)) - expected_load) > 1e-9:
            mismatch.append("expected load differs")
        target = event.get("target_load") or []
        if len(target) != 2 or not float(target[0]) - 1e-12 <= expected_load <= float(target[1]) + 1e-12:
            mismatch.append("expected load is outside target envelope")
        if mismatch:
            raise InkProgramRefusal(
                "dip_intent_mismatch",
                "; ".join(mismatch),
                event_index=event_index,
                input_hashes=_hashes(program),
            )
        output.append(
            ResolvedDip(
                event_index=event_index,
                before_stroke=plan.before_stroke,
                slot_id=plan.slot_id,
                ink_id=plan.ink_id or strokes[plan.before_stroke].ink_id,
                reason=plan.reason,
                charge_before_ul=plan.charge_before_ul,
                charge_after_ul=plan.charge_after_ul,
                cap_fill_ul=plan.cap_fill_ul,
                why_slot=plan.why_slot,
            )
        )
    return output


def expected_ledger_events(
    program: dict[str, Any],
    policy: ink_spec.InkPolicy,
    dips: list[ResolvedDip],
) -> list[dict[str, Any]]:
    """Return deterministic ledger payloads; timestamps and IDs remain runtime facts."""

    strokes = program_strokes(program)
    dips_by_stroke = {dip.before_stroke: dip for dip in dips}
    output: list[dict[str, Any]] = []
    for stroke in strokes:
        dip = dips_by_stroke.get(stroke.stroke_index)
        if dip is not None:
            output.append(
                {
                    "kind": "dip",
                    "mode": policy.mode,
                    "slot": dip.slot_id,
                    "ink_id": dip.ink_id,
                    "reason": dip.reason,
                    "charge_before": dip.charge_before_ul,
                    "charge_after": dip.charge_after_ul,
                    "uptake_ul": dip.charge_after_ul - dip.charge_before_ul,
                }
            )
        output.append(
            {
                "kind": "stroke",
                "mode": policy.mode,
                "ink_id": stroke.ink_id,
                "contact_mm": stroke.need.contact_mm,
                "contact_s": stroke.need.contact_s,
                "ul": policy.stroke_ul(stroke.need.contact_mm, stroke.need.contact_s),
                "ink_event_index": stroke.event_index,
            }
        )
    return output


def metrics(
    program: dict[str, Any],
    policy: ink_spec.InkPolicy,
    dips: list[ResolvedDip],
    *,
    pen_up_travel_m: float = 0.0,
    duration_s: float | None = None,
) -> dict[str, Any]:
    strokes = program_strokes(program)
    contact_s = sum(stroke.need.contact_s for stroke in strokes)
    predicted = sum(policy.stroke_ul(stroke.need.contact_mm, stroke.need.contact_s) for stroke in strokes)
    return {
        "stroke_count": len(strokes),
        "surface_path_m": sum(stroke.need.contact_mm for stroke in strokes) / 1000.0,
        "pen_up_travel_m": float(pen_up_travel_m),
        "contact_duration_s": contact_s,
        "duration_s": float(duration_s if duration_s is not None else contact_s),
        "predicted_ink_use_ul": predicted,
        "dip_count": len(dips),
        "dip_efficiency_m_per_dip": (
            sum(stroke.need.contact_mm for stroke in strokes) / 1000.0 / len(dips) if dips else None
        ),
    }
