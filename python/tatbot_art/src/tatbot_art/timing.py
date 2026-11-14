"""Recorded compiler time plus explicit workflow allowances; no planner or robot access."""
from __future__ import annotations

import math
from collections import Counter

# The dip is a historical full-piece median; rinse and swap are planning allowances.
# None is a measurement of the currently running rig.
GAPS_S = {"dip": 48.0, "rinse": 40.0, "swap": 180.0}
GAP_BASIS = {"dip": "historical dip-gap median", "rinse": "fixed allowance", "swap": "fixed allowance"}


def seconds(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def workflow(program: dict) -> list[dict]:
    """Initial selections and the same tool/ink cost no rinse or swap, as in the executor.

    A different resource can change a ride height while keeping its cartridge and ink.
    Resume/recovery work is outside this uninterrupted-program estimate.
    """
    table = {r["id"]: r for r in program["resources"]}
    current, steps = None, []
    for op in program["ops"]:
        if op["op"] == "stroke":
            continue
        resource = table[op["resource_id"]]
        kind = op["op"]
        if kind == "tool_change":
            identity = ((resource.get("tool") or {}).get("id"), resource["ink_id"])
            same = identity[0] is not None and identity == current
            activation = resource.get("activation") or op.get("activation") or {}
            kind = "selection" if op.get("initial") or same else (
                "rinse" if activation.get("method") == "rinse" else "swap")
            current = identity
        steps.append({"op": op.get("id"), "resource_id": resource["id"], "kind": kind})
    return steps


def _compiler_components(program: dict) -> tuple[dict, dict, float]:
    stats = program.get("stats") or {}
    estimate = stats.get("time_estimate") or {}
    modeled = seconds(estimate.get("modeled_s", stats.get("est_s", 0)), "compiler modeled seconds")
    components = {k: seconds(v, f"compiler {k}") for k, v in (estimate.get("components_s") or {}).items()}
    remainder = modeled - sum(components.values())
    if remainder < -0.001:
        raise ValueError("compiler components exceed its modeled seconds")
    if remainder > 0.001 or not components:
        components["compiler_subtotal"] = max(0, remainder)
    return estimate, components, modeled


def minutes(program: dict, gaps_s: dict | None = None) -> dict:
    """One accounting path for renders and profiles. Never add a workflow cost twice.

    Compiler-provided dip/pen_change components take precedence over fixed gaps. Initial
    and same-tool/same-ink selections are free when adding the missing workflow costs.
    Programs used only as render canvases may have no timing; the profile requires it.
    """
    given = gaps_s or {}
    if set(given) - GAPS_S.keys():
        raise ValueError("gap overrides name dip, rinse or swap")
    gaps = {k: seconds(v, f"{k} gap") for k, v in (GAPS_S | given).items()}
    estimate, components, modeled = _compiler_components(program)
    basis = dict.fromkeys(components, "recorded compiler estimate")
    counts = Counter(step["kind"] for step in workflow(program))
    for kind in GAPS_S:
        compiled = "dip" if kind == "dip" else "pen_change"
        if compiled not in components:
            components[kind] = counts[kind] * gaps[kind]
            basis[kind] = "user-supplied allowance" if kind in given else GAP_BASIS[kind]
    unknown = {k: v for k, v in estimate.get("unknown_operations", {}).items() if k not in ("dip", "pen_change")}
    total = sum(components.values())
    return {"modeled_s": round(modeled, 4), "dips": counts["dip"], "rinses": counts["rinse"],
            "swaps": counts["swap"], "selections": counts["selection"], "gaps_s": gaps,
            "estimated_s": round(total, 4), "minutes": round(total / 60, 1),
            "components_s": {k: round(v, 4) for k, v in components.items()}, "basis": basis,
            "unknown_operations": unknown, "complete": not unknown}
