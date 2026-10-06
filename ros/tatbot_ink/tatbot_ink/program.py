"""Frozen DBV3 artwork -> placed ROS program. No path synthesis, sorting or SVG reader."""
from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
from tatbot_contracts.canonical import canonical_digest, parse_json
from tatbot_contracts.ros_program import FORMAT, VERSION, validate_for_execution
from tatbot_description import repo_root as repo_root
from tatbot_motion import load_motion
from tatbot_motion.estimate import estimate_duration
from tatbot_motion.timelaw import polyline_length

from tatbot_ink.errors import CompileError
from tatbot_ink.input import read_input
from tatbot_ink.place import placed_strokes, print_page, relocate
from tatbot_ink.resources import bind_resources, fitted_bindings
from tatbot_ink.schedule import schedule_ops

DEFAULT_SPEED_M_S = 0.0035
MAX_SEGMENT_S = 60.0


def _implementation():
    import tatbot_contracts
    import tatbot_motion

    sources = {}
    for name, folder in (("tatbot_ink", Path(__file__).parent),
                         ("tatbot_motion", Path(tatbot_motion.__file__).parent),
                         ("tatbot_contracts", Path(tatbot_contracts.__file__).parent)):
        for path in sorted(folder.glob("*.py")):
            sources[f"{name}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sources_sha256": canonical_digest(sources), "python": platform.python_version(), "numpy": np.__version__}


def _provenance(design, path, raw):
    items = design["placements"]
    result = {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "name": design["name"],
              "placements": [{"id": item["id"], "artwork_sha256": design["artworks"][item["artwork_id"]]["content_sha256"],
                              "recipe_sha256": design["artworks"][item["artwork_id"]]["conversion"]["recipe_sha256"],
                              "source_sha256": design["artworks"][item["artwork_id"]]["source_sha256"],
                              "size_m": item["placement"]["physical_scale_m"],
                              "at_m": item["placement"]["target"]["anchor_uv_m"],
                              "rotation_rad": item["placement"]["rotation_rad"],
                              "mirrored": item["placement"]["mirrored"]} for item in items]}
    if len(items) == 1:
        result.update(at_m=result["placements"][0]["at_m"], width_m=result["placements"][0]["size_m"][0])
    return result


def compile(design_path: str | Path, *, arm: str = "right", tool_id: str | None = None,
            repo: str | Path | None = None, speed_m_s: float = DEFAULT_SPEED_M_S,
            inks_path: str | Path | None = None, max_segment_s: float = MAX_SEGMENT_S,
            at_m=None, width_m: float | None = None, stencil=None) -> dict:
    """Prepare acquired artwork at its original metric size; --width is an assertion; `stencil` (a generated
    print's directory) places it on that print's page instead of the nominal 100 x 150 mm one.

    The input is Artwork/2 or Inkmap Design/1 with acquired DBV3 paths. Source
    order, direction, pen IDs and repeats survive rigid placement and timed
    chunking. Computational chunks continue in contact and do not imply dips.
    """
    if not (math.isfinite(speed_m_s) and speed_m_s > 0 and math.isfinite(max_segment_s) and max_segment_s > 0):
        raise CompileError("the draw speed and max_segment_s must be positive")
    root, path = repo_root(repo), Path(design_path)
    raw = path.read_bytes()
    try:
        design = relocate(read_input(parse_json(raw)), at_m=at_m, width_m=width_m)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise CompileError(f"{path.name}: {error}") from error
    motion = load_motion(root / "ros/tatbot_motion/config/motion.yaml")
    resources, bindings, binding_source = (bind_resources(inks_path, design, repo=root, arm=arm, speed_m_s=speed_m_s)
                                          if inks_path else fitted_bindings(design, repo=root, arm=arm, tool_id=tool_id, pen_mode=motion['pen']['mode']))
    table = {resource['id']: resource for resource in resources}
    strokes, page = placed_strokes(design, tool_widths={key: table[value]['tool']['line_width_m'] for key, value in bindings.items()},
                                   page=None if stencil is None else print_page(stencil, root))
    ops = schedule_ops(strokes, bindings, resources, speed=speed_m_s, max_seconds=max_segment_s, motion=motion)
    contact = sum(polyline_length(stroke.points_m) for stroke in strokes)
    travel = sum(float(np.linalg.norm(b.points_m[0] - a.points_m[-1])) for a, b in zip(strokes, strokes[1:], strict=False))
    timing = estimate_duration(ops, speed_m_s, travel, motion, resources=resources)
    widths = sorted({s.generation_width_m for s in strokes})
    notes = []
    for resource in resources:
        tool = resource['tool']
        if tool['line_width_status'] != 'measured':
            notes.append(f"{resource['id']}: physical width is {tool['line_width_status']}; preview may use generation width")
        if tool['line_width_m'] is not None and any(abs(stroke.generation_width_m-tool['line_width_m']) > 1e-10 for stroke in strokes
                                                  if bindings[stroke.src['artwork_sha256'], stroke.src['pen']] == resource['id']):
            notes.append(f"{resource['id']}: generation widths differ from the physical profile; geometry is preserved")
    preparation = {"schema": "tatbot-preparation/2", "adapter": "dbv3-paths-to-ros/2",
                   "max_chunk_s": max_segment_s, "motion_sha256": timing["motion_sha256"],
                   "generation_widths_m": widths, "ink_binding": binding_source, "implementation": _implementation(),
                   "pen_bindings": [{"artwork_sha256": a, "pen_id": p, "resource_id": resource}
                                    for (a, p), resource in bindings.items()]}
    result = {"format": FORMAT, "version": VERSION, "design": _provenance(design, path, raw),
            "arm": arm, "substrate": binding_source['substrate'], "page": {k: page[k] for k in ("kind", "size_m", "clear_m")}, "resources": resources,
            "draw_speed_m_s": float(speed_m_s), "preparation": preparation, "ops": ops,
            "stats": {"strokes": sum(op["op"] == "stroke" for op in ops), "paths": len(strokes),
                      "contact_m": round(contact, 9), "travel_m": round(travel, 9), "dips": sum(op['op'] == 'dip' for op in ops),
                      "tool_changes": sum(op['op'] == 'tool_change' for op in ops), "est_s": round(timing["modeled_s"], 1),
                      "time_estimate": timing, "notes": notes}}
    validate_for_execution(result)
    return result


def write_program(program: dict, path: str | Path) -> Path:
    """program.json: the header indented, one op per line, keys in README order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["{"]
    keys = list(program)
    for position, key in enumerate(keys):
        comma = "," if position < len(keys) - 1 else ""
        if key == "ops":
            body = ",\n".join(f"    {json.dumps(op, separators=(', ', ': '))}" for op in program["ops"])
            lines.append(f'  "ops": [\n{body}\n  ]{comma}' if body else f'  "ops": []{comma}')
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(program[key], separators=(', ', ': '))}{comma}")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")
    return path
