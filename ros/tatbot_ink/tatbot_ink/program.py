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

from tatbot_ink import dose, order, passes
from tatbot_ink.errors import CompileError
from tatbot_ink.input import read_input, read_layers
from tatbot_ink.place import MIN_STROKE_M, drop_dots, keep_region, placed_strokes, print_page, relocate
from tatbot_ink.resources import bind_resources, borrowed_dips, fitted_bindings
from tatbot_ink.schedule import line_per_dip, schedule_ops

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


def _provenance(design, name, raw):
    items = design["placements"]
    result = {"file": name, "sha256": hashlib.sha256(raw).hexdigest(), "name": design["name"],
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
            at_m=None, width_m: float | None = None, stencil=None, min_stroke_m: float = MIN_STROKE_M,
            mark_width_m: float | None = None, keep_region_m=None, max_stroke_m: float | None = None,
            after=(), wiped: bool = False, allow_borrowed_dip: bool = False, allow_marker_loss: bool = False) -> dict:
    """Prepare acquired artwork at its original metric size; --width is an assertion; `stencil` (a generated
    print's directory) places it on that print's page instead of the nominal 128 x 172 mm one. A path inside
    `min_stroke_m` that lies on its own ink's line is dropped (place.drop_dots). The pen's passes are counted at
    `mark_width_m`, else each tool's line width, else each path's generation width (passes.mark), and its pokes over
    each spot at the same width (stats.dose: tatbot_ink.dose, from the planner's time law and motion.yaml
    machine.rate_hz). `keep_region_m`, a
    polygon in page metres, keeps only the placed paths inside it (place.keep_region), their geometry as placed: a
    part of a design drawn again over its own. `max_stroke_m` draws a longer path as strokes of at most that length,
    each lifted (schedule_ops). `after` names the programs already drawn on this print, oldest first, and `wiped`
    says the print was wiped since the last of them: a lighter ink is refused over a darker one's mark, in this
    program or after those (tatbot_ink.order). A cartridge whose datasheet says its dip values are borrowed is
    refused unless `allow_borrowed_dip` (the lab's trials, which are how its card gets drawn; an operator's choice).
    On a fiducial-tile print (`stencil`) the markers the program's ink leaves readable are counted into
    stats.markers, over the ink of the `after` programs, and a program under the budget is refused
    (_marker_budget), unless `allow_marker_loss`.

    The input is acquired DBV3 Artwork/2: one file, or several (a list) drawn as
    layers in the order given, centred on the page or at `at_m`. Source order,
    direction, pen IDs and repeats survive rigid placement and timed chunking.
    Computational chunks continue in contact and do not imply dips.
    """
    if not (math.isfinite(speed_m_s) and speed_m_s > 0 and math.isfinite(max_segment_s) and max_segment_s > 0):
        raise CompileError("the draw speed and max_segment_s must be positive")
    if not (math.isfinite(min_stroke_m) and min_stroke_m >= 0):
        raise CompileError("min_stroke_m must be a finite size, 0 to keep every path")
    if mark_width_m is not None and not (math.isfinite(mark_width_m) and 0 < mark_width_m <= .02):
        raise CompileError("mark_width_m must be a positive width up to 20 mm")
    if max_stroke_m is not None and not (math.isfinite(max_stroke_m) and max_stroke_m >= .001):
        raise CompileError("max_stroke_m must be a finite length of 1 mm or more")
    paths = [Path(p) for p in design_path] if isinstance(design_path, (list, tuple)) else [Path(design_path)]
    root = repo_root(repo)
    raws = [p.read_bytes() for p in paths]
    raw = b"".join(raws)
    try:
        read = read_input(parse_json(raw)) if len(paths) == 1 else read_layers([parse_json(r) for r in raws])
        design = relocate(read, at_m=at_m, width_m=width_m)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise CompileError(f"{'+'.join(p.name for p in paths)}: {error}") from error
    motion = load_motion(root / "ros/tatbot_motion/config/motion.yaml")
    resources, bindings, binding_source = (bind_resources(inks_path, design, repo=root, arm=arm, speed_m_s=speed_m_s)
                                          if inks_path else fitted_bindings(design, repo=root, arm=arm, tool_id=tool_id, pen_mode=motion['pen']['mode']))
    borrowed_dips(resources, allowed=allow_borrowed_dip)
    table = {resource['id']: resource for resource in resources}
    widths = {key: table[value]['tool']['line_width_m'] for key, value in bindings.items()}
    strokes, page = placed_strokes(design, tool_widths=widths, page=None if stencil is None else print_page(stencil, root))
    strokes, dots, alone = drop_dots(strokes, lambda s: widths[s.src['artwork_sha256'], s.src['pen']] or s.generation_width_m,
                                     lambda s: bindings[s.src['artwork_sha256'], s.src['pen']], min_stroke_m)
    region = None
    if keep_region_m is not None:   # after the dots: one on a line of the whole design is ink the skin already has
        strokes, region = keep_region(strokes, keep_region_m, min_size_m=min_stroke_m)
    strokes, stroke_order = _ordered(strokes, bindings, widths, stencil, root)
    ops = schedule_ops(strokes, bindings, resources, speed=speed_m_s, max_seconds=max_segment_s, motion=motion,
                      max_stroke_m=max_stroke_m)
    contact = sum(polyline_length(stroke.points_m) for stroke in strokes)
    travel = sum(float(np.linalg.norm(b.points_m[0] - a.points_m[-1])) for a, b in zip(strokes, strokes[1:], strict=False))
    page_record = {k: page[k] for k in ("kind", "size_m", "clear_m")}
    doses = dose.strokes({"ops": ops, "resources": resources, "page": page_record, "draw_speed_m_s": speed_m_s}, motion)
    timing = estimate_duration(ops, speed_m_s, travel, motion, resources=resources,
                               drawing_s={op_id: d.drawing_s for op_id, d in doses.items()})
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
                   "generation_widths_m": sorted({s.generation_width_m for s in strokes}), "ink_binding": binding_source,
                   "implementation": _implementation(), "min_stroke_m": min_stroke_m, "max_stroke_m": max_stroke_m,
                   "dropped_dots": [{k: s.src[k] for k in ("placement", "layer", "path")} for s in dots],
                   **({} if region is None else {"keep_region": region}),
                   **({} if stroke_order == "source" else {"stroke_order": stroke_order}),
                   "pen_bindings": [{"artwork_sha256": a, "pen_id": p, "resource_id": resource}
                                    for (a, p), resource in bindings.items()]}
    result = {"format": FORMAT, "version": VERSION, "design": _provenance(design, "+".join(p.name for p in paths), raw),
            "arm": arm, "substrate": binding_source['substrate'], "page": page_record, "resources": resources,
            "draw_speed_m_s": float(speed_m_s), "preparation": preparation, "ops": ops,
            "stats": {"strokes": sum(op["op"] == "stroke" for op in ops), "paths": len(strokes),
                      "contact_m": round(contact, 9), "travel_m": round(travel, 9), "dips": sum(op['op'] == 'dip' for op in ops),
                      "tool_changes": sum(op['op'] == 'tool_change' for op in ops), "est_s": round(timing["modeled_s"], 1),
                      "time_estimate": timing, "notes": notes,
                      "dots": {"dropped": len(dots), "dropped_contact_m": round(sum(polyline_length(s.points_m) for s in dots), 9),
                               "kept_alone": alone}}}
    result["stats"]["passes"] = {"marks": {r['id']: passes.mark(r['tool'], mark_width_m) for r in resources}}
    result["stats"]["passes"] = {**passes.summary(passes.program_lines(result)), **result["stats"]["passes"]}
    _ink_order(result, after, wiped)
    result["stats"]["line_per_dip"] = line_per_dip(ops, resources)
    result["stats"]["dose"] = dose.summary(result, motion, doses)
    _marker_budget(result, stencil, root, [prior for prior, _ in _earlier(after, wiped)], allow_marker_loss)
    validate_for_execution(result)
    return result


def _ink_order(program: dict, after, wiped: bool) -> None:
    """stats.ink_order, or the refusal (tatbot_ink.order); preparation.drawn_after names the earlier programs."""
    program["stats"]["ink_order"] = order.check(program, _earlier(after, wiped))
    if after:
        program["preparation"]["drawn_after"] = [
            {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "wiped": wiped and index == len(after) - 1}
            for index, path in enumerate(after)]


def _earlier(after, wiped: bool) -> list:
    """The programs drawn before this one on its print, each with whether the print was wiped after it: `wiped`
    is the last one's, since a wipe takes off what every run before it left wet."""
    earlier = []
    for index, path in enumerate(after):
        try:
            prior = parse_json(Path(path).read_bytes())
            if prior.get("format") != FORMAT or not isinstance(prior.get("ops"), list):
                raise ValueError(f"not a {FORMAT}")
            order.runs(prior)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise CompileError(f"--after {path}: {error}") from error
        earlier.append((prior, wiped and index == len(after) - 1))
    if wiped and not earlier:
        raise CompileError("--wiped says the print was wiped after an earlier run: name its program with --after")
    return earlier


def _ordered(strokes, bindings, widths, stencil, root):
    """(the placed paths in the order they are drawn, that order's name): source order, or, on a print with
    markers and stack.yaml page.markers.order `spare`, each ink's paths ordered to spare the markers
    (markers.spare_order)."""
    from tatbot_ink import markers

    config = None if stencil is None else markers.settings(root)
    if config is None or config["order"] == "source":
        return strokes, "source"
    if config["order"] != "spare":
        raise CompileError(f"stack.yaml page.markers.order is source or spare, not {config['order']!r}")

    def key(stroke):
        return stroke.src["artwork_sha256"], stroke.src["pen"]
    return markers.spare_order(strokes, lambda s: bindings[key(s)], lambda s: widths[key(s)] or s.generation_width_m,
                               stencil, root, config), "spare"


def _marker_budget(program: dict, stencil, root, earlier: list, allow: bool) -> None:
    """stats.markers: what the program's ink leaves of the print's markers for the cameras (markers.predict), over
    the ink of the programs drawn there before (`earlier`, wiped since or not: a wipe takes off wet ink, not a
    line). Under stack.yaml's page.markers budget, with page.markers.enforce, it is refused, each line naming the
    stage and the markers lost, unless `allow` (recorded as preparation.allow_marker_loss, which the draw
    honours): a sheet fixed on its mount is drawn without them
    (page.static_run). Not enforced, the lines go into stats.notes. Nothing is counted without a print (`stencil`
    None) or on a print without markers. Where the count cannot be made (a Python without OpenCV, a layout record
    the detector cannot load) a program the budget would not refuse anyway is prepared with a note saying so."""
    from tatbot_ink import markers

    if stencil is None:
        return
    config = markers.settings(root)
    try:
        report = markers.predict(program, stencil, root, config, drawn=earlier)
    except (ImportError, OSError, KeyError, ValueError) as error:
        why = "this Python has no OpenCV" if isinstance(error, ImportError) else f"the print's markers did not load ({error!r})"
        if config["enforce"] and not allow:
            raise CompileError(f"the marker budget could not be counted: {why}") from error
        program["stats"]["notes"].append(f"marker budget not counted: {why}")
        return
    if report is None:
        return
    lost = markers.refusals(report)
    program["stats"]["markers"] = {**report, "within_budget": not lost}
    if lost and not allow and not config["enforce"]:
        program["stats"]["notes"] += [f"marker budget (not enforced, stack.yaml page.markers.enforce): {line}" for line in lost]
    elif lost and not allow:
        raise CompileError("the design covers the markers its print is tracked by: " + "; ".join(lost)
                           + ". Draw it smaller or off the markers it covers, or pass --allow-marker-loss to draw "
                             "it anyway (a sheet fixed on its mount: stack.yaml page.static_run)")
    if allow:
        program["preparation"]["allow_marker_loss"] = True


def write_program(program: dict, path: str | Path) -> Path:
    """program.json: the header indented, one op per line, keys in README order. Read back as the draw reads it
    (parse_json) first, so a program the draw would refuse is never written."""
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
    text = "\n".join(lines) + "\n"
    parse_json(text)
    path.write_text(text)
    return path
