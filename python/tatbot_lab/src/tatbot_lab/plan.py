"""Proposed trials into runs: cells on the sheet, one program per run, and the manifest the readers measure by.

A run is what one `tatbot ros draw` draws: one cartridge, one depth trim, one speed (tatbot_lab.space). Runs go
cartridge by cartridge in the study's order, and within a cartridge from the lightest ink to the darkest, so black
comes last (2026-11-05: a 30 s rinse does not clear a heavy black run). Cells, one on each of the print's grid markers,
are taken in its fill order (tatbot_lab.tile), skipping those this sheet has already spent; trials that find no cell
stay proposed for the next sheet.

Every trial is its own pen and its own resource, its dip settings over the datasheet's when it sets them. Changing to
another resource of the same tool and ink costs nothing at the arm, and the compiler dips afresh at every resource, so
each cell starts on a fresh dip whatever the cell before it drew; a change of ink rinses. Each run directory holds one
artwork (`tatbot-lab-pattern/1`, the whole tile as its canvas, so the strokes keep their page millimetres, one layer a
trial, lightest ink first), the tatbot-inks/3 file binding each layer to its resource, the compiled program, and
`cells.json` (tatbot.lab-cells/1).
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tatbot_contracts import lab
from tatbot_contracts.artwork import freeze_artwork
from tatbot_contracts.canonical import canonical_digest
from tatbot_contracts.paths import freeze_program

from tatbot_lab import patterns, space, store, tile

ADAPTER = "tatbot-lab-pattern/1"
GENERATION_WIDTH_MM = 0.5   # the artwork's drawn width for previews only; the compile counts passes at the tool's
LIVE = ("planned", "drawn", "read")


@dataclass
class Run:
    key: tuple             # (cartridge, trim_mm, speed_mm_s)
    trials: list[dict]
    cells: list[tile.Cell]


def lightness(catalog: dict, ink: str) -> float:
    """Relative luminance of the ink's catalogue colour, 0 black to 1 white."""
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = catalog[ink]["rgb"]
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def used_cells(trials: dict[str, dict], sheet: str) -> set[str]:
    return {t["placement"]["cell"] for t in trials.values()
            if t.get("placement") and t["placement"]["sheet"] == sheet and t["status"] in LIVE}


def spent(study: dict, trials: dict[str, dict], sheet: str) -> set[str]:
    """The cells of `sheet` taken by this study's trials or any other study's on the same print: a sheet is one
    physical skin, so a study made for new bounds goes on where another left it (2026-11-09)."""
    used = used_cells(trials, sheet)
    for path in sorted(store.root().glob("*/study.json")):
        other = store.load(path.parent.name)
        if other["id"] != study["id"] and other["stencil"]["pattern_id"] == study["stencil"]["pattern_id"]:
            used |= used_cells(store.trials(other["id"]), sheet)
    return used


def runs(study: dict, proposed: list[dict], order: list[tile.Cell], catalog: dict) -> tuple[list[Run], list[dict]]:
    """The proposed trials grouped into runs in drawing order, each trial given the next free cell; the trials left
    without a cell."""
    for t in proposed:
        if len(t["inks"]) != 1:
            raise ValueError(f"trial {t['id']}: a cell holds one ink for now; layering inks is not built")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in proposed:
        groups[space.run_key(t)].append(t)
    rank = {c: i for i, c in enumerate(study["cartridges"])}

    def darkest(key):
        return min(lightness(catalog, t["inks"][0]) for t in groups[key])
    keys = sorted(groups, key=lambda k: (rank[k[0]], -darkest(k), -k[1], k[2]))
    free, out, left = list(order), [], []
    for key in keys:
        members = sorted(groups[key], key=lambda t: (-lightness(catalog, t["inks"][0]), t["id"]))
        taken = members[:len(free)]
        left += members[len(free):]
        if taken:
            out.append(Run(key, taken, free[:len(taken)]))
            free = free[len(taken):]
    return out, left


def _artwork(items: list[tuple[dict, tile.Cell]], tile_mm, catalog: dict, name: str) -> dict:
    """The run's strokes, one layer (pen) a trial named by its id, in drawing order."""
    w, h = tile_mm
    pens, layers = [], []
    for t, cell in items:
        elements = [{"id": f"{t['id']}-{cell.id}-{k + 1}", "kind": "path", "closed": False, "fill": False,
                     "width_m": GENERATION_WIDTH_MM / 1000.0,
                     "deposition": 1 if dep is None else [round(float(d), 4) for d in dep],
                     "points_m": [[round(float(x + w / 2) / 1000.0, 9), round(float(y + h / 2) / 1000.0, 9)]
                                  for x, y in pts]}
                    for k, (pts, dep) in enumerate(patterns.place(t["technique"], t["settings"], cell.box_mm))]
        pens.append({"id": t["id"], "color_srgb": [round(c / 255.0, 6) for c in catalog[t["inks"][0]]["rgb"]]})
        layers.append({"id": t["id"], "ink_id": t["id"], "elements": elements})
    geometry = {"canvas_m": {"width": w / 1000.0, "height": h / 1000.0}, "inks": pens, "layers": layers,
                "negative_space_masks": []}
    source = canonical_digest({"trials": [{k: t[k] for k in ("id", "technique", "settings")} for t, _ in items],
                               "cells": [c.id for _, c in items]})
    program, _ = freeze_program(geometry, source_sha256=source, name=name, adapter=ADAPTER)
    recipe = canonical_digest({"adapter": ADAPTER, "source_sha256": source})
    return freeze_artwork(program, name=name, source={"kind": "imported", "identifier": name, "license": None,
                                                      "attribution": None, "generation": None},
                          conversion={"adapter": ADAPTER, "recipe_sha256": recipe,
                                      "chord_error_m": 1e-9})   # exact, to the points' rounding


def _inks_file(study: dict, cartridge: str, trials: list[dict], artwork: dict) -> str:
    """One resource a trial: its ink in the study's cap, rinsed into in the water cap, its own dip values set."""
    lines = ["format: tatbot-inks", "version: 3", "substrate: silicon_skin", "resources:"]
    for t in trials:
        ink = t["inks"][0]
        lines += [f"- id: {t['id']}", f"  tool_id: {cartridge}", f"  ink_id: {ink}", "  pen_mode: ride",
                  f"  slot: {study['palette'][ink]}", "  activation:", "    method: rinse",
                  f"    slot: {study['palette']['water']}", "    ink_id: water"]
        if dip := space.dip_override(t["settings"]):
            lines += ["  dip:", *(f"    {key}: {value!r}" for key, value in dip.items())]
    lines.append("bindings:")
    for t in trials:
        lines += [f"- artwork_sha256: {artwork['content_sha256']}", f"  pen_id: {t['id']}", f"  resource_id: {t['id']}"]
    return "\n".join(lines) + "\n"


def write_run(study: dict, run: Run, p: tile.Print, sheet: str, plan_id: str, index: int, out: Path,
              catalog: dict, compile_program) -> dict:
    """One run's artworks, inks file, program and cells manifest; returns its summary."""
    run_id = f"run-{index:02d}"
    directory = out / run_id
    directory.mkdir(parents=True, exist_ok=True)
    items = list(zip(run.trials, run.cells, strict=True))   # runs() put them lightest ink first
    inks = list(dict.fromkeys(t["inks"][0] for t in run.trials))
    art = _artwork(items, p.tile_mm, catalog, f"{study['id']} {plan_id} {run_id}")
    art_path = directory / "artwork.json"
    art_path.write_text(json.dumps(art) + "\n")
    inks_path = directory / "inks.yaml"
    inks_path.write_text(_inks_file(study, run.key[0], run.trials, art))
    program = compile_program([art_path], inks_path, directory / "program", p.directory, run.key[2], study["arm"])
    stats, dips = program["stats"], dips_mm(program, run.cells)
    cells = {"schema": lab.CELLS, "study": study["id"], "plan": plan_id, "run": run_id, "sheet": sheet,
             "pattern_id": p.pattern_id,
             "cells": [{"id": cell.id, "trial": t["id"], "box_mm": list(cell.box_mm), "inks": t["inks"],
                        "markers": tile.covered(p, cell.box_mm), "dips_mm": dips[cell.id],
                        "strokes_mm": [[[round(float(x), 4), round(float(y), 4)] for x, y in pts]
                                       for pts, _ in patterns.place(t["technique"], t["settings"], cell.box_mm)]}
                       for t, cell in zip(run.trials, run.cells, strict=True)]}
    lab.cells(cells)
    (directory / "cells.json").write_text(json.dumps(cells, indent=1) + "\n")
    cartridge, trim, speed = run.key
    return {"run": run_id, "program_sha256": canonical_digest(program), "cartridge": cartridge, "trim_mm": trim,
            "speed_mm_s": speed, "inks": inks,
            "trials": [t["id"] for t in run.trials], "cells": [c.id for c in run.cells],
            "program": str(directory / "program" / "program.json"), "est_s": stats.get("est_s"),
            "dips": stats.get("dips"), "draw": f"tatbot ros draw {directory / 'program' / 'program.json'} "
                                             f"--arm {study['arm']} --trim-mm={trim:g}"}


def dips_mm(program: dict, cells: list[tile.Cell]) -> dict[str, list[float]]:
    """Where the compiled program dips, per cell: the millimetres of that cell's strokes drawn before each dip that
    comes before one of them. The compiler dips every mm_per_dip of the cartridge's datasheet, so a cell longer than
    that is drawn in several dip cycles, which its reading takes apart."""
    drawn = {c.id: 0.0 for c in cells}
    out: dict[str, list[float]] = {c.id: [] for c in cells}
    pending = False
    for op in program["ops"]:
        if op["op"] == "dip":
            pending = True
        elif op["op"] == "stroke":
            pts = np.asarray(op["points_m"], float)[:, :2] * 1000
            first = pts[0]
            cell = next(c for c in cells if c.box_mm[0] - 1e-6 <= first[0] <= c.box_mm[2] + 1e-6
                        and c.box_mm[1] - 1e-6 <= first[1] <= c.box_mm[3] + 1e-6)
            if pending:
                out[cell.id].append(round(drawn[cell.id], 2))
                pending = False
            drawn[cell.id] += float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return out


def compile_program(artworks: list[Path], inks: Path, out: Path, stencil: Path, speed_mm_s: float, arm: str) -> dict:
    """`tatbot ros compile`'s own function (tatbot_ink), in this process: program.json and preview.svg written, and
    the program returned."""
    from tatbot_ink import compile as compile_ink
    from tatbot_ink import write_preview, write_program

    # a trial may draw on dip values its cartridge borrowed: that is how the cartridge's own get measured. Its cells
    # are centred on markers and spent in tile.fill_order: the lab keeps its own marker budget
    program = compile_ink([str(a) for a in artworks], arm=arm, speed_m_s=speed_mm_s / 1000.0, inks_path=inks,
                          stencil=stencil, allow_borrowed_dip=True, allow_marker_loss=True)
    write_program(program, out / "program.json")
    write_preview(program, out / "preview.svg")
    return program


def plan(study: dict, trials: dict[str, dict], p: tile.Print, sheet: str, catalog: dict,
         compile_program=compile_program) -> dict:
    """Plan every proposed trial that fits this sheet; record them planned; write and return plan.json."""
    if p.pattern_id != study["stencil"]["pattern_id"]:
        raise ValueError(f"the print is {p.pattern_id}, the study's is {study['stencil']['pattern_id']}")
    lab._id(sheet, "sheet")
    proposed = [t for t in trials.values() if t["status"] == "proposed"]
    if not proposed:
        raise ValueError("no proposed trials to plan")
    sheet_cells = tile.cells(p, study["cell_mm"])
    for t in proposed:   # every pattern must fit a cell before anything is written
        patterns.place(t["technique"], t["settings"], sheet_cells[0].box_mm)
    order = tile.fill_order(sheet_cells, spent(study, trials, sheet))
    run_list, left = runs(study, proposed, order, catalog)
    if not run_list:
        raise ValueError(f"sheet {sheet} has no free cell left: plan onto a fresh sheet")
    plan_id = store.next_ids([d.name for d in store.plans_dir(study["id"]).glob("p*")], 1, "p")[0]
    out = store.plans_dir(study["id"]) / plan_id
    partial = out.with_name(f".{plan_id}-partial")
    shutil.rmtree(partial, ignore_errors=True)
    try:   # a plan that fails to compile leaves nothing behind and no trial moved
        summaries = [write_run(study, r, p, sheet, plan_id, i, partial, catalog, compile_program)
                     for i, r in enumerate(run_list, 1)]
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(out)
    for summary in summaries:
        summary["program"] = summary["program"].replace(str(partial), str(out))
        summary["draw"] = summary["draw"].replace(str(partial), str(out))
    updated = []
    for summary in summaries:
        for trial_id, cell_id in zip(summary["trials"], summary["cells"], strict=True):
            updated.append({**trials[trial_id], "status": "planned",
                            "placement": {"plan": plan_id, "run": summary["run"], "sheet": sheet, "cell": cell_id},
                            "run": {"program_sha256": summary["program_sha256"]}})
    store.append(study["id"], updated)
    result = {"study": study["id"], "plan": plan_id, "created": store.now(), "sheet": sheet,
              "pattern_id": p.pattern_id, "runs": summaries, "unplaced": [t["id"] for t in left],
              "est_s": sum(s["est_s"] or 0 for s in summaries)}
    (out / "plan.json").write_text(json.dumps(result, indent=1) + "\n")
    return result
