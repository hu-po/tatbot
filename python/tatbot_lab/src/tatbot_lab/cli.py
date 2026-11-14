"""`tatbot lab`: studies, trials, plans and cell readings (docs/lab.md). No robot here: a plan's
programs are drawn by `tatbot ros draw`, and `read` takes the frames a draw's before and after views left."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from tatbot_contracts import lab

from tatbot_lab import plan, propose, search, space, store, tile

REPO = Path(__file__).resolve().parents[4]
CONSTANTS = {"battery_v": 6.0, "machine_hz": 51.0, "stroke_mm": 4.0}   # 2026-11-08: held for now
OVERHEAD = ("cell", "trial", "views", "markers", "ecc", "noise_dL", "dL_mean", "da_light", "db_light")


def _catalog() -> dict:
    return yaml.safe_load((REPO / "config" / "inks.yaml").read_text())


def _pairs(text: str, where: str) -> dict[str, str]:
    out = {}
    for item in filter(None, text.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"{where}: expected name=value, got {item!r}")
        out[key.strip()] = value.strip()
    return out


def cmd_new(ns) -> int:
    p = tile.load(ns.stencil)
    bounds = {}
    for item in ns.bound:
        name, sep, span = item.partition("=")
        low, sep2, high = span.partition(":")
        if not (sep and sep2):
            raise ValueError(f"--bound {item!r}: expected name=low:high")
        bounds[name] = [float(low), float(high)]
    constants = {**CONSTANTS, **{k: float(v) for k, v in _pairs(ns.constants or "", "--constants").items()}}
    if space.DOSE in bounds:   # the dose is the speed's bound, through the machine rate the study holds
        if "speed_mm_s" in bounds:
            raise ValueError(f"--bound {space.DOSE} stands for speed_mm_s: give one of them")
        bounds["speed_mm_s"] = space.speed_bounds(bounds.pop(space.DOSE), constants["machine_hz"])
        print(f"lab new: {space.DOSE} at {constants['machine_hz']:g} Hz is speed_mm_s "
              f"{bounds['speed_mm_s'][0]:g}:{bounds['speed_mm_s'][1]:g}")
    record = {"schema": lab.STUDY, "id": ns.id, "created": store.now(), "question": ns.question, "arm": ns.arm,
              "stencil": {"pattern_id": p.pattern_id, "layout": p.layout, "tile_mm": list(p.tile_mm)},
              "cell_mm": ns.cell_mm, "constants": constants, "cartridges": ns.cartridges.split(","),
              "inks": ns.inks.split(","), "palette": _pairs(ns.palette, "--palette"),
              "techniques": ns.techniques.split(","), "bounds": bounds}
    cells = tile.cells(p, ns.cell_mm)
    directory = store.create(record)
    print(f"lab new: {ns.id} on {p.layout} ({p.pattern_id[:16]}), {len(cells)} cells of {ns.cell_mm:g} mm a sheet "
          f"-> {directory}")
    return 0


def cmd_propose(ns) -> int:
    study = store.load(ns.study)
    if sum(map(bool, (ns.file, ns.card, ns.dip_card, ns.search))) != 1:
        raise ValueError("propose takes one of --file PROPOSALS.jsonl, --card CARTRIDGE, --dip-card CARTRIDGE or "
                         "--search N")
    if ns.card:
        records = propose.add_card(study, ns.card, (ns.inks or ",".join(study["inks"])).split(","))
    elif ns.dip_card:
        records = propose.add_dip_card(study, ns.dip_card, (ns.inks or study["inks"][0]).split(",")[0])
    elif ns.search:
        if not ns.technique:
            raise ValueError("--search needs --technique: each technique's marks are their own archive")
        cartridge, ink = ns.cartridge or study["cartridges"][0], ns.ink or study["inks"][0]
        records = propose.add(study, search.batch(study, store.trials(ns.study), ns.technique, cartridge, ink,
                                                  ns.search, seed=ns.seed, max_runs=ns.max_runs))
        for r in records:
            print(f"  {r['id']} {r['settings']}: {r['hypothesis']}")
    else:
        text = Path(ns.file).read_text()
        proposals = [json.loads(line) for line in text.splitlines() if line.strip()]
        records = propose.add(study, proposals)
    runs = len({space.run_key(r) for r in records})
    print(f"lab propose: {len(records)} trials ({records[0]['id']}..{records[-1]['id']}) in {runs} runs "
          f"of cartridge, depth and speed")
    return 0


def cmd_map(ns) -> int:
    """Each archive of the study: which bins of its two descriptors its read trials reached, and each bin's elite."""
    store.load(ns.study)
    tables = [search.grid(g) for g in search.groups(store.trials(ns.study))]
    if ns.json:
        print(json.dumps(tables, indent=1))
        return 0
    if not tables:
        print(f"lab map: {ns.study} has no trials yet")
    for t in tables:
        print(f"{t['technique']} / {t['cartridge']} / {t['ink']}: {t['read']} of {t['trials']} trials read, "
              f"{len(t['elites'])} of {t['bins']} bins reached")
        corner = f"{t['axes'][0]} \\ {t['axes'][1]}"
        width, side = max(8, *map(len, t["cols"])), max(len(corner), *map(len, t["rows"]))
        print(f"  {corner:>{side}}  " + " ".join(f"{c:>{width}}" for c in t["cols"]))
        for r in reversed(range(len(t["rows"]))):   # darkest at the top
            row = [t["elites"].get(f"{r},{c}", {}).get("trial", ".") for c in range(len(t["cols"]))]
            print(f"  {t['rows'][r]:>{side}}  " + " ".join(f"{cell:>{width}}" for cell in row))
    return 0


def cmd_plan(ns) -> int:
    study = store.load(ns.study)
    result = plan.plan(study, store.trials(ns.study), tile.load(ns.stencil), ns.sheet, _catalog())
    for r in result["runs"]:
        est = f"{r['est_s'] / 60:.0f} min modelled" if r["est_s"] else "untimed"
        dose = space.dose_per_mm(study, r)
        print(f"  {r['run']}: {r['cartridge']} trim {r['trim_mm']:g} speed {r['speed_mm_s']:g} ({dose:.1f} pokes/mm), "
              f"{len(r['trials'])} cells, {', '.join(r['inks'])}, {est}\n    {r['draw']}")
    print(f"lab plan: {result['plan']} on sheet {ns.sheet}, {len(result['runs'])} runs, "
          f"{sum(len(r['trials']) for r in result['runs'])} cells, {result['est_s'] / 60:.0f} min modelled"
          + (f"; {len(result['unplaced'])} trials wait for a fresh sheet" if result["unplaced"] else ""))
    return 0


def cmd_status(ns) -> int:
    ids = [ns.study] if ns.study else sorted(d.name for d in store.root().glob("*") if (d / "study.json").exists())
    out = []
    for study_id in ids:
        study, trials = store.load(study_id), store.trials(study_id)
        sheets = Counter(t["placement"]["sheet"] for t in trials.values()
                         if t.get("placement") and t["status"] in plan.LIVE)
        out.append({"study": study_id, "question": study["question"], "trials": dict(Counter(t["status"]
                                                                                             for t in trials.values())),
                    "sheets": dict(sheets), "plans": sorted(d.name for d in store.plans_dir(study_id).glob("p*"))})
    if ns.json:
        print(json.dumps(out, indent=1))
    for s in out if not ns.json else ():
        counts = ", ".join(f"{n} {k}" for k, n in s["trials"].items()) or "no trials"
        print(f"{s['study']}: {counts}; plans {', '.join(s['plans']) or 'none'}; cells used "
              + (", ".join(f"{n} on {k}" for k, n in s["sheets"].items()) or "none") + f"\n  {s['question']}")
    if not out:
        print(f"lab status: no studies under {store.root()}")
    return 0


def cmd_capture(ns) -> int:
    """A run's views on the rig (`tatbot ros views`) into <run>/views/<side>/: the overhead D555's first (unless
    --no-overhead), then every cell from a close-up hold over its centre, or with --rest-view the one view where the arm
    stands; --overhead-only never touches the arm. after and dry aim by the page the before side used, so their poses
    match; with --sheet-at the holds aim by the overhead tracker at the sheet nearest that point instead (no set-up),
    which after and dry take from the before side. before holds the arm awake for the draw, after lands it and marks
    the run's trials drawn, dry (the same views once the ink has dried) changes no trial."""
    import subprocess

    study = store.load(ns.study)
    run_dir = store.plans_dir(ns.study) / ns.plan / ns.run
    cells = lab.cells(json.loads((run_dir / "cells.json").read_text()))
    here = run_dir / "views" / ns.side
    argv = [str(REPO / "scripts" / "tatbot"), "ros", "views", "--arm", study["arm"],
            "--label", f"{ns.study}-{ns.plan}-{ns.run}-{ns.side}", "--fetch", str(here)]
    argv += ["--overhead"] if ns.overhead or ns.overhead_only else []
    if ns.overhead_only:
        argv.append("--no-wrist")
    elif not ns.rest_view:
        argv.append("--points=" + ";".join(f"{(c['box_mm'][0] + c['box_mm'][2]) / 2:.3f},"
                                           f"{(c['box_mm'][1] + c['box_mm'][3]) / 2:.3f}" for c in cells["cells"]))
        argv += _aim(ns, run_dir, cells["pattern_id"])
    argv += ["--hold"] if ns.side == "before" else []
    code = subprocess.run(argv).returncode
    if code:
        return code
    if ns.side == "after":   # drawn: its cells hold ink now, whatever reads them
        trials = store.trials(ns.study)
        store.append(ns.study, [{**trials[c["trial"]], "status": "drawn"} for c in cells["cells"]
                                if trials[c["trial"]]["status"] == "planned"])
    print(f"lab capture: {ns.plan}/{ns.run} {ns.side}, {len(cells['cells'])} cells -> {here}")
    return 0


OVERHEAD_HOVER_MM = 12.0   # the pen's height over the cell at a hold the overhead aims: the stack's close-up height
# (page.inspect.closeup_hover_m). It was 35 while the xArm rig's overhead read the skin 9 mm off (2026-11-08); since its
# re-registration through the wrist D405 it agrees with the arm to ~2 mm (2026-11-09), and the D405s, ~150 mm behind
# the tip, see the cell ~15% finer this low


OVERHEAD_WAIT_S = 20.0   # how long a capture waits for the overhead to measure its sheet: with five of forearm18's six
# cells inked, each over a marker, it measured the sheet in 3 samples of 10, and 5 s found none (2026-11-09)


RUN_STEPS = ("before", "draw", "after", "read")


def cmd_run(ns) -> int:
    """One planned run on the rig, end to end: its cells photographed before (the arm left holding), the run drawn
    by `tatbot ros draw` (its cartridge stated, so a cartridge the mount does not hold is refused; on --sheet-at's
    sheet; no inspection, the arm landed after), photographed after (from that rest, as before), and read from the
    wrist's views and the overhead's. Each step's outcome goes to <run>/lab-run.json; a failed step stops the run,
    and --from resumes at it."""
    from types import SimpleNamespace

    study = store.load(ns.study)
    run_dir = store.plans_dir(ns.study) / ns.plan / ns.run
    plan_path = run_dir.parent / "plan.json"
    summary = next((r for r in json.loads(plan_path.read_text())["runs"] if r["run"] == ns.run), None)
    if summary is None:
        raise ValueError(f"{ns.plan} has no run {ns.run} ({plan_path})")
    record_path = run_dir / "lab-run.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {"plan": ns.plan, "run": ns.run,
                                                                                "steps": {}}
    capture = {"study": ns.study, "plan": ns.plan, "run": ns.run, "rest_view": False, "overhead": True,
               "overhead_only": False}
    for step in RUN_STEPS[RUN_STEPS.index(ns.start):]:
        print(f"lab run: {ns.plan}/{ns.run} {step}", flush=True)
        if step == "before":
            code, extra = cmd_capture(SimpleNamespace(**capture, side="before", sheet_at=ns.sheet_at)), {}
        elif step == "draw":
            code, extra = _draw(study, summary, ns.sheet_at, getattr(ns, "ends", None))
        elif step == "after":
            code, extra = cmd_capture(SimpleNamespace(**capture, side="after", sheet_at=None)), {}
        else:
            drew = (record["steps"].get("draw") or {}).get("ros_run_id")
            code, extra = _read_both(ns, drew), {}
        record["steps"][step] = {"at": store.now(), "exit": code, **extra}
        record_path.write_text(json.dumps(record, indent=1) + "\n")
        if code:
            print(f"lab run: {step} exited {code}; once it is fixed, `tatbot lab run {ns.study} --plan {ns.plan} "
                  f"--run {ns.run} --stencil {ns.stencil} --from {step}`")
            return code
    print(f"lab run: {ns.plan}/{ns.run} drawn and read: {len(summary['trials'])} cells -> {run_dir}")
    return 0


def draw_argv(study: dict, summary: dict, sheet_at: str | None, ends: str | None = None) -> list[str]:
    """`tatbot ros draw` for a planned run: its cartridge stated, its trim, no inspection, and with `ends` its stroke
    ends drawn that way (`--ends`: stop or fly, over the stack's motion.yaml). The arm lands after it: held over the
    sheet it hid the sheet from the overhead, and the after views, aimed by the overhead from rest as the before
    views were, found none (2026-11-09)."""
    return [str(REPO / "scripts" / "tatbot"), "--ee-tool", summary["cartridge"], "ros", "draw", summary["program"],
            "--arm", study["arm"], f"--trim-mm={summary['trim_mm']:g}", "--no-inspect",
            *([f"--sheet-at={sheet_at}"] if sheet_at else []), *([f"--ends={ends}"] if ends else [])]


def _draw(study: dict, summary: dict, sheet_at: str | None, ends: str | None = None) -> tuple[int, dict]:
    """The draw, its output passed through as it comes; the ros node's run id from the client's result line, and
    the ends it was drawn with when the run was given them."""
    import contextlib
    import subprocess

    ros_run = None
    with subprocess.Popen(draw_argv(study, summary, sheet_at, ends), stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.startswith("{") and '"run_id"' in line:
                with contextlib.suppress(ValueError):
                    ros_run = json.loads(line).get("run_id") or ros_run
    return proc.returncode, {"ros_run_id": ros_run, **({"ends": ends} if ends else {})}


def _read_both(ns, ros_run: str | None) -> int:
    """`lab read` of the after views from the wrist, then from the overhead; the first failure's code."""
    from types import SimpleNamespace

    for instrument in ("wrist", "overhead"):
        code = cmd_read(SimpleNamespace(study=ns.study, plan=ns.plan, run=ns.run, stencil=ns.stencil, before=None,
                                        after=None, instrument=instrument, side="after", run_id=ros_run, roi=None))
        if code:
            return code
    return 0


def _aim(ns, run_dir: Path, pattern_id: str) -> list[str]:
    """What the holds aim by: --sheet-at's sheet of the print as the overhead reads it, else (after, dry) the page or
    the sheet the before side aimed by."""
    path = run_dir / "views" / "before" / "views.json"
    before = json.loads(path.read_text()) if ns.side != "before" and path.is_file() else {}
    sheet_at = ns.sheet_at or before.get("sheet_at")
    if sheet_at:
        return [f"--overhead-pages={pattern_id}", f"--sheet-at={sheet_at}", f"--hover-mm={OVERHEAD_HOVER_MM:g}",
                f"--seconds={OVERHEAD_WAIT_S:g}"]
    if ns.side == "before":
        return []
    if not before.get("page_run"):
        raise ValueError(f"{ns.plan}/{ns.run} has no held before views to match: capture --side before first")
    return ["--run", before["page_run"]]


def _region(ns, before: Path, p) -> tuple[list | None, str | None]:
    """Where the markers are read: --roi as given, else for the overhead the sheet the capture chose (--sheet-at:
    views.json's one sheet, projected through its intrinsics), else the whole frame."""
    if ns.roi:
        return _roi(ns.roi), "given"
    path = before / "views.json"
    meta = json.loads(path.read_text()) if path.is_file() else {}
    sheets = [s for s in meta.get("sheets") or [] if s.get("world_from_page")]
    intrinsics = (meta.get("overhead") or {}).get("intrinsics")
    if ns.instrument != "overhead" or not meta.get("sheet_at") or len(sheets) != 1 or not intrinsics:
        return None, None
    from tatbot_lab import read

    roi = read.frames.sheet_roi(sheets[0]["world_from_page"], p.tile_mm, intrinsics)
    if roi is None:
        raise ValueError(f"the chosen sheet ({meta['sheet_at']} mm) lies outside the overhead's frame")
    return roi, "sheet"


def _roi(text: str | None):
    """`x0,y0,x1,y1` image pixels, or None."""
    if not text:
        return None
    box = [float(v) for v in text.split(",")]
    if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("--roi is x0,y0,x1,y1 in image pixels, x0 < x1 and y0 < y1")
    return box


def cmd_read(ns) -> int:
    import cv2

    from tatbot_lab import read

    study, trials = store.load(ns.study), store.trials(ns.study)
    run_dir = store.plans_dir(ns.study) / ns.plan / ns.run
    cells = lab.cells(json.loads((run_dir / "cells.json").read_text()))
    p = tile.load(ns.stencil)
    if p.pattern_id != cells["pattern_id"]:
        raise ValueError(f"the print is {p.pattern_id}, the run's cells are on {cells['pattern_id']}")
    before = Path(ns.before) if ns.before else run_dir / "views" / "before"
    after = Path(ns.after) if ns.after else run_dir / "views" / ns.side
    names = sorted(f.name for f in before.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")
                   and (f.name == "overhead.png") == (ns.instrument == "overhead")) if before.is_dir() else []
    if not names or any(not (after / n).exists() for n in names):
        raise ValueError(f"{before} and {after} hold the same frame names, one pair a view (tatbot lab capture)")
    views = [(cv2.imread(str(before / n)), cv2.imread(str(after / n))) for n in names]
    t = read.frames.tile(p.directory)
    roi, roi_from = _region(ns, before, p)
    if roi_from == "sheet":
        print(f"  the overhead read within the chosen sheet's region {roi} px")
    readings, updated = [], []
    for cell in cells["cells"]:
        trial = trials[cell["trial"]]
        others = [c for c in cells["cells"] if c["id"] != cell["id"]]
        result = read.read_cell(cell, trial["technique"], space.resolve(trial["technique"], trial["settings"]),
                                views, t, others, instrument=ns.instrument, roi=roi)
        if ns.instrument == "overhead":   # ~0.86 px/mm: a cell's mean change, never a line (docs/lab.md)
            result = {k: v for k, v in result.items() if k in OVERHEAD}
        readings.append(result)
        metrics = {k: v for k, v in result.items() if k not in ("cell", "trial")}
        entry = {"at": store.now(), "views": len(views), "instrument": ns.instrument, "side": ns.side,
                 "run_id": ns.run_id, **({"roi": roi, "roi_from": roi_from} if roi else {}), "metrics": metrics}
        drew = {**trial["run"], "run_id": ns.run_id} if ns.run_id and trial.get("run") else trial.get("run")
        updated.append({**trial, "status": "read", "reads": [*trial.get("reads", []), entry], "run": drew,
                        "metrics": _trial_metrics(trial, ns.instrument, metrics)})
        print(f"  {cell['id']} {trial['id']} {trial['technique']}: dL {result.get('dL_mean')} "
              f"(floor {result['noise_dL']}), {len(result['markers'])} markers")
    out = run_dir / ("reads.json" if ns.instrument == "wrist" else f"reads-{ns.instrument}.json")
    out.write_text(json.dumps({"study": study["id"], "plan": ns.plan, "run": ns.run, "instrument": ns.instrument,
                               "views": len(views), "roi": roi, "roi_from": roi_from, "cells": readings},
                              indent=1) + "\n")
    store.append(ns.study, updated)
    print(f"lab read: {len(readings)} cells of {ns.plan}/{ns.run} from {len(views)} views -> {out}")
    return 0


def _trial_metrics(trial: dict, instrument: str, metrics: dict) -> dict:
    """The metrics a trial is judged by (the search's archive and surrogate): the wrist's, which resolve a line, once it
    has a wrist read; the overhead's cell mean only until then. Each read keeps its own in its record: lab run reads the
    wrist and then the overhead, and the overhead's replaced the wrist's ink widths (2026-11-09)."""
    if instrument != "wrist" and any(r.get("instrument") == "wrist" for r in trial.get("reads", [])):
        return trial.get("metrics") or metrics
    return metrics


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tatbot lab")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("new", help="a study: its question, print, cartridges, inks, caps and bounds")
    p.add_argument("--id", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--stencil", required=True, help="the generated print its cells sit on (its directory)")
    p.add_argument("--arm", required=True, help="the drawing arm (the tatbot lab verb gives stack.yaml's)")
    p.add_argument("--cell-mm", type=float, default=16.0)
    p.add_argument("--cartridges", required=True, help="tool ids, in the order to fit them")
    p.add_argument("--inks", required=True)
    p.add_argument("--palette", required=True, help="ink=slot,...,water=slot: the caps that hold them")
    p.add_argument("--techniques", default=",".join(space.TECHNIQUES))
    p.add_argument("--bound", action="append", default=[], help="setting=low:high; repeat. Unbounded settings are "
                                                                "held at their technique's default")
    p.add_argument("--constants", help="override the held machine constants: battery_v=..,machine_hz=..,stroke_mm=..")
    p.set_defaults(func=cmd_new)
    p = sub.add_parser("propose", help="trials from a JSON-lines file, or a cartridge's characterization card")
    p.add_argument("study")
    p.add_argument("--file")
    p.add_argument("--card", metavar="CARTRIDGE")
    p.add_argument("--dip-card", dest="dip_card", metavar="CARTRIDGE",
                   help="the dip card: the cartridge's mark, carry and flood over the dip settings the study bounds")
    p.add_argument("--inks", help="the card's inks, its main first (default the study's); the dip card's is the first")
    p.add_argument("--search", type=int, metavar="N", help="N trials aimed at what the study does not yet know")
    p.add_argument("--technique", help="with --search: the technique whose marks to search")
    p.add_argument("--cartridge", help="with --search: default the study's first")
    p.add_argument("--ink", help="with --search: default the study's first")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-runs", type=int, default=3, help="with --search: distinct depth and speed pairs, each a run")
    p.set_defaults(func=cmd_propose)
    p = sub.add_parser("run", help="one planned run on the rig: views before, the draw, views after, the reads")
    p.add_argument("study")
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--stencil", required=True, help="the study's print (its directory), for the reads")
    p.add_argument("--sheet-at", dest="sheet_at", help="the sheet to draw on, x,y mm in the arm's base")
    p.add_argument("--ends", choices=("stop", "fly"),
                   help="draw the run's stroke ends stopping or moving (`tatbot ros draw --ends`); lab-run.json keeps it")
    p.add_argument("--from", dest="start", default="before", choices=RUN_STEPS, help="resume at this step")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("map", help="the marks each archive's read trials reached, and each bin's elite")
    p.add_argument("study")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_map)
    p = sub.add_parser("plan", help="the proposed trials into runs on a sheet: programs and cell manifests")
    p.add_argument("study")
    p.add_argument("--stencil", required=True)
    p.add_argument("--sheet", required=True, help="the physical sheet's id: a fresh one starts with every cell free")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("status", help="each study's trials by status, plans and cells used")
    p.add_argument("study", nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("capture", help="a run's views on the rig, before or after it draws (tatbot ros views)")
    p.add_argument("study")
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--side", required=True, choices=("before", "after", "dry"))
    p.add_argument("--rest-view", action="store_true", help="the one view where the arm stands, no motion")
    p.add_argument("--no-overhead", dest="overhead", action="store_false", help="no overhead D555 view")
    p.add_argument("--overhead-only", action="store_true", help="the overhead's view alone: the arm is not touched")
    p.add_argument("--sheet-at", dest="sheet_at", help="aim at the sheet nearest x,y mm in the arm's base, as the "
                                                       "overhead reads it (no set-up); after and dry take the before's")
    p.set_defaults(func=cmd_capture)
    p = sub.add_parser("read", help="a drawn run's cells from before and after frames, one pair per view")
    p.add_argument("study")
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--stencil", required=True)
    p.add_argument("--before", help="a directory of frames taken before the run (default its captured views)")
    p.add_argument("--after", help="the same views after it, by the same names (default its captured views)")
    p.add_argument("--instrument", default="wrist", choices=("wrist", "overhead"),
                   help="the wrist views' frames, or the overhead's (overhead.png)")
    p.add_argument("--side", default="after", choices=("after", "dry"), help="the captured views to compare the "
                                                                             "before views with")
    p.add_argument("--run-id", help="the ros-draw run that drew it")
    p.add_argument("--roi", help="x0,y0,x1,y1: read markers only inside this part of the frames (image px), as when "
                                 "two sheets of one print are in the overhead's view")
    p.set_defaults(func=cmd_read)
    return ap


def main(argv=None) -> int:
    ns = parser().parse_args(argv)
    try:
        return ns.func(ns)
    except (ValueError, KeyError, OSError) as error:
        print(f"lab {ns.command}: {error}", file=sys.stderr)
        return 2
