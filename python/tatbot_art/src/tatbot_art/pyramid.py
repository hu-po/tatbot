"""The pyramid: what each design on the wall has earned, and the work the next pass spends on it.

0 sketch      every generated image: the Flow picture and its plan-only separation into the load's inks.
1 drawn       a sketch the owner kept: DBV3 acquires its robot paths and the render model times them.
2 fill study  a drawn design the owner kept: its solid regions acquired again in every other contained fill, each
              variant a design of its own on the wall, so a vote on a variant is a vote on the fill alone.
3 shortlist   a design the owner starred: what to draw when the rig is back.

A keep promotes; a pass spends a budget of DBV3 separations (one per drawing, one per variant), handed
out round-robin across styles, so one style's run of keeps cannot take every slot.
"""
from __future__ import annotations

import copy
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .loop import SEPARATION, measured, votes_by_design

TIERS = ("sketch", "drawn", "fill study", "shortlist")
WIDTH_MM = 60.0  # the drawn width of a design composed for no particular size; a tile round's is its tile's


def stars(votes: dict) -> set[str]:
    """Designs any viewer starred: `s` in their vote document."""
    return {did for doc in votes.values() if isinstance(doc, dict) and isinstance(doc.get("s"), dict)
            for did, on in doc["s"].items() if on}


def tiers(designs: dict[str, dict], votes: dict) -> dict[str, int]:
    starred = stars(votes)
    studied = {d["study_of"] for d in designs.values() if d.get("study_of")}
    out = {}
    for did, doc in designs.items():
        if did in starred:
            out[did] = 3
        elif doc.get("study_of") or did in studied:
            out[did] = 2
        else:
            out[did] = 1 if (doc.get("views") or {}).get("drawn") else 0
    return out


def separation_of(doc: dict) -> str:
    return doc.get("separation") or SEPARATION.get(doc.get("style"), "flash")


def promotions(designs: dict[str, dict], votes: dict, fills: list[str], budget: int) -> list[dict]:
    """The pass's work: kept sketches to draw, then kept drawings to study, each stratified by style.

    Within a style the best-kept and newest go first; across styles one at a time, best style first.
    """
    totals, level = votes_by_design(votes), tiers(designs, votes)
    queues: dict[str, dict[str, list]] = {"draw": {}, "study": {}}
    for did, doc in designs.items():
        drawn = (doc.get("views") or {}).get("drawn")
        if totals.get(did, 0) <= 0 or doc.get("study_of") or (level[did] > 1 and drawn):
            continue   # a starred sketch is still drawn: its bundle waits on the robot's paths
        kind = "study" if drawn else "draw"
        queues[kind].setdefault(doc["genes"]["style"], []).append((-totals[did], -doc.get("round", 0), did))
    work, spent = [], 0
    for kind, by_style in queues.items():
        for q in by_style.values():
            q.sort()
        order = sorted(by_style, key=lambda s: (by_style[s][0][0], s))
        while any(by_style[s] for s in order):
            for style in order:
                if not by_style[style]:
                    continue
                _, _, did = by_style[style].pop(0)
                own = designs[did].get("fill") or "packed"
                todo = [own] if kind == "draw" else [f for f in fills if f != own]
                if spent + len(todo) > budget:
                    return work
                work.append({"id": did, "kind": kind, "style": style, "fills": todo, "score": totals[did]})
                spent += len(todo)
    return work


def on_disk(designs: dict[str, dict], corpus: Path) -> dict[str, dict]:
    """The designs as the corpus has them: drawn once `drawn.png` exists, studied once a variant's
    `study.json` does, whether or not the wall has been updated yet."""
    out = {did: {**d, "views": dict(d.get("views") or {})} for did, d in designs.items()}
    for did, d in out.items():
        if not d["views"].get("drawn") and (corpus / "separations" / did / "drawn.png").exists():
            d["views"]["drawn"] = "acquired"
    for path in corpus.glob("separations/*~*/study.json"):
        study = json.loads(path.read_text())
        src = out.get(study.get("of"))
        if src is not None and path.parent.name not in out:
            out[path.parent.name] = {**src, "study_of": study["of"], "fill": study.get("fill"), "parents": [study["of"]]}
    return out


def images(corpus: Path) -> dict[str, Path]:
    return {p.stem: p for p in corpus.glob("flow/*/*.jpg")}


def sketched_at(sep_dir: Path) -> tuple[float, Path | None]:
    """The drawn width and print of a design's sketch, so it is drawn at the size it was shown at: a tile round's
    on its tile's print, anything else at WIDTH_MM on the nominal page."""
    path = sep_dir / "separation.json"
    if not path.exists():
        return WIDTH_MM, None
    report = json.loads(path.read_text())
    stencil = (report.get("page") or {}).get("stencil")
    return float(report["canvas_mm"][0]), Path(stencil) if stencil else None


def run(item: dict, doc: dict, corpus: Path, catalog: dict, app: Path, log=print) -> list[Path]:
    """Acquire one promotion through DBV3, at the size and on the print of the design's sketch. A study variant starts
    from its source's acquisitions, so only the solid layers, the ones the fill changes, go back to DBV3."""
    from . import pipeline

    image = images(corpus)[item["id"]]
    inks = [i["id"] for i in doc["inks"]]
    base = corpus / "separations" / item["id"]
    width_mm, stencil = sketched_at(base)
    outs = []
    for fill in item["fills"]:
        out = base if item["kind"] == "draw" else corpus / "separations" / f"{item['id']}~{fill}"
        if item["kind"] == "study" and (base / "acq").is_dir():
            shutil.copytree(base / "acq", out / "acq", dirs_exist_ok=True)
        sep, _looks, jobs, canvas, report = pipeline.plan(
            image, inks, catalog, out, width_mm=width_mm, style=separation_of(doc), px_per_mm=10.0,
            line_width_mm=0.6, mixes=True, seed=42, tone_path=None, solid_fill=fill)
        report = pipeline.build(sep, catalog, jobs, canvas, report, out, app=app, substrate="silicone", stencil=stencil)
        (out / "separation.json").write_text(json.dumps(report, indent=1) + "\n")
        if item["kind"] == "study":
            (out / "study.json").write_text(json.dumps(
                {"schema": "tatbot.art-study/1", "of": item["id"], "fill": fill,
                 "made": datetime.now(UTC).isoformat(timespec="seconds")}, indent=1) + "\n")
        log(f"art promote: {item['kind']} {out.name}: {pipeline.summary_line(report)}")
        outs.append(out)
    return outs


def study_docs(base: dict[str, dict], corpus: Path, assets: dict) -> dict[str, dict]:
    """Wall documents for the fill-study variants of `base`: the source's genes and pictures with the
    variant's fill, robot paths and measurements; `study_of` and `parents` name the source."""
    out = {}
    for path in sorted(corpus.glob("separations/*~*/study.json")):
        study = json.loads(path.read_text())
        src = base.get(study.get("of"))
        sid = path.parent.name
        if src is None or study.get("schema") != "tatbot.art-study/1":
            continue
        doc = copy.deepcopy(src)
        for key in ("minutes", "dips", "colours", "mixes", "fidelity"):
            doc.pop(key, None)
        doc.update({"fill": study["fill"], "study_of": study["of"], "parents": [study["of"]],
                    "variant": "fill-study", "order": src.get("order", 0) + 0.5})
        doc["genes"] = {**src["genes"], "fill": study["fill"], "variant": "fill-study"}
        doc["views"] = {k: v for k, v in src.get("views", {}).items() if k in ("flow", "ours")}
        if f"{sid}-drawn" in assets:
            doc["views"]["drawn"] = f"/_blob/{assets[f'{sid}-drawn']}"
        sep_path = path.parent / "separation.json"
        if sep_path.exists():
            doc.update(measured(json.loads(sep_path.read_text())))
        out[sid] = doc
    return out

