"""`tatbot art tiles`: a round composed for a stencil tile, each design separated at the size that tile holds.

`tatbot art propose --tile LAYOUT` composes a round for a tile: every prompt asks for a small tattoo filling it. Each
of the round's images then goes through `tatbot art separate --fit --stencil PRINT`, drawn at the widest its canvas
the print's page holds and, with DBV3, compiled onto that print at its centre: one program a tile. The separations
land in the corpus's `separations/<id>/`, as any round's do, so the wall, `promote` and `bundle` find them.

A round proposed for no tile, or for another, is refused: a design is drawn at the size it was composed for, never
fitted down to a smaller area. The index (`tatbot.art-tiles/1`) gives per design its size, the inks that drew
nothing, its detail under the style skills' limits and, once acquired, its minutes and dips, flagging a design over
the tile's minute budget (advisory: the votes decide). Provenance stays where the pipeline keeps it: the generation
sidecar in separation.json and in every DBV3 job.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tatbot_contracts.digest import sha256_file

from .loop import round_images, tile_round
from .pipeline import lost_inks


@dataclass(frozen=True)
class Design:
    id: str
    image: Path
    inks: list[str]
    style: str          # the separation style: flash, geometric or linocut
    fill: str
    prompt_id: str


def round_designs(plan_path: Path, corpus: Path, ids: list[str]) -> tuple[dict, list[Design]]:
    """The tile a round was composed for, and its designs in the corpus (only `ids` when given), each with its
    prompt's inks, separation style and fill."""
    plan = json.loads(plan_path.read_text())
    if not plan.get("tile"):
        raise ValueError(f"{plan_path.name} was not composed for a tile; propose a round with --tile instead of "
                         "fitting its designs to one")
    designs = [Design(image.stem, image, list(prompt["inks"]), prompt.get("separation") or "flash",
                      prompt.get("fill") or "packed", prompt["id"])
               for image, _side, prompt in round_images(plan, corpus)]
    if not designs:
        raise ValueError(f"no image in {corpus / 'flow'} answers a prompt of {plan_path.name}")
    if missing := sorted(set(ids) - {d.id for d in designs}):
        raise ValueError(f"not designs of {plan_path.name}: {', '.join(missing)}")
    tile = {**tile_round(plan["tile"]["layout"]), **plan["tile"]}   # what an older plan lacks comes from today's tile
    return tile, [d for d in designs if not ids or d.id in ids]


def print_of(stencil: Path) -> dict:
    """The print's tracking manifest: the print's folder, or a file in it, as `tatbot ros compile --stencil` takes."""
    directory = stencil if stencil.is_dir() else stencil.parent
    return {**json.loads((directory / "tracking.json").read_text()), "dir": str(directory)}


def check_print(tile: dict, manifest: dict) -> None:
    """The print must be the tile the round was composed for: its page that tile's size."""
    if [float(v) for v in manifest["page_mm"]] != [float(v) for v in tile["tile_mm"]]:
        (w, h), (tw, th) = manifest["page_mm"], tile["tile_mm"]
        raise ValueError(f"the print's page is {w:g} x {h:g} mm, and the round was composed for the {tile['layout']} "
                         f"tile, {tw:g} x {th:g} mm")


def row(design: Design, sep_dir: Path, budget_min: float) -> dict:
    """One design's line in the index, from its separation report; once acquired, whether it models over the budget."""
    report = json.loads((sep_dir / "separation.json").read_text())
    entry = {"id": design.id, "prompt_id": design.prompt_id, "image_sha256": report["image_sha256"],
             "inks": design.inks, "style": design.style, "fill": design.fill, "canvas_mm": report["canvas_mm"],
             "fits": report["page"]["fits"], "lost": lost_inks(report), "detail": report["detail"],
             "separation": str(sep_dir), "separation_sha256": sha256_file(sep_dir / "separation.json")}
    if markers := (report.get("fit") or {}).get("markers"):   # the size the print's markers carry, under the page's
        entry["markers"] = {key: markers[key] for key in ("page_mm", "carried_mm")}
    if "time" in report:
        t = report["time"]
        entry.update({"minutes": t["minutes"], "dips": t["dips"], "rinses": t["rinses"],
                      "over_budget": t["minutes"] > budget_min, "fidelity": report["fidelity"]["vs_target"]["mean"],
                      "program": str(sep_dir / "program" / "program.json"), "render": str(sep_dir / "render.png")})
    return entry


def write_index(path: Path, plan_path: Path, tile: dict, manifest: dict, rows: list[dict], failed: list[dict],
                acquired: bool) -> dict:
    index = {"schema": "tatbot.art-tiles/1", "made": datetime.now(UTC).isoformat(timespec="seconds"),
             "plan": str(plan_path), "tile": tile["layout"], "minutes_budget": tile["minutes_budget"],
             "stencil": {"dir": manifest["dir"], "pattern_id": manifest["pattern_id"], "page_mm": manifest["page_mm"]},
             "acquired": acquired, "tiles": rows, "failed": failed}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=1) + "\n")
    return index


def summary(index: dict) -> str:
    rows = index["tiles"]
    lost = sum(bool(r["lost"]) for r in rows)
    text = f"{len(rows)} designs on {index['tile']} print {index['stencil']['pattern_id'][:16]}, {lost} with an ink that drew nothing"
    if minutes := sorted(r["minutes"] for r in rows if "minutes" in r):
        over = sum(r["over_budget"] for r in rows if "minutes" in r)
        text += (f", {minutes[0]}-{minutes[-1]} min each (median {minutes[len(minutes) // 2]}), {over} over the "
                 f"{index['minutes_budget']} min budget")
    return text + (f", {len(index['failed'])} failed" if index["failed"] else "")
