"""`tatbot art bundle`: a drawn design as the files the rig takes, ready for review.

bundle OUT/<id>/
  layers/       each acquired layer's artwork.json, byte for byte, in drawing order (`<nn>-<ink>-<fill>.json`)
  inks.yaml     the separation's tatbot-inks/3 draft: which cap holds which ink (with --one-load, one map for all)
  program/      `tatbot ros compile layers/*.json --inks inks.yaml --at=X,Y`: program.json and preview.svg
  render.png    the program predicted on the substrate
  target.png, drawn.png   the separation's assigned colours and acquired paths, for comparison
  bundle.json   tatbot.art-bundle/1: layers, inks and caps, minutes, dips, fidelity, evidence, file digests

The compile draws the layers on one canvas centred at `--at`, as `tatbot art separate` compiled them. Nothing is
re-acquired: the bundle reuses the separation's acquisitions, so a bundle is free and its program equals the
separation's when placed at the page centre. A design is acquired at its drawn size, so a bundle moves it on the
page but never turns or scales it; Inkmap places the source image instead (`tatbot art separate --brief`).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import yaml
from tatbot_contracts.digest import sha256_file

from .acquire import INK_SLOTS
from .color import to_8bit
from .loop import votes_by_design
from .png import write_png
from .pyramid import stars
from .render import render_program

INK_SUBSTRATES = ("silicon_skin", "human_skin")


def layers(sep_dir: Path) -> list[dict]:
    """The separation's acquired layers in drawing order: {layer, path, ink, fill, angle_deg}.

    Each job's own acquisition, by its job hash, not every artwork under acq/: a fill-study variant's acq/ also
    holds its source's copies. A layer DBV3 refused is left out, as the separation's compile left it out.
    """
    from .pipeline import _acq_dir  # the acquisition directory is named by the job's hash

    report = json.loads((sep_dir / "separation.json").read_text())
    stem = Path(report["image"]).stem
    refused = {r["job"] for r in report.get("refused", [])}
    out = []
    for lay in report["layers"]:
        if lay["job"] in refused:
            continue
        art = _acq_dir(sep_dir / "acq", sep_dir / "jobs" / lay["job"]) / "artwork.json"
        if not art.exists():
            raise ValueError(f"{sep_dir.name}: layer {lay['job']} has no acquisition; draw it with `tatbot art promote`")
        key = lay["job"][len(stem) + 1:] if lay["job"].startswith(f"{stem}-") else lay["job"]
        out.append({"layer": key, "path": art, "ink": lay["ink"], "fill": lay["fill"],
                    "angle_deg": lay.get("angle_deg")})
    if not out:
        raise ValueError(f"{sep_dir.name}: no acquired layers")
    return out


def one_load(sep_dirs: list[Path]) -> dict[str, str]:
    """One cap per ink across several designs, so none is swapped between them: the inks drawing the most line
    together take the largest caps, as a single separation's draft orders its own. Water keeps its large cap."""
    line: dict[str, float] = {}
    for sep in sep_dirs:
        for ink, mm in json.loads((sep / "separation.json").read_text()).get("line_mm", {}).items():
            line[ink] = line.get(ink, 0.0) + mm
    held = {r["ink_id"] for sep in sep_dirs for r in yaml.safe_load((sep / "inks.yaml").read_text())["resources"]}
    inks = sorted(held, key=lambda ink: (-line.get(ink, 0.0), ink))
    if len(inks) > len(INK_SLOTS):
        raise ValueError(f"one load: {len(inks)} inks ({', '.join(inks)}) need more caps than the palette's "
                         f"{len(INK_SLOTS)} beside water; bundle fewer designs together")
    return dict(zip(inks, INK_SLOTS, strict=False))


def _recapped(text: str, caps: dict[str, str]) -> str:
    """The draft with each resource's own cap from `caps`; its rinse water's slot is indented deeper and stays."""
    lines, ink = [], None
    for line in text.splitlines(keepends=True):
        if not line.startswith(" "):
            ink = None
        elif line.startswith("  ink_id: "):
            ink = line.split(":", 1)[1].strip()
        elif ink and line.startswith("  slot: "):
            line = f"  slot: {caps[ink]}\n"
        lines.append(line)
    return "".join(lines)


def inks_draft(sep_dir: Path, arts: dict[str, dict], substrate: str | None,
               caps: dict[str, str] | None = None) -> tuple[str, dict]:
    """The separation's tatbot-inks/3 draft, its substrate and caps as asked; it must bind exactly these artworks."""
    text = (sep_dir / "inks.yaml").read_text()
    draft = yaml.safe_load(text)
    bound = {b["artwork_sha256"] for b in draft["bindings"]}
    held = {a["content_sha256"] for a in arts.values()}
    if bound != held:
        raise ValueError(f"{sep_dir.name}: inks.yaml binds other artwork than the acquisitions; separate it again")
    if substrate and substrate != draft["substrate"]:
        if substrate not in INK_SUBSTRATES:
            raise ValueError(f"unknown ink substrate {substrate!r}; have {', '.join(INK_SUBSTRATES)}")
        text = re.sub(r"(?m)^substrate: .*$", f"substrate: {substrate}", text, count=1)
        draft["substrate"] = substrate
    if caps:
        text = _recapped(text, caps)
        draft = yaml.safe_load(text)
    return text, draft


def build(sep_dir: Path, out: Path, catalog: dict, compile_program, *, at_mm=(0.0, 0.0),
          stencil: Path | None = None, substrate: str = "silicone", ink_substrate: str | None = None,
          spread_mm: float = 0.0, wall: dict | None = None, caps: dict[str, str] | None = None) -> dict:
    """One design's bundle. `compile_program(artworks, inks, out_dir, extra_args)` returns the compile's stats."""
    report = json.loads((sep_dir / "separation.json").read_text())
    lays = layers(sep_dir)
    sep_sha = sha256_file(sep_dir / "separation.json")
    out.mkdir(parents=True, exist_ok=True)
    (out / "layers").mkdir(exist_ok=True)
    files = []
    for lay in lays:   # the frozen records travel byte for byte, so inks.yaml still binds them
        files.append(out / "layers" / f"{lay['layer']}.json")
        shutil.copyfile(lay["path"], files[-1])
    arts = {lay["layer"]: json.loads(path.read_text()) for lay, path in zip(lays, files, strict=True)}
    inks_text, draft = inks_draft(sep_dir, arts, ink_substrate, caps)
    (out / "inks.yaml").write_text(inks_text)
    stats = compile_program(files, out / "inks.yaml", out / "program",
                            [f"--at={at_mm[0]},{at_mm[1]}", *(["--stencil", str(stencil)] if stencil else [])])
    program = json.loads((out / "program" / "program.json").read_text())
    rendering = render_program(program, catalog, substrate, report.get("px_per_mm", 10.0), spread_mm=spread_mm)
    write_png(out / "render.png", to_8bit(rendering.srgb))
    for name in ("target.png", "drawn.png"):
        if (sep_dir / name).exists():
            shutil.copyfile(sep_dir / name, out / name)
    caps = {r["ink_id"]: r.get("slot") for r in draft["resources"]}
    line = report.get("line_mm", {})
    side = report.get("generation") or {}
    bundle = {
        "schema": "tatbot.art-bundle/1", "id": sep_dir.name,
        "image": Path(report["image"]).name, "image_sha256": report["image_sha256"],
        "separation_sha256": sep_sha, "prompt": side.get("prompt"), "model": side.get("model"),
        "style": report.get("style"), "solid_fill": report.get("solid_fill") or "packed",
        "genes": (wall or {}).get("genes"), "round": (wall or {}).get("round"),
        "canvas_mm": report["canvas_mm"],
        "placement": {"page": str(stencil) if stencil else "nominal stencil page 128 x 172 mm",
                      "at_mm": list(at_mm)},
        "layers": [{"layer": lay["layer"], "file": f"layers/{lay['layer']}.json",
                    "artwork_sha256": arts[lay["layer"]]["content_sha256"], "ink": lay["ink"], "fill": lay["fill"],
                    "angle_deg": lay["angle_deg"]} for lay in lays],
        "inks": [{"ink": ink, "cap": caps.get(ink), "line_mm": line.get(ink),
                  "evidence": report.get("inks", {}).get(ink, {}).get("source")} for ink in caps],
        "substrate": {"inks": draft["substrate"], "render": substrate, "spread_mm": spread_mm},
        "compile": {k: stats.get(k) for k in ("strokes", "paths", "contact_m", "dips", "tool_changes", "est_s")},
        "time": rendering.report["time"], "fidelity": (report.get("fidelity") or {}).get("vs_target"),
        "weakest_source": report.get("weakest_source"),
        "refused": report.get("refused", []),
        "review": {"status": "pending", "note": "read render.png, program/preview.svg and the minutes before drawing"},
        "draw": "tatbot ros draw program/program.json",
    }
    bundle["files"] = {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*"))
                       if p.is_file() and p.name != "bundle.json"}
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1) + "\n")
    return bundle


def chosen(votes: dict, *, kept: bool) -> list[str]:
    """The shortlist: designs any viewer starred, and with `kept` every design with a positive vote total."""
    pick = set(stars(votes))
    if kept:
        pick |= {did for did, total in votes_by_design(votes).items() if total > 0}
    return sorted(pick)


def _drawn(sep_dir: Path) -> bool:
    return (sep_dir / "separation.json").exists() and "time" in json.loads((sep_dir / "separation.json").read_text())


def run(ids: list[str], corpus: Path, out: Path, catalog: dict, compile_program, *, designs: dict | None = None,
        shared_load: bool = False, poses: dict[str, tuple[float, float]] | None = None, log=print, **options) -> dict:
    """Bundle each drawn design; a sketch with no robot paths yet is listed as waiting for `art promote`.

    `shared_load` gives every drawn design the same caps (`one_load`), so they are drawn in turn on one palette;
    `poses` centres a design at its own page position instead of `at_mm`, so several share one page.
    """
    done, waiting = [], []
    drawn = [corpus / "separations" / did for did in ids if _drawn(corpus / "separations" / did)]
    caps = one_load(drawn) if shared_load and drawn else None
    for did in ids:
        sep_dir = corpus / "separations" / did
        if sep_dir not in drawn:
            waiting.append({"id": did, "reason": "not drawn yet: promote it, then bundle"})
            continue
        posed = {**options, "at_mm": (poses or {})[did]} if did in (poses or {}) else options
        bundle = build(sep_dir, out / did, catalog, compile_program, wall=(designs or {}).get(did), caps=caps,
                       **posed)
        t = bundle["time"]
        log(f"art bundle: {did}: {len(bundle['layers'])} layers, {len(bundle['inks'])} inks, {t['minutes']} min "
            f"({t['dips']} dips, {t['rinses']} rinses) -> {out / did}")
        done.append({"id": did, "dir": did, "at_mm": bundle["placement"]["at_mm"], "minutes": t["minutes"], "dips": t["dips"],
                     "inks": [i["ink"] for i in bundle["inks"]], "style": bundle["style"]})
    index = {"schema": "tatbot.art-bundles/1", "bundles": done, "waiting": waiting}
    if caps:
        index["load"] = caps
    (out / "bundles.json").write_text(json.dumps(index, indent=1) + "\n")
    return index
