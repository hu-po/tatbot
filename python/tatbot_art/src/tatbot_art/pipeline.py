"""`tatbot art separate` and `tatbot art tone` end to end, writing everything under one output directory.

separate OUT/
  target.png          the colours the separation assigned, over white
  jobs/<id>/          one tatbot.dbv3-job/3 per (ink, fill), with its greyscale source; an offset-fill layer's
                      offset.json instead (tatbot_art.offset: its cartridge and pitch)
  acq/<id>-<job sha>/ the acquisitions, DBV3's and the offset fill's (reused while the job is unchanged)
  inks.yaml           tatbot-inks/3 draft: assumed caps, water rinses
  program/            `tatbot ros compile` output
  render.png          the program predicted on the substrate
  drawn.png           the acquired paths over white at the canvas size, beside target.png
  separation.json     palette use, layers, minutes, fidelity and evidence
  inkmap-render.json  with --brief: the predicted ink on white, for Inkmap to show on the body
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import acquire, image, tone
from .color import to_8bit
from .inks import InkLook, look, weakest
from .metrics import blurred_delta_e
from .png import write_png
from .render import render_program
from .separate import STYLES, Layer, _disk, detail, ink_order, separate, usage

MAX_INKS = len(acquire.INK_SLOTS)


def _sidecar(image_path: Path) -> dict | None:
    side = image_path.with_suffix(".json")
    if side.exists():
        data = json.loads(side.read_text())
        if data.get("schema") == "tatbot.art-generation/1":
            return data
    return None


def _acq_dir(root: Path, job_dir: Path) -> Path:
    spec = job_dir / "job.json" if (job_dir / "job.json").exists() else job_dir / OFFSET_SPEC
    return root / f"{job_dir.name}-{hashlib.sha256(spec.read_bytes()).hexdigest()[:8]}"


OFFSET_SPEC = "offset.json"   # an offset-fill layer's job: its cartridge, pitch and source


def _offset_layers(layers: list[Layer], tools: dict[str, str], overlap_mm: float, px_per_mm: float) -> list[Layer]:
    """The solid layers for the offset fill, each ink's cartridge's together. A lighter ink grows `overlap_mm` under
    the inks drawn after it (never into paper), so a small misplacement between two cartridges' calibrations hides
    under the darker line rather than showing paper.

    A negative `overlap_mm` separates for the other order, the darkest ink drawn first: the layers come darkest
    first, and each lighter ink keeps that far clear of every darker one drawn before it, a gap in place of the
    underlap, so its needles never run through the darker ink (2026-11-10: a red drawn after the black it reached
    under carried black into the red cap; the compile refuses that order, tatbot_ink.order)."""
    for lay in layers:
        if lay.fill == "solid":
            lay.fill = "solid:offset"
    if overlap_mm < 0:
        return _dark_first(layers[::-1], tools, round(-overlap_mm * px_per_mm), px_per_mm)
    order = list(dict.fromkeys(tools[lay.ink_id] for lay in layers))
    layers = sorted(layers, key=lambda lay: order.index(tools[lay.ink_id]))   # stable: lightest first within a tool
    solids = [lay for lay in layers if lay.fill == "solid:offset"]
    grow = round(overlap_mm * px_per_mm)
    for k, lay in enumerate(solids):
        later = np.zeros(lay.source.shape, bool)
        for other in solids[k + 1:]:
            later |= other.source < 255
        if grow >= 1 and later.any():
            region = lay.source < 255
            grown = ndi.binary_dilation(region, structure=_disk(grow)) & (region | later)
            lay.source = np.where(grown, lay.source[region].min(), 255).astype(np.uint8)
            lay.area_mm2 = float(grown.sum()) / px_per_mm ** 2
    return layers


def _dark_first(layers: list[Layer], tools: dict[str, str], gap: int, px_per_mm: float) -> list[Layer]:
    """The layers darkest first, each cartridge's together, every solid kept `gap` pixels clear of the solids drawn
    before it."""
    order = list(dict.fromkeys(tools[lay.ink_id] for lay in layers))
    layers = sorted(layers, key=lambda lay: order.index(tools[lay.ink_id]))   # stable: darkest first within a tool
    drawn = None
    for lay in (lay for lay in layers if lay.fill == "solid:offset"):
        region = lay.source < 255
        if drawn is not None and gap >= 1 and lay.ink_id not in drawn[1]:
            kept = region & ~ndi.binary_dilation(drawn[0], structure=_disk(gap))
            if not kept.any():
                raise ValueError(f"--overlap-mm {-gap / px_per_mm:g}: nothing of {lay.ink_id} stays that far clear of "
                                 "the inks drawn before it")
            lay.source = np.where(kept, lay.source[region].min(), 255).astype(np.uint8)
            lay.area_mm2 = float(kept.sum()) / px_per_mm ** 2
        drawn = (region if drawn is None else drawn[0] | region, {lay.ink_id} | (drawn[1] if drawn else set()))
    return layers


def _write_offset_job(root: Path, jid: str, layer: Layer, tool_id: str, pitch_mm: float, canvas_mm) -> Path:
    d = root / jid
    (d / "input").mkdir(parents=True, exist_ok=True)
    png = write_png(d / "input" / "source.png", np.repeat(layer.source[..., None], 3, axis=2))
    spec = {"schema": "tatbot.art-offset-job/1", "id": jid, "ink_id": layer.ink_id, "tool_id": tool_id,
            "settings": {"pitch_mm": pitch_mm}, "size_mm": [round(float(v), 3) for v in canvas_mm],
            "source": {"file": "input/source.png", "sha256": hashlib.sha256(png.read_bytes()).hexdigest()}}
    (d / OFFSET_SPEC).write_text(json.dumps(spec, indent=1) + "\n")
    return d


def _offset_artwork(job_dir: Path, ink: InkLook, px_per_mm: float, out: Path) -> Path:
    """An offset-fill layer's acquisition: its passes, frozen as artwork, and how they were made (offset.json)."""
    from . import offset

    acq = _acq_dir(out, job_dir)
    if not (acq / "artwork.json").exists():
        spec = json.loads((job_dir / OFFSET_SPEC).read_text())
        from PIL import Image

        with Image.open(job_dir / spec["source"]["file"]) as im:
            region = np.asarray(im.convert("L")) < 255
        paths, info = offset.generate(region, px_per_mm, offset.Offset(**spec["settings"]))
        art = offset.artwork(paths, tuple(spec["size_mm"]), ink.ink_id.replace("_", "-"), ink.srgb, info,
                             name=spec["id"], source_sha256=spec["source"]["sha256"], identifier=spec["id"])
        acq.mkdir(parents=True, exist_ok=True)
        (acq / OFFSET_SPEC).write_text(json.dumps({**spec, "result": info}, indent=1) + "\n")
        (acq / "artwork.json").write_text(json.dumps(art, indent=1) + "\n")
    return acq / "artwork.json"


def _texture(layers, solid_fill: str) -> None:
    """Swap the solid layers' fill for another texture: its PFM, and the source grey that PFM wants."""
    if solid_fill == "packed":
        return
    grey = acquire.SOLID_FILLS[solid_fill][3]
    for lay in layers:
        if lay.fill == "solid":
            lay.fill = f"solid:{solid_fill}"
            lay.source = np.where(lay.source < 255, grey, 255).astype(np.uint8)


def plan(image_path: Path, inks: list[str], catalog: dict, out: Path, *, width_mm: float, style: str,
         px_per_mm: float, line_width_mm: float, mixes: bool, seed: int, tone_path: Path | None,
         solid_fill: str = "packed", tools: dict[str, str] | None = None, pitch_mm: dict[str, float] | None = None,
         overlap_mm: float = 0.0):
    """The separation and its jobs. `tools` binds inks to cartridges (default: the fitted one); with the offset fill
    each ink's passes lie `pitch_mm` apart (default: its cartridge's mark), and lighter inks overlap darker ones
    (a negative `overlap_mm`: darkest first, lighter inks kept clear of them)."""
    if not 1 <= len(inks) <= MAX_INKS:
        raise ValueError(f"a load holds 1 to {MAX_INKS} inks beside the rinse water, got {len(inks)}")
    looks = {i: look(catalog, i) for i in inks}
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; have {', '.join(STYLES)}")
    unknown = set(tools or {}) - set(inks)
    if unknown:
        raise ValueError(f"--tool names inks the load does not hold: {', '.join(sorted(unknown))}")
    tools = {ink: (tools or {}).get(ink) or acquire.fitted_tool() for ink in inks}
    fill_offset = solid_fill == "offset"
    if not fill_offset and solid_fill not in acquire.SOLID_FILLS:
        raise ValueError(f"unknown fill {solid_fill!r}; have {', '.join((*acquire.SOLID_FILLS, 'offset'))}")
    if (pitch_mm or overlap_mm) and not fill_offset:
        raise ValueError("--pitch-mm and --overlap-mm set the offset fill's passes; DBV3's fills take the line width")
    rgb = image.prepare(image_path, width_mm, px_per_mm, keep_lines=STYLES[style]["lines"])
    sep = separate(rgb, looks.values(), style=style, tone=tone.load(tone_path), px_per_mm=px_per_mm,
                   line_width_mm=line_width_mm, mixes=mixes, offset=fill_offset)
    if fill_offset:
        from . import offset

        pitch = {ink: (pitch_mm or {}).get(ink) or offset.mark_mm(tools[ink]) for ink in inks}
        sep.layers = _offset_layers(sep.layers, tools, overlap_mm, px_per_mm)
    else:
        _texture(sep.layers, solid_fill)
    out.mkdir(parents=True, exist_ok=True)
    write_png(out / "target.png", to_8bit(sep.target))
    canvas_mm = (sep.labels.shape[1] / px_per_mm, sep.labels.shape[0] / px_per_mm)
    side = _sidecar(image_path)
    jobs = []
    for k, layer in enumerate(sep.layers):
        jid = acquire.job_id(image_path.stem, k, layer)
        prov = acquire.provenance(jid, side, image_path.name)
        if layer.fill == "solid:offset":
            jobs.append(_write_offset_job(out / "jobs", jid, layer, tools[layer.ink_id], pitch[layer.ink_id], canvas_mm))
            continue
        jobs.append(acquire.write_job(out / "jobs", jid, layer, looks[layer.ink_id], canvas_mm, line_width_mm,
                                      seed, prov))
    report = {"schema": "tatbot.art-separation/1", "image": str(image_path),
              "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(), "generation": side,
              "inks": {i: {"source": lk.source} for i, lk in looks.items()}, "weakest_source": weakest(looks.values()),
              "style": style, "solid_fill": solid_fill, "tools": tools,
              **({"pitch_mm": pitch, "overlap_mm": overlap_mm} if fill_offset else {}),
              "canvas_mm": [round(v, 2) for v in canvas_mm], "px_per_mm": px_per_mm,
              "line_width_mm": line_width_mm, "mixes": mixes, "colours": usage(sep),
              "layers": [{"job": j.name, "ink": lay.ink_id, "fill": lay.fill, "angle_deg": lay.angle_deg,
                          "area_mm2": round(lay.area_mm2, 1)} for lay, j in zip(sep.layers, jobs, strict=True)],
              "detail": detail(sep, ink_order(looks.values())[-1])}
    return sep, looks, jobs, canvas_mm, report


def check_page(report: dict, canvas_mm, stencil: Path | None, at: str | None) -> str | None:
    """Record whether the canvas fits its page; a plain refusal when it does not."""
    page = report["page"] = acquire.page_fit(canvas_mm, stencil, at)
    if page["fits"]:
        return None
    (w, h), (limit_w, limit_h) = page["canvas_mm"], page["drawable_mm"]
    where = f"at {at} " if at else ""
    return (f"the design is {w:g} x {h:g} mm, and {'this print' if stencil else 'the nominal page'} {where}holds "
            f"{limit_w:g} x {limit_h:g} mm: draw it smaller{'' if stencil else ', or name a larger print with --stencil'}")


def build(sep, catalog, jobs, canvas_mm, report: dict, out: Path, *, app: Path, substrate: str,
          stencil: Path | None = None, at: str | None = None, allow_marker_loss: bool = False) -> dict:
    if refusal := check_page(report, canvas_mm, stencil, at):
        raise ValueError(refusal)
    compile_args = [*(["--stencil", str(stencil)] if stencil else []), *([f"--at={at}"] if at else []),
                    *(["--allow-marker-loss"] if allow_marker_loss else [])]
    looks = {i: look(catalog, i) for i in report["inks"]}
    if app is None and any(lay.fill != "solid:offset" for lay in sep.layers):
        raise ValueError("DBV3 layers need --app, an activated DBV3 Premium")
    arts = [_offset_artwork(j, looks[lay.ink_id], sep.px_per_mm, out / "acq") if lay.fill == "solid:offset"
            else acquire.generate(j, _acq_dir(out / "acq", j), app) for lay, j in zip(sep.layers, jobs, strict=True)]
    refused = [{"job": j.name, "reason": (j / "refused.txt").read_text().strip()}
               for j, a in zip(jobs, arts, strict=True) if a is None]
    pairs = [(lay, a) for lay, a in zip(sep.layers, arts, strict=True) if a is not None]
    if not pairs:
        raise RuntimeError(f"DBV3 refused every layer: {refused}")
    arts = [a for _, a in pairs]
    line = {}
    for lay, art in pairs:
        line[lay.ink_id] = line.get(lay.ink_id, 0.0) + acquire.line_mm(art)
    inks = acquire.inks_file(out / "inks.yaml", pairs, line, report.get("tools") or acquire.fitted_tool())
    stats = acquire.compile_program(arts, inks, out / "program", compile_args)
    program = json.loads((out / "program" / "program.json").read_text())
    rendering = render_program(program, catalog, substrate, sep.px_per_mm)
    write_png(out / "render.png", to_8bit(rendering.srgb))
    flat = render_program(acquire.canvas_program(pairs, canvas_mm), catalog, (255, 255, 255), sep.px_per_mm, fade=False)
    write_png(out / "drawn.png", to_8bit(flat.srgb))
    h, w = (min(a, b) for a, b in zip(sep.target.shape[:2], flat.srgb.shape[:2], strict=True))
    inked = (sep.target[:h, :w] < 0.98).any(-1) | (flat.srgb[:h, :w] < 0.98).any(-1)
    report.update({"refused": refused, "line_mm": {k: round(v) for k, v in line.items()}, "compile": stats,
                   "time": rendering.report["time"], "substrate": substrate,
                   "fidelity": {"vs_target": blurred_delta_e(sep.target[:h, :w], flat.srgb[:h, :w], sep.px_per_mm,
                                                             where=inked)}})
    return report


def run_tone(app: Path, out: Path, width_mm: float, seed: int) -> dict:
    wedge, layout = tone.wedge()
    grey = wedge[..., 0]
    layer = Layer("true_black", "hatch", grey, 45.0, 0.0)
    black = InkLook("true_black", "Wedge black", (0.0, 0.0, 0.0), "nominal", False)
    canvas = (grey.shape[1] / layout["px_per_mm"], grey.shape[0] / layout["px_per_mm"])
    prov = {"kind": "fixture", "identifier": "tone-wedge", "license": None, "generation": None,
            "attribution": "tatbot art tone step wedge"}
    job = acquire.write_job(out / "jobs", "tone-wedge", layer, black, canvas, width_mm, seed, prov)
    art = acquire.generate(job, _acq_dir(out / "acq", job), app)
    if art is None:
        raise RuntimeError(f"DBV3 refused the tone wedge: {(job / 'refused.txt').read_text().strip()}")
    points = tone.measure(json.loads(art.read_text()), layout, width_mm)
    curve = {"pfm": acquire.FILLS["hatch"][0], "pen_width_mm": width_mm, "points": points,
             "source": f"step wedge acquired by tatbot art tone, seed {seed}"}
    (out / "tone.json").write_text(json.dumps(curve, indent=1) + "\n")
    return curve


def lost_inks(report: dict) -> list[str]:
    """The load's inks that drew nothing, which a design's thinnest colour does at small sizes."""
    drawn = report["line_mm"] if "line_mm" in report else {lay["ink"] for lay in report["layers"]}
    return [ink for ink in report["inks"] if ink not in drawn]


def ink_on_white(out: Path, catalog: dict, canvas_mm, px_per_mm: float, at_mm) -> np.ndarray:
    """The compiled program's predicted ink over white, cropped to the design's canvas on the page. White is what
    lets a viewer lay it on any skin: ink there multiplies what lies under it, and white leaves it alone."""
    program = json.loads((out / "program" / "program.json").read_text())
    picture = render_program(program, catalog, (255, 255, 255), px_per_mm).srgb
    width, height = (int(round(v * px_per_mm)) for v in canvas_mm)
    left = int(round(picture.shape[1] / 2 + (at_mm[0] - canvas_mm[0] / 2) * px_per_mm))
    top = int(round(picture.shape[0] / 2 - (at_mm[1] + canvas_mm[1] / 2) * px_per_mm))
    return picture[top:top + height, left:left + width]


def detail_note(report: dict) -> str:
    """The detail the drawn size costs, by the style skills' limits: the share of colour in regions narrower than the
    fine width, naming each colour ink mostly in them, and the paper gaps ink spread would close."""
    d = report.get("detail")
    if not d:
        return ""
    colour = {ink: row for ink, row in d["inks"].items() if ink != d["darkest"] and row["area_mm2"]}
    notes = []
    if colour:
        share = sum(r["under_mm2"] for r in colour.values()) / sum(r["area_mm2"] for r in colour.values())
        mostly = [f"{ink} {round(100 * r['share'])}%" for ink, r in colour.items() if r["share"] >= 0.5]
        notes.append(f"colour under {d['fine_mm']:g} mm {round(100 * share)}%" + (f" ({', '.join(mostly)})" if mostly else ""))
    if d["gaps_mm2"]:
        notes.append(f"{d['gaps_mm2']:g} mm² of paper gaps under {d['gap_mm']:g} mm")
    return "".join(f"; {n}" for n in notes)


def summary_line(report: dict) -> str:
    colours = len(report["colours"])
    mixes = sum(len(c["layers"]) > 1 for c in report["colours"])
    traced = sum(lay.get("fill") == "solid:offset" for lay in report["layers"])
    kind = f"{len(report['layers']) - traced} DBV3 layers" + (f" and {traced} offset" if traced else "")
    base = (f"{len(report['inks'])} inks, {colours} colours ({mixes} mixes), {kind}, "
            f"weakest source {report['weakest_source']}")
    lost = lost_inks(report)
    note = (f"; nothing drawn in {', '.join(lost)}" if lost else "") + detail_note(report)
    if "time" not in report:
        return base + note
    t = report["time"]
    swaps = f", {t['swaps']} cartridge swaps" if t.get("swaps") else ""
    return (f"{base}; {t['minutes']} min ({t['dips']} dips, {t['rinses']} rinses{swaps}), "
            f"fidelity dE00 {report['fidelity']['vs_target']['mean']} mean at 30 cm{note}")

