"""`tatbot art`: offline art-studio commands. No robot, camera or network."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .acquire import REPO, SOLID_FILLS
from .color import to_8bit
from .gamut import achievable
from .inks import load_catalog, look, weakest
from .png import write_png
from .render import SUBSTRATES, render_program
from .swatch import sheet

STYLES = ("flash", "geometric", "linocut")


def _out(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_swatch(ns) -> int:
    catalog = load_catalog(ns.catalog)
    looks = {i: look(catalog, i, ns.substrate) for i in ns.inks.split(",")}
    sub = tuple(c / 255 for c in SUBSTRATES[ns.substrate])
    swatches = achievable(list(looks.values()), sub, ns.width_mm, n=ns.n, min_delta_e=ns.min_delta_e)
    out = _out(ns.out)
    write_png(out / "swatches.png", to_8bit(sheet(swatches, looks, sub, ns.px_per_mm)))
    report = {"inks": {i: {"source": lk.source, "hex": "#%02x%02x%02x" % tuple(to_8bit(lk.srgb))}
                       for i, lk in looks.items()},
              "weakest_source": weakest(looks.values()), "substrate": ns.substrate, "width_mm": ns.width_mm,
              "n": ns.n, "min_delta_e": ns.min_delta_e, "count": len(swatches),
              "swatches": [sw.as_dict() for sw in swatches]}
    (out / "swatches.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"art swatch: {len(swatches)} colours from {len(looks)} inks on {ns.substrate} "
          f"(weakest source {report['weakest_source']}) -> {out}")
    return 0


def cmd_render(ns) -> int:
    program = json.loads(Path(ns.program).read_text())
    rendering = render_program(program, load_catalog(ns.catalog), ns.substrate, ns.px_per_mm, ns.width_mm,
                               ns.spread_mm, fade=not ns.no_fade)
    out = _out(ns.out)
    write_png(out / "render.png", to_8bit(rendering.srgb))
    (out / "render.json").write_text(json.dumps(rendering.report, indent=1) + "\n")
    t = rendering.report["time"]
    print(f"art render: {len(rendering.coverage)} inks on {ns.substrate}, {t['minutes']} min "
          f"({t['dips']} dips, {t['rinses']} rinses, {t['swaps']} swaps), weakest source "
          f"{rendering.report['weakest_source']} -> {out}")
    return 0


def cmd_profile(ns) -> int:
    from . import profile

    gaps = {k: getattr(ns, f"{k}_s") for k in ("dip", "rinse", "swap") if getattr(ns, f"{k}_s") is not None}
    report = profile.write(ns.program, ns.out, gaps)
    print(profile.text(report))
    print(f"\nReport: {ns.out / 'profile.html'}")
    return 0


def _separation_input(ns) -> tuple[Path, float, list[str], str, dict]:
    """The image, width, inks and style to separate, and what the report records of how: as given, fitted to the
    page (`fit`, the room it was fitted to), or one placement of an Inkmap brief (`brief`)."""
    if not ns.brief:
        if ns.image is None or (ns.width_mm is None) == (not ns.fit) or not ns.inks:
            raise ValueError("separate takes an image with --inks and one of --width-mm or --fit, or --brief")
        if not ns.fit:
            return ns.image, ns.width_mm, ns.inks.split(","), ns.style or "flash", {}
        from . import acquire, image

        room = acquire.page_room(ns.stencil, ns.at)
        return (ns.image, image.fit_width(ns.image, room, ns.px_per_mm), ns.inks.split(","), ns.style or "flash",
                {"fit": {"room_mm": [round(v, 2) for v in room]}})
    from . import brief

    if ns.image is not None or ns.width_mm is not None or ns.fit:
        raise ValueError("--brief carries the image and its width; Inkmap is where they change")
    given = brief.source(ns.brief, ns.placement, ns.out)
    inks = ns.inks.split(",") if ns.inks else given["inks"]
    if not inks:
        raise ValueError("the brief's art records no inks; pass --inks")
    style = ns.style or (given["style"] if given["style"] in STYLES else "flash")
    return given["image"], given["width_mm"], inks, style, {"brief": given["report"]}


def _per_ink(values: list[str] | None, flag: str, cast=str) -> dict:
    """`INK=VALUE` arguments, each named once."""
    out = {}
    for value in values or []:
        ink, sep, given = value.partition("=")
        if not sep or not ink or not given or ink in out:
            raise ValueError(f"{flag} takes INK=VALUE, each ink once (got {value!r})")
        out[ink] = cast(given)
    return out


def _planned_only(ns, report: dict, canvas_mm, record: dict | None) -> None:
    """Without DBV3: the separation and its jobs are written, and the page check said."""
    from . import fit, pipeline

    refusal = pipeline.check_page(report, canvas_mm, ns.stencil, ns.at)
    (ns.out / "separation.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"art separate: {pipeline.summary_line(report)} -> {ns.out}")
    if record:
        print(f"art separate: {fit.line(record)}, by its regions: its paths are counted once DBV3 draws them")
    print("art separate: planned only; pass --app with an activated DBV3 Premium to acquire, compile and render")
    if refusal:
        print(f"art separate: {refusal}")


def _built(ns, catalog: dict, planned, record: dict | None) -> tuple[dict, bool]:
    """(the report of one size acquired, compiled and rendered; whether its size stands). With a marker fit under
    way (`record`) a size whose compiled paths are under the marker budget does not stand: the compile counted it
    so (stats.markers), or, the budget enforced, refused it."""
    from . import fit, pipeline

    sep, _looks, jobs, canvas_mm, report = planned
    try:
        report = pipeline.build(sep, catalog, jobs, canvas_mm, report, ns.out, app=ns.app, substrate=ns.substrate,
                                stencil=ns.stencil, at=ns.at, allow_marker_loss=ns.allow_marker_loss)
    except (ValueError, RuntimeError) as exc:
        if record is not None and fit.REFUSED in str(exc):
            record["steps"].append({"width_mm": canvas_mm[0], "by": "program", "within": False})
            return report, False
        (ns.out / "separation.json").write_text(json.dumps({**report, "failed": str(exc)}, indent=1) + "\n")
        raise
    counted = None if record is None else report["compile"].get("markers")
    if counted is None:
        return report, True
    record["steps"].append(fit.step(canvas_mm[0], counted, "program"))
    return report, counted["within_budget"]


def _separate_one(ns, catalog: dict) -> None:
    import tempfile

    from . import fit, pipeline  # imaging dependencies load only for the commands that read images

    image, width_mm, inks, style, how = _separation_input(ns)

    def plan_at(width: float, out: Path):
        return pipeline.plan(
            image, inks, catalog, out, width_mm=width, style=style, px_per_mm=ns.px_per_mm,
            line_width_mm=ns.line_width_mm, mixes=not ns.no_mixes, seed=ns.seed, tone_path=ns.tone, solid_fill=ns.fill,
            tools=_per_ink(ns.tool, "--tool"), pitch_mm=_per_ink(ns.pitch_mm, "--pitch-mm", float),
            overlap_mm=ns.overlap_mm)

    def tried(width: float):   # a size tried on the way down leaves no jobs behind
        with tempfile.TemporaryDirectory() as scratch:
            sep, looks, _jobs, canvas_mm, _report = plan_at(width, Path(scratch))
        return sep, looks, canvas_mm

    # --fit on a print with markers: the widest the page holds, then down until the marker budget holds
    record = None
    if "fit" in how and ns.stencil is not None and not ns.allow_marker_loss:
        width_mm, record = fit.carried(tried, width_mm, ns.px_per_mm, ns.stencil, ns.at)
    written = ns.out / "separation.json"
    while True:
        planned = plan_at(width_mm, ns.out)
        sep, canvas_mm, report = planned[0], planned[3], planned[4]
        report.update(how if record is None else {**how, "fit": {**how["fit"], "markers": record}})
        if not ns.app and any(lay.fill != "solid:offset" for lay in sep.layers):
            return _planned_only(ns, report, canvas_mm, record)
        report, stands = _built(ns, catalog, planned, record)
        if stands:
            break
        width_mm = fit.smaller(width_mm, ns.px_per_mm)
    if record:
        record["carried_mm"] = round(width_mm, 2)
        report["fit"]["markers"] = record
        print(f"art separate: {fit.line(record)}")
    written.write_text(json.dumps(report, indent=1) + "\n")
    print(f"art separate: {pipeline.summary_line(report)} -> {ns.out}")
    if "brief" in how:
        from . import brief

        picture = pipeline.ink_on_white(ns.out, catalog, canvas_mm, ns.px_per_mm, report["page"]["at_mm"])
        print(f"art separate: load {brief.write_render(ns.out, report, picture, pipeline.lost_inks(report))} in Inkmap")


def cmd_separate(ns) -> int:
    catalog = load_catalog(ns.catalog)
    if ns.brief and ns.placement is None:
        from . import brief

        if len(order := brief.in_order(ns.brief)) > 1:
            return _separate_each(ns, catalog, order)
    _separate_one(ns, catalog)
    return 0


def _separate_each(ns, catalog: dict, order: list[tuple[str, str]]) -> int:
    """A brief's placements in drawing order, one numbered folder each; one that fails does not stop the rest."""
    failed = 0
    for placement_id, folder in order:
        try:
            _separate_one(argparse.Namespace(**{**vars(ns), "placement": placement_id, "out": ns.out / folder}), catalog)
        except (ValueError, RuntimeError) as exc:
            failed += 1
            print(f"art: {folder}: {exc}", file=sys.stderr)
    print(f"art separate: {len(order) - failed} of {len(order)} placements separated -> {ns.out}")
    return 1 if failed else 0


def cmd_stock_renders(ns) -> int:
    """Inkmap's stock renders, made as any other: a brief per design and width, through `separate`. Each file binds its
    art by digest, and web/inkmap/tests/designs.test.ts fails on one the collection has moved past."""
    from . import brief

    designs = ns.designs
    (designs / "renders").mkdir(exist_ok=True)
    manifest = json.loads((designs / "manifest.json").read_text())
    for entry in manifest["designs"]:
        record = brief.stock_art(entry, (designs.parent / entry["image"]).read_bytes())
        entry["renders"] = []
        for width in (float(value) for value in ns.widths.split(",")):
            name = f"{entry['id']}-{width:g}"
            folder = ns.out / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "brief.json").write_text(json.dumps(brief.stock_brief(entry["id"], record, width)))
            catalog = ["--catalog", str(ns.catalog)] if ns.catalog else []
            if rc := main([*catalog, "separate", "--brief", str(folder / "brief.json"), "--out", str(folder), "--app", str(ns.app)]):
                return rc
            shutil.copy(folder / "inkmap-render.json", designs / "renders" / f"{name}.json")
            entry["renders"].append(f"designs/renders/{name}.json")
    (designs / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"art stock-renders: {sum(len(e['renders']) for e in manifest['designs'])} renders -> {designs / 'renders'}")
    return 0


def cmd_tone(ns) -> int:
    from . import pipeline

    curve = pipeline.run_tone(ns.app, ns.out, ns.width_mm, ns.seed)
    print(f"art tone: {curve['points']} -> {ns.out / 'tone.json'}")
    return 0


def cmd_shade(ns) -> int:
    import hashlib

    from . import cartridge
    from .shade import Shading, artwork, generate, preview_program, tone_from_image

    tone = tone_from_image(ns.image, ns.width_mm, ns.px_per_mm)
    shading = Shading(technique=ns.technique, angle_deg=ns.angle_deg, pitch_mm=ns.pitch_mm, floor=ns.floor,
                      rise_mm=ns.rise_mm)
    strokes, info = generate(tone, ns.px_per_mm, shading, cartridge.load(ns.tool, ns.row_deg))
    if not strokes:
        raise ValueError(f"no tone in {ns.image} reaches the floor {ns.floor:g}: nothing to shade")
    size_mm = (tone.shape[1] / ns.px_per_mm, tone.shape[0] / ns.px_per_mm)
    program = preview_program(strokes, size_mm, ns.ink, info["mark"])
    rendering = render_program(program, load_catalog(ns.catalog), ns.substrate, ns.px_per_mm, fade=False)
    out = _out(ns.out)
    write_png(out / "render.png", to_8bit(rendering.srgb))
    (out / "preview-program.json").write_text(json.dumps(program) + "\n")
    art = artwork(strokes, size_mm, ns.ink, info, name=f"{Path(ns.image).stem}-{ns.technique}",
                  source_sha256=hashlib.sha256(Path(ns.image).read_bytes()).hexdigest(), identifier=Path(ns.image).name)
    (out / "artwork.json").write_text(json.dumps(art, indent=1) + "\n")
    report = {"image": str(ns.image), "size_mm": [round(v, 2) for v in size_mm], "ink": ns.ink, "shading": info,
              "strokes": len(strokes), "render": rendering.report}
    (out / "shade.json").write_text(json.dumps(report, indent=1) + "\n")
    line, t, mark = rendering.report["inks"][ns.ink], rendering.report["time"], info["mark"]
    print(f"art shade: {ns.technique} with {ns.tool} ({mark['kind']}, {mark['width_mm']} mm wide) at "
          f"{info['angle_deg']:g} deg, {info['off_descent_deg']:g} deg off the tone's descent, {len(strokes)} strokes, "
          f"{line['contact_mm']} mm of line ({line['partial_mm']} mm below full deposition), "
          f"{t['minutes']} min modeled -> {out}")
    return 0


def cmd_wall(ns) -> int:
    from .loop import round_images, wall_design, write_json
    from .pyramid import study_docs
    from .wall_previews import enrich

    catalog = load_catalog(ns.catalog)
    assets = json.loads(ns.assets.read_text())["assets"]
    designs = {}
    for order, (image, side, prompt) in enumerate(round_images(json.loads(ns.plan.read_text()), ns.corpus), 1):
        sep_path = ns.corpus / "separations" / image.stem / "separation.json"
        separation = json.loads(sep_path.read_text()) if sep_path.exists() else None
        designs[image.stem] = wall_design(image, side, prompt["inks"], catalog, assets, ns.round, order, separation,
                                          prompt.get("fill", "packed"), prompt.get("parents"), prompt.get("separation"))
    studies = study_docs(designs, ns.corpus, assets)
    designs.update(studies)
    enrich(designs, ns.corpus, catalog, assets, ns.stencil_previews)
    write_json(ns.out, designs)
    drawn = sum("drawn" in d["views"] for d in designs.values())
    print(f"art wall: round {ns.round}, {len(designs) - len(studies)} designs and {len(studies)} fill-study variants "
          f"({drawn} with robot paths) -> {ns.out}")
    return 0


def cmd_propose(ns) -> int:
    from .loop import gene_scores, propose, tile_round, write_json

    designs = json.loads(ns.designs.read_text())
    votes = json.loads(ns.votes.read_text()) if ns.votes.exists() else {}
    asks = json.loads(ns.asks.read_text()) if ns.asks and ns.asks.exists() else {}
    tile = tile_round(ns.tile) if ns.tile else None
    prompts = propose(designs, votes, load_catalog(ns.catalog), ns.round, ns.n, explore=ns.explore, seed=ns.seed,
                      asks=asks, tile=tile)
    used = sorted({p["ask"] for p in prompts if p.get("ask")})
    write_json(ns.out, {"generator": "flow-ui", "model": "Nano Banana Pro", "aspect": tile["aspect"] if tile else "1:1",
                        "outputs_per_prompt": 3, "round": ns.round, "project": None, "tile": tile, "prompts": prompts,
                        "asks_used": used, "gene_scores": gene_scores(designs, votes)})
    composed = f", composed for the {ns.tile} tile (Flow aspect {tile['aspect']})" if tile else ""
    print(f"art propose: round {ns.round}, {len(prompts)} prompts ({len(used)} asks answered, "
          f"{sum(p['variant'] == 'explore' for p in prompts)} exploring){composed} -> {ns.out}")
    return 0


def cmd_map(ns) -> int:
    from datetime import UTC, datetime

    from .atlas import descriptor, graph
    from .loop import votes_by_design, write_json
    from .pyramid import images, separation_of

    designs = json.loads(ns.designs.read_text())
    votes = json.loads(ns.votes.read_text()) if ns.votes and ns.votes.exists() else {}
    found = images(ns.corpus)
    looks, cache = {}, {}
    for did in designs:
        src = found.get(did.split("~")[0])  # a fill-study variant looks like its source image
        if src is not None:
            looks[did] = cache.setdefault(src, descriptor(src))
    family = {d["genes"]["style"]: separation_of(d) for d in designs.values() if "genes" in d}
    doc = graph({k: v for k, v in designs.items() if "genes" in v}, looks, family, votes_by_design(votes),
                datetime.now(UTC).isoformat(timespec="seconds"))
    write_json(ns.out, doc)
    missing = len(designs) - len(doc["ids"])
    print(f"art map: {len(doc['ids'])} designs in {len(doc['clusters'])} clusters, {len(doc['near'])} links"
          f"{f', {missing} without a Flow image' if missing else ''} -> {ns.out}")
    return 0


def cmd_promote(ns) -> int:
    from .loop import FILLS, write_json
    from .pyramid import TIERS, on_disk, promotions, run, tiers

    designs = on_disk(json.loads(ns.designs.read_text()), ns.corpus)
    votes = json.loads(ns.votes.read_text()) if ns.votes.exists() else {}
    fills = ns.fills.split(",") if ns.fills else list(FILLS)
    unknown = [f for f in fills if f not in SOLID_FILLS]
    if unknown:
        raise ValueError(f"unknown fill {', '.join(unknown)}; have {', '.join(SOLID_FILLS)}")
    work = promotions(designs, votes, fills, ns.budget)
    level = tiers(designs, votes)
    write_json(ns.out, {"schema": "tatbot.art-promotions/1", "budget": ns.budget, "fills": fills, "work": work,
                        "tiers": {t: sum(v == k for v in level.values()) for k, t in enumerate(TIERS)}})
    runs = sum(len(w["fills"]) for w in work)
    print(f"art promote: {sum(w['kind'] == 'draw' for w in work)} to draw, {sum(w['kind'] == 'study' for w in work)} "
          f"to study, {runs} DBV3 separations of {ns.budget} -> {ns.out}")
    if not ns.app:
        print("art promote: planned only; pass --app with an activated DBV3 Premium to acquire them")
        return 0
    catalog = load_catalog(ns.catalog)
    for item in work:
        run(item, designs[item["id"]], ns.corpus, catalog, ns.app)
    return 0


def cmd_bundle(ns) -> int:
    from . import acquire
    from .bundle import chosen, run

    ids = list(ns.ids)
    if ns.votes:
        ids += [did for did in chosen(json.loads(ns.votes.read_text()), kept=ns.kept) if did not in ids]
    if not ids:
        raise ValueError("name designs, or pass --votes to bundle the starred ones")
    at, poses = (0.0, 0.0), {}
    for given in ns.at:
        did, _, xy = given.rpartition("=")
        try:
            pose = tuple(float(v) for v in xy.split(","))
        except ValueError:
            pose = ()
        if len(pose) != 2:
            raise ValueError(f"--at takes X,Y or ID=X,Y in page millimetres, got {given!r}")
        if not did:
            at = pose
        elif did in ids:
            poses[did] = pose
        else:
            raise ValueError(f"--at {given}: {did} is not among the designs bundled")
    index = run(ids, ns.corpus, ns.out, load_catalog(ns.catalog),
                acquire.compile_program,
                designs=json.loads(ns.designs.read_text()) if ns.designs else None, at_mm=at, poses=poses,
                stencil=ns.stencil, substrate=ns.substrate,
                ink_substrate=ns.ink_substrate, spread_mm=ns.spread_mm, shared_load=ns.one_load)
    if load := index.get("load"):
        print("art bundle: one load: " + ", ".join(f"{ink} in {cap}" for ink, cap in load.items()))
    for item in index["waiting"]:
        print(f"art bundle: {item['id']}: {item['reason']}")
    print(f"art bundle: {len(index['bundles'])} bundled, {len(index['waiting'])} waiting -> {ns.out / 'bundles.json'}")
    return 0


def cmd_tiles(ns) -> int:
    from . import tiles

    tile, designs = tiles.round_designs(ns.plan, ns.corpus, list(ns.ids))
    manifest = tiles.print_of(ns.stencil)
    tiles.check_print(tile, manifest)
    catalog = load_catalog(ns.catalog)
    rows, failed = [], []
    for design in designs:
        out = ns.corpus / "separations" / design.id
        argv = ["separate", str(design.image), "--inks", ",".join(design.inks), "--style", design.style,
                "--fill", design.fill, "--fit", "--stencil", str(ns.stencil), "--substrate", ns.substrate,
                "--out", str(out), *(["--app", str(ns.app)] if ns.app else [])]
        try:
            _separate_one(build_parser().parse_args(argv), catalog)
        except (ValueError, RuntimeError) as exc:
            failed.append({"id": design.id, "reason": str(exc)})
            print(f"art tiles: {design.id}: {exc}", file=sys.stderr)
            continue
        rows.append(tiles.row(design, out, tile["minutes_budget"]))
    index = tiles.write_index(ns.out, ns.plan, tile, manifest, rows, failed, acquired=bool(ns.app))
    print(f"art tiles: {tiles.summary(index)} -> {ns.out}")
    if not ns.app:
        print("art tiles: planned only; pass --app with an activated DBV3 Premium for programs, renders and minutes")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tatbot art", description=__doc__)
    parser.add_argument("--catalog", type=Path, help="ink catalogue (default: config/inks.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)
    sw = sub.add_parser("swatch", help="the achievable-palette sheet for a palette load")
    sw.add_argument("--inks", required=True, help="comma-separated catalogue ink ids, the palette load")
    sw.add_argument("--substrate", default="silicone", choices=sorted(SUBSTRATES))
    sw.add_argument("--width-mm", type=float, default=0.6, help="deposited line width")
    sw.add_argument("--n", type=float, default=2.0, help="Yule-Nielsen factor until the swatch card fits it")
    sw.add_argument("--min-delta-e", type=float, default=2.0, help="CIEDE2000 below which two swatches are one")
    sw.add_argument("--px-per-mm", type=float, default=10.0)
    sw.add_argument("--out", type=Path, required=True)
    sw.set_defaults(func=cmd_swatch)
    rd = sub.add_parser("render", help="predicted appearance and minutes of a tatbot-program")
    rd.add_argument("program", type=Path)
    rd.add_argument("--substrate", default="silicone", choices=sorted(SUBSTRATES))
    rd.add_argument("--width-mm", type=float, help="deposited width where the tool has no measured one")
    rd.add_argument("--spread-mm", type=float, default=0.0, help="ink spread blur (skin; larger to preview ageing)")
    rd.add_argument("--no-fade", action="store_true", help="draw every stroke at full strength")
    rd.add_argument("--px-per-mm", type=float, default=10.0)
    rd.add_argument("--out", type=Path, required=True)
    rd.set_defaults(func=cmd_render)
    pf = sub.add_parser("profile", help="explain a saved program's drawing, travel and workflow time")
    pf.add_argument("program", type=Path, help="an original compiled tatbot-program with duration/2 statistics")
    pf.add_argument("--out", type=Path, required=True, help="directory for profile.html and profile.json")
    for kind in ("dip", "rinse", "swap"):
        pf.add_argument(f"--{kind}-s", type=float, help=f"{kind} gap in seconds, overriding the historical allowance")
    pf.set_defaults(func=cmd_profile)
    sp = sub.add_parser("separate", help="a flat-colour image into per-ink DBV3 layers, a program and a render")
    sp.add_argument("image", type=Path, nargs="?", help="the source image; --brief supplies it instead")
    sp.add_argument("--inks", help="comma-separated catalogue ink ids, at most five (with --brief: the art's own)")
    sp.add_argument("--width-mm", type=float, help="drawn width; height follows the image (--brief: the placed one)")
    sp.add_argument("--fit", action="store_true",
                    help="instead of --width-mm, the widest the page holds: --stencil's print at --at, else the "
                         "nominal; on a print with markers, the widest that also keeps its marker budget")
    sp.add_argument("--brief", type=Path, help="an Inkmap tattoo brief: separate its placed image at its width")
    sp.add_argument("--placement", help="with --brief, one placement; without it, every placement in drawing order")
    sp.add_argument("--style", choices=STYLES,
                    help="outline policy: linocut traces no edges (default flash, or the art's own with --brief)")
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--app", type=Path, help="activated DBV3 Premium 1.6.22; without it the jobs are only planned")
    sp.add_argument("--substrate", default="silicone", choices=sorted(SUBSTRATES))
    sp.add_argument("--no-mixes", action="store_true", help="single-ink tints only, no two-ink hatch mixes")
    sp.add_argument("--fill", default="packed", choices=(*SOLID_FILLS, "offset"),
                    help="the DBV3 texture for regions an ink covers fully (default: ECS packing), or offset: passes "
                         "one mark apart along their edges, made by tatbot_art.offset, which needs no DBV3")
    sp.add_argument("--tool", action="append", metavar="INK=TOOL",
                    help="the cartridge an ink is drawn with (default: the fitted one); another cartridge is a manual "
                         "exchange, the arm landed, where an ink on the same one is a rinse")
    sp.add_argument("--pitch-mm", action="append", metavar="INK=MM",
                    help="with --fill offset: an ink's passes this far apart (default: its cartridge's mark)")
    sp.add_argument("--overlap-mm", type=float, default=0.0,
                    help="with --fill offset: a lighter ink reaches this far under the inks drawn after it; negative "
                         "separates for the darkest ink drawn first, each lighter ink kept that far clear of it")
    sp.add_argument("--px-per-mm", type=float, default=10.0)
    sp.add_argument("--line-width-mm", type=float, default=0.6)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--tone", type=Path, help="tone curve from `tatbot art tone` (default: the measured 0.6 mm one)")
    sp.add_argument("--stencil", type=Path, help="passed to tatbot ros compile: the page a stencil describes")
    sp.add_argument("--allow-marker-loss", action="store_true",
                    help="passed to tatbot ros compile: draw over the print's markers past the marker budget, and "
                         "with --fit keep the widest the page holds (a sheet fixed on its mount)")
    sp.add_argument("--at", help="passed to tatbot ros compile: the canvas centre, X,Y mm in the page frame")
    sp.set_defaults(func=cmd_separate)
    tn = sub.add_parser("tone", help="measure DBV3 Sketch Lines' grey-to-coverage response with a step wedge")
    tn.add_argument("--app", type=Path, required=True)
    tn.add_argument("--width-mm", type=float, default=0.6)
    tn.add_argument("--seed", type=int, default=42)
    tn.add_argument("--out", type=Path, required=True)
    tn.set_defaults(func=cmd_tone)
    sh = sub.add_parser("shade", help="a tone image as whip strokes or pendulum zigzags fading with the tone")
    sh.add_argument("image", type=Path, help="darker is more ink; white and transparent draw nothing")
    sh.add_argument("--width-mm", type=float, required=True, help="drawn width; the height follows the image")
    sh.add_argument("--technique", default="whip", choices=("whip", "pendulum"))
    sh.add_argument("--angle-deg", type=float, help="stroke direction, dark to light, CCW from +x (default: the tone's)")
    sh.add_argument("--tool", default="lutin-3rl-bugpin", help="the cartridge's datasheet id: its mark's width")
    sh.add_argument("--row-deg", type=float, default=0.0,
                    help="a magnum's needle row on the page, CCW from +x (default: the rig's, along the print's x)")
    sh.add_argument("--pitch-mm", type=float, help="between neighbouring strokes (default: the mark's width)")
    sh.add_argument("--floor", type=float, default=0.05, help="tone under this draws nothing")
    sh.add_argument("--rise-mm", type=float, default=4.0, help="shortest path for deposition to rise from 0 to 1")
    sh.add_argument("--ink", default="true_black", help="catalogue ink id the render draws with")
    sh.add_argument("--substrate", default="silicone", choices=sorted(SUBSTRATES))
    sh.add_argument("--px-per-mm", type=float, default=10.0)
    sh.add_argument("--out", type=Path, required=True)
    sh.set_defaults(func=cmd_shade)
    wl = sub.add_parser("wall", help="a round's designs as voting-wall documents")
    wl.add_argument("--round", type=int, required=True)
    wl.add_argument("--plan", type=Path, required=True, help="the round's Flow prompt plan (prompt ids and inks)")
    wl.add_argument("--corpus", type=Path, required=True, help="the art corpus: flow/ images and separations/")
    wl.add_argument("--assets", type=Path, required=True, help="{assets: {<image>-<view>: asset id}} from the uploads")
    wl.add_argument("--stencil-previews", type=Path, help="export scale previews for recorded stencil placements")
    wl.add_argument("--out", type=Path, required=True)
    wl.set_defaults(func=cmd_wall)
    pr = sub.add_parser("propose", help="the next round's prompts from the wall's votes")
    pr.add_argument("--designs", type=Path, required=True, help="the wall's designs, {id: document}")
    pr.add_argument("--votes", type=Path, required=True, help="the wall's votes, {viewer: {design: +1 or -1}}")
    pr.add_argument("--round", type=int, required=True, help="the round being proposed")
    pr.add_argument("--n", type=int, default=9, help="prompts (three images each)")
    pr.add_argument("--explore", type=float, default=0.34, help="share of prompts on untried genes, 0 to 1")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--asks", type=Path, help="the wall's asks, {id: document}; open ones take the first prompts")
    pr.add_argument("--tile", help="compose every design for this stencil tile (config/stencil-tiles.json), e.g. forearm18")
    pr.add_argument("--out", type=Path, required=True)
    pr.set_defaults(func=cmd_propose)
    mp = sub.add_parser("map", help="the wall's map: layout, neighbourhoods and nearest designs")
    mp.add_argument("--designs", type=Path, required=True, help="the wall's designs, {id: document}")
    mp.add_argument("--votes", type=Path, help="the wall's votes, to score each neighbourhood")
    mp.add_argument("--corpus", type=Path, required=True, help="the art corpus: flow/ images")
    mp.add_argument("--out", type=Path, required=True)
    mp.set_defaults(func=cmd_map)
    pm = sub.add_parser("promote", help="the pyramid's next pass: kept sketches to draw, kept drawings to study")
    pm.add_argument("--designs", type=Path, required=True, help="the wall's designs, {id: document}")
    pm.add_argument("--votes", type=Path, required=True, help="the wall's vote documents, {viewer: {v, s}}")
    pm.add_argument("--corpus", type=Path, required=True, help="the art corpus: flow/ images and separations/")
    pm.add_argument("--budget", type=int, default=40, help="DBV3 separations this pass may spend")
    pm.add_argument("--fills", help="comma-separated fills a study tries (default: the contained ones, loop.FILLS)")
    pm.add_argument("--app", type=Path, help="activated DBV3 Premium 1.6.22; without it the pass is only planned")
    pm.add_argument("--out", type=Path, required=True)
    pm.set_defaults(func=cmd_promote)
    bd = sub.add_parser("bundle", help="drawn designs as reviewable bundles: layers, inks, program, render")
    bd.add_argument("ids", nargs="*", help="design ids: separations/<id> in the corpus")
    bd.add_argument("--corpus", type=Path, required=True, help="the art corpus: flow/ images and separations/")
    bd.add_argument("--votes", type=Path, help="the wall's vote documents; adds every starred design")
    bd.add_argument("--kept", action="store_true", help="with --votes, also every design with a positive vote total")
    bd.add_argument("--designs", type=Path, help="the wall's designs, {id: document}, for genes and round")
    bd.add_argument("--at", action="append", default=[],
                    help="the design's centre on the page, X,Y mm (x right, y toward the top); ID=X,Y poses one "
                         "design, so several share one page (repeatable)")
    bd.add_argument("--stencil", type=Path, help="passed to tatbot ros compile: the print the design is drawn on")
    bd.add_argument("--substrate", default="silicone", choices=sorted(SUBSTRATES), help="the render's substrate")
    bd.add_argument("--ink-substrate", choices=("silicon_skin", "human_skin"),
                    help="tatbot-inks/3 substrate (default: the separation's draft, silicon_skin)")
    bd.add_argument("--spread-mm", type=float, default=0.0, help="ink spread blur for skin renders")
    bd.add_argument("--one-load", action="store_true",
                    help="give every design the same caps, by their combined line, so none is swapped between them")
    bd.add_argument("--out", type=Path, required=True, help="one directory per design, and bundles.json")
    bd.set_defaults(func=cmd_bundle)
    sr = sub.add_parser("stock-renders", help="Inkmap's robot renders of its stock designs, through DBV3")
    sr.add_argument("--app", type=Path, required=True, help="activated DBV3 Premium 1.6.22")
    sr.add_argument("--widths", default="45,60", help="comma-separated drawn widths, mm: the skin tile's and the default")
    sr.add_argument("--designs", type=Path, default=REPO / "web/inkmap/public/designs", help="the stock collection")
    sr.add_argument("--out", type=Path, required=True, help="working directory: each separation, kept for review")
    sr.set_defaults(func=cmd_stock_renders)
    tl = sub.add_parser("tiles", help="a round composed for a tile, separated at its size: one program a tile")
    tl.add_argument("ids", nargs="*", help="the round's designs to take (default: all of them)")
    tl.add_argument("--plan", type=Path, required=True, help="the round's Flow prompt plan, from propose --tile")
    tl.add_argument("--corpus", type=Path, required=True,
                    help="the art corpus: the round's flow/ images; their separations go to separations/<id>/")
    tl.add_argument("--stencil", type=Path, required=True, help="a generated print of the tile the round was composed for")
    tl.add_argument("--app", type=Path, help="activated DBV3 Premium 1.6.22; without it the designs are only planned")
    tl.add_argument("--substrate", default="silicone", choices=sorted(SUBSTRATES))
    tl.add_argument("--out", type=Path, required=True, help="the index, tatbot.art-tiles/1 JSON")
    tl.set_defaults(func=cmd_tiles)
    return parser


def main(argv=None) -> int:
    ns = build_parser().parse_args(argv)
    try:
        return ns.func(ns)
    except ValueError as exc:
        print(f"art: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:   # a DBV3 or compile failure: the input was well formed
        print(f"art: {exc}", file=sys.stderr)
        return 1
