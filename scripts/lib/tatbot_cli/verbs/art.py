"""The art studio: ink gamut and predicted appearance; offline, no robot."""
from __future__ import annotations

from tatbot_cli.registry import OFFLINE, declare_noun, verb

from ._common import sh

declare_noun("art", "development", "palette-aware design: ink gamut, predicted appearance and minutes")

SCRIPT = "scripts/art.sh"
WRAPS = (SCRIPT, "scripts/art.py", "scripts/lib/drawing_python.sh", "scripts/lib/drawing_python.py",
         "python/tatbot_art/src/tatbot_art/cli.py", "python/tatbot_art/src/tatbot_art/inks.py")
SUBSTRATES = ("paper", "silicone", "skin-1", "skin-2", "skin-3", "skin-4", "skin-5", "skin-6")


def _swatch_args(p):
    p.add_argument("--inks", required=True, help="comma-separated catalogue ink ids: the palette load")
    p.add_argument("--substrate", default="silicone", choices=SUBSTRATES)
    p.add_argument("--width-mm", default="0.6", help="deposited line width")
    p.add_argument("--out", required=True, help="output directory for swatches.png and swatches.json")


@verb(noun="art", verb="swatch", tier=OFFLINE, visibility="public", audience="primary",
      disposition="what a palette load can draw, before any image is generated or separated",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="the achievable-palette sheet: every tint and two-ink hatch mix a palette load can draw",
      args=_swatch_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/gamut.py",
                                "python/tatbot_art/src/tatbot_art/swatch.py"), doc="docs/art.md",
      example=("--inks", "golden_yellow,bright_red,light_green,light_purple,true_black", "--out", "/tmp/swatches"),
      invariants=("Colours come from config/inks.yaml; the report names the weakest evidence used.",
                  "No robot, camera or network access beyond provisioning the pinned interpreter."))
def art_swatch(ctx, ns, rest):
    return sh(ctx, SCRIPT, "swatch", "--inks", ns.inks, "--substrate", ns.substrate, "--width-mm", ns.width_mm,
              "--out", ns.out, *rest)


def _render_args(p):
    p.add_argument("program", help="a tatbot-program from `tatbot ros compile`")
    p.add_argument("--substrate", default="silicone", choices=SUBSTRATES)
    p.add_argument("--spread-mm", default="0", help="ink spread blur for skin; larger previews ageing")
    p.add_argument("--out", required=True, help="output directory for render.png and render.json")


@verb(noun="art", verb="render", tier=OFFLINE, visibility="public", audience="primary",
      disposition="judges a colour drawing and its minutes without the robot",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="predicted appearance of a program as ink on a substrate, with dips, rinses and minutes",
      args=_render_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/render.py",
                                "python/tatbot_art/src/tatbot_art/raster.py",
                                "python/tatbot_art/src/tatbot_art/timing.py"), doc="docs/art.md",
      example=("/tmp/out/program.json", "--substrate", "skin-3", "--out", "/tmp/render"),
      invariants=("A prediction labelled with its weakest ink evidence, not the compile preview.",
                  "Minutes fill missing compiler costs with historical dip gaps and rinse/swap allowances."))
def art_render(ctx, ns, rest):
    return sh(ctx, SCRIPT, "render", ns.program, "--substrate", ns.substrate, "--spread-mm", ns.spread_mm,
              "--out", ns.out, *rest)


def _profile_args(p):
    p.add_argument("program", help="an original compiled program with its recorded duration estimate")
    p.add_argument("--out", required=True, help="directory for profile.html and profile.json")
    for kind in ("dip", "rinse", "swap"):
        p.add_argument(f"--{kind}-s", help=f"{kind} gap in seconds, overriding the historical allowance")


@verb(noun="art", verb="profile", tier=OFFLINE, visibility="public", audience="primary",
      disposition="explains what makes a saved design slow, without the rig",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="drawing, descent, lift, travel, dipping and rinse time from a saved compiled program",
      args=_profile_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/profile.py",
                                 "python/tatbot_art/src/tatbot_art/timing.py"), doc="docs/art.md",
      example=("/tmp/out/program.json", "--out", "/tmp/time-profile"),
      invariants=("Uses the program's recorded compiler estimate; no motion planning or rig access.",
                  "Fixed allowances fill missing workflow costs once; initial and same-tool/ink selections are free.",
                  "The report separates compiler estimates, historical gaps and user-supplied allowances."))
def art_profile(ctx, ns, rest):
    flags = [s for k in ("dip", "rinse", "swap") if getattr(ns, f"{k}_s") is not None
             for s in (f"--{k}-s", getattr(ns, f"{k}_s"))]
    return sh(ctx, SCRIPT, "profile", ns.program, "--out", ns.out, *flags, *rest)


def _separate_args(p):
    p.add_argument("image", nargs="?",
                   help="a flat-colour source image (PNG or JPEG); a tatbot.art-generation/1 sidecar beside it is read")
    p.add_argument("--inks", help="comma-separated catalogue ink ids, at most five (with --brief: the art's own)")
    p.add_argument("--width-mm", help="drawn width; the height follows the image (with --brief: the placed one)")
    p.add_argument("--fit", action="store_true",
                   help="instead of --width-mm, the widest the page holds: --stencil's print at --at, else the "
                        "nominal; on a print with markers, the widest that also keeps its marker budget")
    p.add_argument("--allow-marker-loss", action="store_true",
                   help="draw over the print's markers past the marker budget (a sheet fixed on its mount); with "
                        "--fit, keep the widest the page holds")
    p.add_argument("--brief", help="an Inkmap tattoo brief (tatbot.inkmap-brief/1): its image at its placed width")
    p.add_argument("--placement", help="with --brief, one placement; without it, every placement in drawing order")
    p.add_argument("--style", choices=("flash", "geometric", "linocut"),
                   help="default flash, or the art's own with --brief")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--app", help="activated DBV3 Premium 1.6.22; without it the jobs are only planned")
    p.add_argument("--substrate", default="silicone", choices=SUBSTRATES)
    p.add_argument("--no-mixes", action="store_true", help="single-ink tints only")
    p.add_argument("--fill", default="packed",
                   help="DBV3 texture for regions an ink covers fully, or offset: passes one mark apart along their "
                        "edges, made without DBV3; the art CLI lists the choices")
    p.add_argument("--tool", action="append", metavar="INK=TOOL",
                   help="the cartridge an ink is drawn with (default: the fitted one); another is a manual exchange")
    p.add_argument("--pitch-mm", action="append", metavar="INK=MM",
                   help="with --fill offset: an ink's passes this far apart (default: its cartridge's mark)")
    p.add_argument("--overlap-mm", help="with --fill offset: a lighter ink reaches this far under the inks drawn after it; negative "
                                          "(--overlap-mm=-0.3) separates for the darkest ink first, lighter inks kept that far clear")
    p.add_argument("--stencil", help="passed to tatbot ros compile")
    p.add_argument("--at", help="passed to tatbot ros compile: canvas centre X,Y in mm")
    p.add_argument("--tone", help="tone curve from `tatbot art tone` (default: the measured 0.6 mm one)")


@verb(noun="art", verb="separate", tier=OFFLINE, visibility="public", audience="primary",
      disposition="turns a generated image into the multi-ink DBV3 layers, program and render it would draw",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="a flat-colour image into per-ink DBV3 layers, a compiled program and a predicted render",
      args=_separate_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/separate.py",
                                  "python/tatbot_art/src/tatbot_art/acquire.py",
                                  "python/tatbot_art/src/tatbot_art/offset.py",
                                  "python/tatbot_art/src/tatbot_art/pipeline.py",
                                  "python/tatbot_art/src/tatbot_art/brief.py"), doc="docs/art.md",
      example=("flash.png", "--inks", "golden_yellow,bright_red,light_green,blue_sky,true_black", "--width-mm", "60",
               "--out", "/tmp/sep", "--app", "/opt/drawingbotv3"),
      invariants=("DBV3 makes every path but the offset fill's (tatbot_art.offset); the separation writes their sources.",
                  "With --brief the width is the one placed in Inkmap; the crop must match the brief's art_px.",
                  "With --fit the width is the widest whose canvas the page holds; the design is never turned.",
                  "A design that cannot fit its page is refused before DBV3 runs.",
                  "The inks file is a draft: its caps are assumed, not read from the palette."))
def art_separate(ctx, ns, rest):
    args = ["separate", *([ns.image] if ns.image else []), "--out", ns.out, "--substrate", ns.substrate,
            "--fill", ns.fill]
    for name in ("inks", "width_mm", "brief", "placement", "style", "app", "stencil", "tone", "overlap_mm"):
        if value := getattr(ns, name):
            args += [f"--{name.replace('_', '-')}", value]
    for name in ("tool", "pitch_mm"):
        for value in getattr(ns, name) or []:
            args += [f"--{name.replace('_', '-')}", value]
    if ns.at:
        args += [f"--at={ns.at}"]
    for flag in ("no_mixes", "fit", "allow_marker_loss"):
        if getattr(ns, flag):
            args.append(f"--{flag.replace('_', '-')}")
    return sh(ctx, SCRIPT, *args, *rest)


def _tiles_args(p):
    p.add_argument("ids", nargs="*", help="the round's designs to take (default: all of them)")
    p.add_argument("--plan", required=True, help="the round's Flow prompt plan, from tatbot art propose --tile")
    p.add_argument("--corpus", required=True,
                   help="the art corpus: the round's flow/ images; their separations go to separations/<id>/")
    p.add_argument("--stencil", required=True, help="a generated print of the tile the round was composed for")
    p.add_argument("--app", help="activated DBV3 Premium 1.6.22; without it the designs are only planned")
    p.add_argument("--substrate", default="silicone", choices=SUBSTRATES)
    p.add_argument("--out", required=True, help="the index, tatbot.art-tiles/1 JSON")


@verb(noun="art", verb="tiles", tier=OFFLINE, visibility="public", audience="advanced",
      disposition="separates a round composed for a small stencil tile at that tile's size, one program a tile",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="a round composed for a stencil tile, each design separated at its size and compiled onto the print",
      args=_tiles_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/tiles.py",
                               "python/tatbot_art/src/tatbot_art/pipeline.py",
                               "python/tatbot_art/src/tatbot_art/loop.py"), doc="docs/art.md",
      example=("--plan", "/tmp/corpus/flow/prompts-r6.json", "--corpus", "/tmp/corpus", "--stencil",
               "/tmp/stencil-forearm18", "--out", "/tmp/corpus/tiles-r6.json", "--app", "/opt/drawingbotv3"),
      invariants=("Only a round proposed with --tile, on a print of that tile: no design is fitted down to one.",
                  "Each design is `tatbot art separate --fit --stencil`: the widest it fits, compiled at the centre.",
                  "One design that fails does not stop the rest; the index lists it and the command exits nonzero.",
                  "Writes files only; drawing stays an explicit tatbot ros draw."))
def art_tiles(ctx, ns, rest):
    args = ["tiles", *ns.ids, "--plan", ns.plan, "--corpus", ns.corpus, "--stencil", ns.stencil,
            "--substrate", ns.substrate, "--out", ns.out, *(["--app", ns.app] if ns.app else [])]
    return sh(ctx, SCRIPT, *args, *rest)


def _stock_render_args(p):
    p.add_argument("--app", required=True, help="activated DBV3 Premium 1.6.22")
    p.add_argument("--widths", default="45,60", help="comma-separated drawn widths, mm")
    p.add_argument("--out", required=True, help="working directory: each separation, kept for review")


@verb(noun="art", verb="stock-renders", tier=OFFLINE, visibility="public", audience="primary",
      disposition="remakes the robot renders Inkmap shows for its stock designs",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="Inkmap's robot renders of its stock designs, each through DBV3 at its width",
      args=_stock_render_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/brief.py",
                                      "python/tatbot_art/src/tatbot_art/pipeline.py"), doc="docs/art.md",
      example=("--app", "/opt/drawingbotv3", "--out", "/tmp/stock-renders"),
      invariants=("Each render binds its art by digest; Inkmap's tests fail on one the collection has moved past.",
                  "Writes web/inkmap/public/designs/renders/ and the manifest's render list, nothing else in the tree."))
def art_stock_renders(ctx, ns, rest):
    return sh(ctx, SCRIPT, "stock-renders", "--app", ns.app, "--widths", ns.widths, "--out", ns.out, *rest)


def _tone_args(p):
    p.add_argument("--app", required=True, help="activated DBV3 Premium 1.6.22")
    p.add_argument("--width-mm", default="0.6", help="pen width the curve is for")
    p.add_argument("--out", required=True, help="output directory for tone.json")


@verb(noun="art", verb="tone", tier=OFFLINE, visibility="public", audience="advanced",
      disposition="re-measures the hatch fill's grey-to-coverage curve when DBV3 or the pen width changes",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="measure the hatch fill's grey-to-coverage response with a DBV3 step wedge",
      args=_tone_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/tone.py"), doc="docs/art.md",
      example=("--app", "/opt/drawingbotv3", "--out", "/tmp/tone"))
def art_tone(ctx, ns, rest):
    return sh(ctx, SCRIPT, "tone", "--app", ns.app, "--width-mm", ns.width_mm, "--out", ns.out, *rest)


def _shade_args(p):
    p.add_argument("image", help="a tone image: darker is more ink, white and transparent draw nothing")
    p.add_argument("--width-mm", required=True, help="drawn width; the height follows the image")
    p.add_argument("--technique", default="whip", choices=("whip", "pendulum"),
                   help="whip: a stroke per run of tone; pendulum: runs joined into zigzags")
    p.add_argument("--angle-deg", help="stroke direction, dark to light, CCW from +x (default: the tone's descent)")
    p.add_argument("--tool", default="lutin-3rl-bugpin", help="the cartridge's datasheet id: its mark's width")
    p.add_argument("--row-deg", help="a magnum's needle row on the page, CCW from +x (default: along the print's x)")
    p.add_argument("--pitch-mm", help="between neighbouring strokes (default: the mark's width)")
    p.add_argument("--ink", default="true_black", help="catalogue ink id the render draws with")
    p.add_argument("--substrate", default="silicone", choices=SUBSTRATES)
    p.add_argument("--out", required=True,
                   help="output directory for render.png, artwork.json, preview-program.json and shade.json")


@verb(noun="art", verb="shade", tier=OFFLINE, visibility="public", audience="advanced",
      disposition="designs whip or pendulum shading and judges it before the robot can draw deposition",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="a tone image as whip strokes or pendulum zigzags whose deposition fades with the tone, rendered",
      args=_shade_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/shade.py",
                               "python/tatbot_art/src/tatbot_art/cartridge.py",
                               "python/tatbot_art/src/tatbot_art/render.py",
                               "ros/tatbot_motion/tatbot_motion/estimate.py"), doc="docs/art.md",
      example=("/tmp/tone.png", "--width-mm", "30", "--technique", "pendulum", "--tool", "lutin-9m-ex",
               "--out", "/tmp/shade"),
      invariants=("artwork.json compiles with `tatbot ros compile` for a tool whose datasheet records "
                  "fade_lift_mm; preview-program.json only previews and times the strokes.",
                  "Deposition never rises along a stroke faster than 1 per --rise-mm (default 4 mm)."))
def art_shade(ctx, ns, rest):
    optional = [arg for name in ("angle_deg", "row_deg", "pitch_mm") if getattr(ns, name) is not None
                for arg in (f"--{name.replace('_', '-')}", getattr(ns, name))]
    return sh(ctx, SCRIPT, "shade", ns.image, "--width-mm", ns.width_mm, "--technique", ns.technique,
              "--tool", ns.tool, "--ink", ns.ink, "--substrate", ns.substrate, "--out", ns.out, *optional, *rest)


def _wall_args(p):
    p.add_argument("--round", required=True, help="the round these designs belong to")
    p.add_argument("--plan", required=True, help="the round's Flow prompt plan (prompt ids and ink loads)")
    p.add_argument("--corpus", required=True, help="the art corpus: flow/ images and separations/")
    p.add_argument("--assets", required=True, help="the wall's uploaded asset ids, {assets: {<image>-<view>: id}}")
    p.add_argument("--stencil-previews", help="export scale previews for recorded stencil placements to this directory")
    p.add_argument("--out", required=True, help="the designs as {id: document}, for the wall's database")


@verb(noun="art", verb="wall", tier=OFFLINE, visibility="public", audience="advanced",
      disposition="turns a round of generated designs into the voting wall's documents",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="a round's designs as voting-wall documents: genes, views, minutes and fidelity",
      args=_wall_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/loop.py"), doc="docs/art.md",
      example=("--round", "1", "--plan", "/tmp/corpus/flow/prompts.json", "--corpus", "/tmp/corpus", "--assets",
               "/tmp/corpus/wall/assets.json", "--out", "/tmp/r1-designs.json"))
def art_wall(ctx, ns, rest):
    preview = ["--stencil-previews", ns.stencil_previews] if ns.stencil_previews else []
    return sh(ctx, SCRIPT, "wall", "--round", ns.round, "--plan", ns.plan, "--corpus", ns.corpus, "--assets", ns.assets,
              "--out", ns.out, *preview, *rest)


def _propose_args(p):
    p.add_argument("--designs", required=True, help="the wall's designs, {id: document}")
    p.add_argument("--votes", required=True, help="the wall's votes, {viewer: {design: +1 or -1}}")
    p.add_argument("--round", required=True, help="the round being proposed")
    p.add_argument("--n", default="9", help="prompts, three images each")
    p.add_argument("--explore", default="0.34", help="share of prompts on untried genes, 0 to 1")
    p.add_argument("--seed", default="0", help="the draw; try several and keep the plan covering the most styles and fills")
    p.add_argument("--asks", help="the wall's asks, {id: document}; open ones take the first prompts")
    p.add_argument("--tile", help="compose every design for this stencil tile (config/stencil-tiles.json), e.g. forearm18")
    p.add_argument("--out", required=True, help="the next round's Flow prompt plan")


@verb(noun="art", verb="propose", tier=OFFLINE, visibility="public", audience="advanced",
      disposition="the voting loop's next step: votes into the next round's prompts",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="the next round's Flow prompts from the wall's votes: the best genes, plus some untried ones",
      args=_propose_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/loop.py"), doc="docs/art.md",
      example=("--designs", "/tmp/designs.json", "--votes", "/tmp/votes.json", "--round", "2", "--out", "/tmp/r2.json"))
def art_propose(ctx, ns, rest):
    optional = [arg for name in ("asks", "tile") if getattr(ns, name) for arg in (f"--{name}", getattr(ns, name))]
    return sh(ctx, SCRIPT, "propose", "--designs", ns.designs, "--votes", ns.votes, "--round", ns.round, "--n", ns.n,
              "--explore", ns.explore, "--seed", ns.seed, *optional, "--out", ns.out, *rest)


def _map_args(p):
    p.add_argument("--designs", required=True, help="the wall's designs, {id: document}")
    p.add_argument("--votes", help="the wall's votes, to score each neighbourhood")
    p.add_argument("--corpus", required=True, help="the art corpus: flow/ images")
    p.add_argument("--out", required=True, help="the map document for the wall's database")


@verb(noun="art", verb="map", tier=OFFLINE, visibility="public", audience="advanced",
      disposition="lays the wall out as a browsable map of neighbourhoods, for asking for more of one",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="the voting wall's map: designs placed by genes and look, clustered, linked to their nearest",
      args=_map_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/atlas.py",
                             "python/tatbot_art/src/tatbot_art/pyramid.py"), doc="docs/art.md",
      example=("--designs", "/tmp/designs.json", "--corpus", "/tmp/corpus", "--out", "/tmp/map.json"))
def art_map(ctx, ns, rest):
    votes = ["--votes", ns.votes] if ns.votes else []
    return sh(ctx, SCRIPT, "map", "--designs", ns.designs, *votes, "--corpus", ns.corpus, "--out", ns.out, *rest)


def _promote_args(p):
    p.add_argument("--designs", required=True, help="the wall's designs, {id: document}")
    p.add_argument("--votes", required=True, help="the wall's vote documents, {viewer: {v, s}}")
    p.add_argument("--corpus", required=True, help="the art corpus: flow/ images and separations/")
    p.add_argument("--budget", default="40", help="DBV3 separations this pass may spend")
    p.add_argument("--fills", help="comma-separated fills a study tries (default: the contained ones)")
    p.add_argument("--app", help="activated DBV3 Premium 1.6.22; without it the pass is only planned")
    p.add_argument("--out", required=True, help="the pass's work list and tier counts")


@verb(noun="art", verb="promote", tier=OFFLINE, visibility="public", audience="advanced",
      disposition="spends DBV3 time only where votes earned it: kept sketches drawn, kept drawings fill-studied",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="the design pyramid's next pass: draw kept sketches, study kept drawings in every fill",
      args=_promote_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/pyramid.py",
                                 "python/tatbot_art/src/tatbot_art/pipeline.py"), doc="docs/art.md",
      example=("--designs", "/tmp/designs.json", "--votes", "/tmp/votes.json", "--corpus", "/tmp/corpus", "--out",
               "/tmp/promote.json"),
      invariants=("Only votes promote; the budget is shared round-robin across styles.",
                  "A study variant reuses its source's acquisitions and re-acquires only the solid layers."))
def art_promote(ctx, ns, rest):
    args = ["promote", "--designs", ns.designs, "--votes", ns.votes, "--corpus", ns.corpus, "--budget", ns.budget,
            "--out", ns.out]
    for name in ("fills", "app"):
        if value := getattr(ns, name):
            args += [f"--{name}", value]
    return sh(ctx, SCRIPT, *args, *rest)


def _bundle_args(p):
    p.add_argument("ids", nargs="*", help="design ids: separations/<id> in the corpus")
    p.add_argument("--corpus", required=True, help="the art corpus: flow/ images and separations/")
    p.add_argument("--votes", help="the wall's vote documents; adds every starred design")
    p.add_argument("--kept", action="store_true", help="with --votes, also every design with a positive vote total")
    p.add_argument("--designs", help="the wall's designs, {id: document}, for genes and round")
    p.add_argument("--at", action="append", default=[],
                   help="the design's centre on the page, X,Y mm (x right, y toward the top); ID=X,Y poses one "
                        "design, so several share one page (repeatable)")
    p.add_argument("--stencil", help="the generated print the design is drawn on (passed to tatbot ros compile)")
    p.add_argument("--substrate", default="silicone", choices=SUBSTRATES, help="the render's substrate")
    p.add_argument("--ink-substrate", choices=("silicon_skin", "human_skin"),
                   help="the tatbot-inks/3 substrate (default: the separation's draft)")
    p.add_argument("--one-load", action="store_true",
                   help="give every design the same caps, by their combined line, so none is swapped between them")
    p.add_argument("--out", required=True, help="one directory per design, and bundles.json")


@verb(noun="art", verb="bundle", tier=OFFLINE, visibility="public", audience="advanced",
      disposition="hands a shortlisted design to review and the rig: its layers, inks, program, render, minutes",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="drawn designs as reviewable bundles: the acquired layers, their ink loads, program and render",
      args=_bundle_args, wraps=(*WRAPS, "python/tatbot_art/src/tatbot_art/bundle.py",
                                "python/tatbot_art/src/tatbot_art/acquire.py"), doc="docs/art.md",
      example=("--votes", "/tmp/votes.json", "--corpus", "/tmp/corpus", "--out", "/tmp/bundles"),
      invariants=("No DBV3: a bundle reuses the separation's acquisitions, so a sketch waits for promote.",
                  "Every layer shares one origin on the page; nothing is turned, rescaled or re-acquired.",
                  "Writes files only; drawing stays an explicit tatbot ros draw."))
def art_bundle(ctx, ns, rest):
    args = ["bundle", *ns.ids, "--corpus", ns.corpus, "--substrate", ns.substrate, "--out", ns.out]
    for name in ("votes", "designs", "stencil", "ink_substrate"):
        if value := getattr(ns, name):
            args += [f"--{name.replace('_', '-')}", value]
    for pose in ns.at:
        args.append(f"--at={pose}")   # = keeps a negative X,Y from reading as an option
    for flag in ("kept", "one_load"):
        if getattr(ns, flag):
            args.append(f"--{flag.replace('_', '-')}")
    return sh(ctx, SCRIPT, *args, *rest)
