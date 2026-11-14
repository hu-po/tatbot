"""The voting loop: designs on the wall, votes back, the next round's prompts.

Each design carries genes, the choices that made it: style, subject, palette (the ink load) and variant
(a prompt modifier). A vote is +1 (keep) or -1 (drop). A gene's score is the sum of its designs' votes
shrunk toward zero, (sum) / (count + PRIOR), so one lucky vote does not decide a direction. The next
round keeps most of its prompts on the best-scoring genes and spends the rest on genes not tried yet.

The owner can also ask for more of something: more like one design, its subject in other styles, two
designs blended, or more of one neighbourhood of the map. An open ask takes the first prompts of the next
round. Every prompt names the kept designs it builds on, its `parents`, which the map draws as lineage.

A round may be composed for a stencil tile (`tile_round`): every prompt asks for a small tattoo on the tile, mostly
outline and bare skin, its load at most TILE_INKS inks and its style one the tile can draw, so the design is made for
that size rather than shrunk to it.

The wall (python/tatbot_art/gallery.html) keeps designs, rounds, votes and asks in its database. The agent
running the loop moves them in and out with the Artifact tools; this module only reads and writes JSON.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

from .acquire import REPO
from .inks import look

PRIOR = 2.0
STYLE_OF_PREFIX = {"trad": "american-traditional", "lino": "linocut", "geo": "geometric"}
# The tested Flow templates from the style skills; {palette} lists the inks with their hex values.
TEMPLATES = {
    "american-traditional": (
        "American traditional tattoo flash of {subject}. Bold, even black outlines of a single weight, flat solid "
        "colour fills with no gradients, no blending and no soft shading; shadows are solid black shapes. Use only "
        "these exact colours and no others: {palette}. Large simple shapes with generous gaps between lines, no "
        "tiny details. Isolated on a plain white background, centered, no text, no lettering, no signs, no border."),
    "linocut": (
        "Linocut relief print of {subject}, designed as a tattoo. A carved black ink mass with a few bold white "
        "gouge lines, each at least a millimetre wide at tattoo size, describing form, plus flat second colour "
        "blocks printed under the black. Only these inks: {palette}. Bold and high contrast, no gradients, no grey, "
        "no halftone dots. Flat digital artwork, not a photograph: pure white background, no paper texture, no "
        "shadow, no frame, no signs, no text, no border. Centered, isolated."),
    "geometric": (
        "Geometric tattoo design of {subject} constructed on a seed-of-life circle grid. The subject's silhouette is "
        "bold and clearly readable; the grid stays light behind it. Precise black linework of one weight, compass "
        "and straight-edge construction circles kept as part of the design; flat fills in only these colours: "
        "{palette}, inside some of the cells. No gradients, no shading, no texture. Plain white background, "
        "centered, isolated, no text."),
    "neo-traditional": (
        "Neo-traditional tattoo design of {subject}: bold black outlines with varied line weight, ornate details and "
        "a decorative frame of leaves or filigree, jewel-toned flat colour fills from only {palette}. No gradients, no "
        "soft shading: shadows are solid shapes or clean hatching. Flat digital artwork on a pure white background, "
        "centered, isolated, no text."),
    "blackwork": (
        "Blackwork tattoo design of {subject}: heavy solid black areas and crisp negative-space shapes, a bold graphic "
        "silhouette, high contrast, inks limited to {palette}. No grey, no gradients, no fine texture. Flat digital "
        "artwork, pure white background, centered, isolated, no text."),
    "fineline": (
        "Fine-line tattoo design of {subject}: delicate single-weight thin black linework, minimal detail, elegant "
        "contours, small flat accents of colour from only {palette}, lots of negative space. No shading, no gradients. "
        "Flat digital artwork, pure white background, centered, isolated, no text."),
    "engraving": (
        "Antique engraving style tattoo of {subject}: a line-engraved etching where parallel hatching and "
        "cross-hatching build every tone, like an old book illustration, inks limited to {palette}. No grey wash, no "
        "gradients. Flat digital artwork, pure white background, centered, isolated, no text."),
    "risograph": (
        "Risograph print style illustration of {subject}: two or three flat spot colours from {palette}, overprinted "
        "with slight misregistration where they overlap, bold simple shapes, no outlines. No gradients, no halftone "
        "dots, no paper texture. Flat digital artwork, pure white background, centered, isolated, no text."),
    "stained-glass": (
        "Stained-glass window design of {subject}: thick black leading lines dividing the picture into flat colour "
        "panes, each pane one solid colour from {palette}, bold simple shapes. No gradients, no texture. Flat digital "
        "artwork, pure white background, centered, isolated, no text."),
    "art-nouveau": (
        "Art nouveau tattoo design of {subject}: flowing whiplash curves, elegant black outlines, an ornamental arched "
        "frame with stylised botanical motifs, flat colour fills from {palette}. No gradients. Flat digital artwork, "
        "pure white background, centered, isolated, no text."),
    "woodblock": (
        "Woodblock print style tattoo of {subject}: bold black key lines, flat areas of colour from {palette}, stylised "
        "waves, clouds or wind lines framing the subject, graphic and decorative. No gradients, no paper texture. Flat "
        "digital artwork, pure white background, centered, isolated, no text."),
    "comic": (
        "Bold comic-book style tattoo of {subject}: thick black ink outlines, dynamic shapes, flat cel colours from "
        "{palette}, solid black shadows. No halftone dots, no gradients, no speech bubbles. Flat digital artwork, pure "
        "white background, centered, isolated, no text."),
    "color-block": (
        "Minimal tattoo of {subject}: a clean thin black line drawing laid over loose abstract blobs of flat colour "
        "from {palette} that sit slightly offset from the lines. No gradients, no shading. Flat digital artwork, pure "
        "white background, centered, isolated, no text."),
    "dotwork": (
        "Dotwork tattoo design of {subject}: crisp black outlines with all shading built from stippled dots that grow "
        "denser into the shadows, ornamental mandala or sacred-geometry accents, small flat accents of colour from "
        "only {palette}. No grey wash, no gradients. Flat digital artwork, pure white background, centered, isolated, "
        "no text."),
    "trash-polka": (
        "Trash polka tattoo design of {subject}: a collage of a bold graphic subject, heavy black brush strokes, "
        "smears and abstract geometric shapes, with only the inks {palette}; the brightest colour splashed in big "
        "flat strokes. Raw and high contrast, no gradients, no grey. Flat digital artwork, pure white background, "
        "centered, isolated, no text."),
    "ignorant": (
        "Ignorant-style tattoo of {subject}: a naive, deliberately clumsy hand-drawn doodle with a single wobbly "
        "black line weight, childlike proportions and charm, a few flat fills from {palette}. No shading, no "
        "gradients. Flat digital artwork, pure white background, centered, isolated, no text."),
    "psychedelic": (
        "1960s psychedelic concert-poster style design of {subject}: swirling melting curves, the subject's outline "
        "echoed in concentric bands, vibrating flat colours from only {palette}, bold black outlines. No gradients, "
        "no lettering. Flat digital artwork, pure white background, centered, isolated, no text."),
    "pop-art": (
        "Pop art style tattoo of {subject}: thick black comic outlines, flat saturated colours from only {palette}, "
        "bold graphic shine marks and a burst shape behind the subject. No halftone dots, no gradients. Flat digital "
        "artwork, pure white background, centered, isolated, no text."),
    "sticker": (
        "Cute kawaii sticker design of {subject}: chunky rounded shapes, a thick uniform black outline around "
        "everything, simple flat colour fills from only {palette}, a tiny highlight or two, a cheerful face where it "
        "fits. No gradients, no drop shadow. Flat digital artwork, pure white background, centered, isolated, no "
        "text."),
    "celtic-knotwork": (
        "Celtic knotwork tattoo design of {subject}: the subject formed from interlaced over-under ribbon bands with "
        "black outlines, knot borders and spirals, flat colour fills inside the ribbons from only {palette}. No "
        "gradients, no shading. Flat digital artwork, pure white background, centered, isolated, no text."),
    "watercolor": (
        "Watercolour tattoo design of {subject}: a confident thin black line drawing with soft transparent "
        "watercolour washes and splashes in only {palette}, bleeding a little past the lines, lighter and darker "
        "tints of each colour. No black wash. Flat digital artwork, pure white background, centered, isolated, no "
        "text."),
    "bauhaus": (
        "Bauhaus poster style design of {subject}, built only from circles, half-circles, rectangles and triangles, "
        "flat solid colours from {palette} and solid black, strong asymmetric balance, no outlines. No gradients, no "
        "texture. Flat digital artwork, pure white background, centered, isolated, no text."),
}
# How `tatbot art separate` treats each style's black: traced and packed (flash), drawn once as lines
# (geometric), or a mass whose edge is its outline (linocut).
SEPARATION = {"american-traditional": "flash", "neo-traditional": "flash", "stained-glass": "flash",
              "art-nouveau": "flash", "woodblock": "flash", "comic": "flash", "geometric": "geometric",
              "fineline": "geometric", "engraving": "geometric", "color-block": "geometric", "linocut": "linocut",
              "blackwork": "linocut", "risograph": "linocut", "dotwork": "geometric", "trash-polka": "linocut",
              "ignorant": "geometric", "psychedelic": "flash", "pop-art": "flash", "sticker": "flash",
              "celtic-knotwork": "flash", "watercolor": "geometric", "bauhaus": "linocut"}
# The solid fills whose line stays inside its own regions: at most 0.5% of line more than 0.4 mm outside its
# layer's source over every corpus acquisition (the tatbot-dbv3-recipes table). The other acquire.SOLID_FILLS
# textures (curves, superformula, waves, sweep, maze) cross into neighbouring regions until a layer is
# clipped to its own, so the loop neither proposes nor studies them.
FILLS = ["packed", "contour", "flow", "squares"]
COLOUR_INKS = ["rose_pink", "bright_red", "bright_orange", "golden_yellow", "lemon_yellow", "lime_green", "light_green",
               "blue_sky", "light_purple", "brown", "silver"]
SUBJECTS = [
    "a sea lion barking on a rock", "a San Francisco cable car", "a swallow over California poppies",
    "an old shipyard gantry crane at a pier", "a Golden Gate Bridge tower in fog", "a Dungeness crab",
    "a sea otter floating on its back holding a shell", "a hummingbird at a flower", "a pelican diving",
    "a Victorian row house", "a sourdough loaf with wheat", "a robot arm holding a tattoo machine",
    "a koi fish", "a tiger head", "a luna moth", "a rose and dagger", "a lighthouse in a storm", "an octopus",
    "a mushroom cluster", "a skull crowned with flowers", "a sun and moon face", "a fox curled asleep",
    "a cactus in a pot", "a vintage rocket ship", "an astronaut floating", "a whale tail above a wave",
    "a bonsai tree", "a coffee cup with steam", "a snake coiled around a dagger", "a hand holding an eye",
    "a jellyfish trailing tentacles", "a heron standing in reeds", "an anatomical heart",
    "a dragon coiled in clouds", "a phoenix rising", "a black cat with a crescent moon", "a bee on a honeycomb",
    "a lotus flower", "a ship in a bottle", "an owl on a branch", "a Muni streetcar", "a hot air balloon",
    "a quadruped robot dog",
]
PALETTES = {  # every load ends in black; at most five inks
    "trad-primaries": ["golden_yellow", "bright_red", "light_green", "blue_sky", "true_black"],
    "warm": ["lemon_yellow", "golden_yellow", "bright_orange", "bright_red", "true_black"],
    "cool": ["lime_green", "light_green", "blue_sky", "light_purple", "true_black"],
    "orange-violet": ["bright_orange", "light_purple", "true_black"],
    "red-green": ["bright_red", "light_green", "true_black"],
    "sky-gold": ["blue_sky", "golden_yellow", "true_black"],
    "earth": ["brown", "golden_yellow", "bright_orange", "true_black"],
    "pink-violet": ["rose_pink", "light_purple", "true_black"],
    "red-black": ["bright_red", "true_black"],
    "orange-black": ["bright_orange", "true_black"],
    "sky-black": ["blue_sky", "true_black"],
}


# A tile round's limits, from the style skills' 35 mm budget: black and one or two colours, and the traditional style's
# 25 minutes at 35 mm, scaled by area as the skill scales it. The first tile round (forearm18) modelled about 9 minutes
# per 100 mm2 inked, so that budget is roughly a third of the tile inked.
TILE_INKS = 3
BUDGET_35MM_MIN = 25.0
# Styles a tile cannot draw, whatever the votes: the reason is a capability, never taste.
TILE_AVOID_STYLES = {"dotwork": "programs have no dot, so stippled shading drops out at this size"}
# Flow's image aspects (2026-11-08), width over height
FLOW_ASPECTS = {"16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0, "3:4": 3 / 4, "9:16": 9 / 16}


def tile_round(layout: str) -> dict:
    """What a round composed for a stencil tile asks of Flow: the tile's size and how little of it to ink in each
    prompt, the aspect nearest the tile's, at most TILE_INKS inks a load, no style the tile cannot draw, and the
    minutes a design may model before `tatbot art tiles` flags it. `config/stencil-tiles.json` gives the tile."""
    tiles = json.loads((REPO / "config/stencil-tiles.json").read_text())["tiles"]
    if layout not in tiles:
        raise ValueError(f"unknown tile {layout!r}; have {', '.join(tiles)}")
    w, h = tiles[layout]["tile_mm"]
    aspect = min(FLOW_ASPECTS, key=lambda a: abs(math.log(FLOW_ASPECTS[a] * h / w)))
    shape = "upright" if h > w else "wide" if w > h else "square"
    sentence = (f"Designed as a small tattoo inside an {shape} {w:g} by {h:g} mm area, mostly outline and bare skin: "
                "ink covers no more than a third of it, colour sits in a few small shapes, and there is no filled "
                "background, panel or frame. One bold subject of a few large shapes and thick lines, nothing finer "
                "than 2 mm and no gap narrower than 1 mm at that size.")
    return {"layout": layout, "tile_mm": [w, h], "aspect": aspect, "max_inks": TILE_INKS,
            "avoid_styles": dict(TILE_AVOID_STYLES), "minutes_budget": round(BUDGET_35MM_MIN * w * h / 35 ** 2),
            "sentence": sentence}


def fits_tile(tile: dict | None, style: str, inks: list[str]) -> bool:
    """Whether a prompt can be composed for the tile: its load within the tile's inks, its style one the tile draws."""
    return tile is None or (len(inks) <= tile["max_inks"] and style not in tile.get("avoid_styles", {}))


def palette_key(inks: list[str]) -> str:
    return next((k for k, v in PALETTES.items() if v == list(inks)), "+".join(inks))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def measured(separation: dict | None) -> dict:
    """What an acquired separation measured, for the wall; nothing for a plan-only one."""
    if not separation or "time" not in separation:
        return {}
    return {"minutes": separation["time"]["minutes"], "dips": separation["time"]["dips"],
            "colours": len(separation["colours"]), "mixes": sum(len(c["layers"]) > 1 for c in separation["colours"]),
            "fidelity": separation["fidelity"]["vs_target"]["mean"]}


def wall_design(image: Path, sidecar: dict, inks: list[str], catalog: dict, assets: dict, round_no: int,
                order: int, separation: dict | None, fill: str = "packed", parents: list[str] | None = None,
                separation_style: str | None = None) -> dict:
    """One wall document: genes, the three views' asset urls, lineage, and what the separation measured."""
    did = image.stem
    views = {v: f"/_blob/{assets[f'{did}-{v}']}" for v in ("flow", "ours", "drawn") if f"{did}-{v}" in assets}
    prefix = (sidecar.get("prompt_id") or did).split("-")[0]
    style = sidecar.get("style") or STYLE_OF_PREFIX.get(prefix, prefix)
    doc = {"round": round_no, "order": order, "style": style, "subject": sidecar.get("subject") or did,
           "prompt": sidecar.get("prompt"), "variant": sidecar.get("variant"), "views": views,
           "separation": separation_style or SEPARATION.get(style, "flash"), "parents": list(parents or []),
           "inks": [{"id": i, "name": look(catalog, i).name,
                     "hex": "#%02x%02x%02x" % tuple(round(c * 255) for c in look(catalog, i).srgb)} for i in inks],
           "genes": {"style": style, "subject": sidecar.get("subject") or did, "palette": palette_key(inks),
                     "variant": sidecar.get("variant") or "base", "fill": fill}}
    doc["fill"] = fill
    doc.update(measured(separation))
    return doc


def ballots(votes: dict) -> dict[str, dict[str, int]]:
    """{viewer: {design: ±1}} from either that shape or the wall's vote documents, {viewer: {v, s, at}}."""
    out = {}
    for viewer, doc in votes.items():
        doc = doc or {}
        out[viewer] = doc["v"] if isinstance(doc.get("v"), dict) else doc
    return out


def votes_by_design(votes: dict) -> dict[str, int]:
    """Sum every viewer's votes per design: {viewer: {design: ±1}} -> {design: total}."""
    out: dict[str, int] = {}
    for per_viewer in ballots(votes).values():
        for did, v in per_viewer.items():
            out[did] = out.get(did, 0) + int(v)
    return out


def gene_scores(designs: dict[str, dict], votes: dict) -> dict[str, dict[str, dict]]:
    """{gene: {value: {score, sum, n}}} over voted designs only."""
    totals = votes_by_design(votes)
    acc: dict[str, dict[str, list[float]]] = {}
    for did, total in totals.items():
        d = designs.get(did)
        if d is None or not total:
            continue
        for gene, value in d.get("genes", {}).items():
            acc.setdefault(gene, {}).setdefault(value, []).append(total)
    return {gene: {value: {"score": round(sum(vs) / (len(vs) + PRIOR), 3), "sum": sum(vs), "n": len(vs)}
                   for value, vs in by_value.items()} for gene, by_value in acc.items()}


def _pick(rng: random.Random, options: list[str], scores: dict[str, dict], explore: bool) -> str:
    if explore:
        untried = [o for o in options if o not in scores]
        if untried:
            return rng.choice(untried)
    weights = [2.0 ** (4 * scores.get(o, {"score": 0.0})["score"]) for o in options]
    return rng.choices(options, weights=weights)[0]


def prompt_text(style: str, subject: str, inks: list[str], catalog: dict) -> str:
    names = []
    for i in inks:
        if i.endswith("_black"):  # a photographed bottle reads dark grey; the prompt asks for black
            names.append("#000000 black")
            continue
        lk = look(catalog, i)
        names.append("#%02X%02X%02X %s" % (*(round(c * 255) for c in lk.srgb), lk.name.lower()))
    return TEMPLATES[style].format(subject=subject, palette=", ".join(names))


def random_palette(rng: random.Random, max_inks: int = 5) -> list[str]:
    """One to four colour inks from everything owned (fewer under `max_inks`), black last: a palette nobody chose."""
    return sorted(rng.sample(COLOUR_INKS, rng.randint(1, min(4, max_inks - 1)))) + ["true_black"]


def _gene(rng, exploring, options, scores):
    return _pick(rng, options, scores, exploring)


def parents_for(style: str, subject: str, palette: str, designs: dict[str, dict], totals: dict[str, int],
                k: int = 3) -> list[str]:
    """The kept designs a prompt builds on: those sharing at least two of its style, subject and ink load,
    the closest and best kept first."""
    found = []
    for did, total in totals.items():
        d = designs.get(did)
        if total <= 0 or not d or "genes" not in d or d.get("study_of"):
            continue
        g = d["genes"]
        shared = (g["style"] == style) + (g["subject"] == subject) + (g["palette"] == palette)
        if shared >= 2:
            found.append((-shared, -total, did))
    return [did for *_, did in sorted(found)[:k]]


ASK_KINDS = ("near", "branch", "blend", "cluster")
STEER_CHARS = 200


def open_asks(asks: dict | None) -> list[tuple[str, dict]]:
    """The owner's open asks, oldest first."""
    return sorted(((aid, a) for aid, a in (asks or {}).items()
                   if a.get("status", "open") == "open" and a.get("kind") in ASK_KINDS and a.get("anchors")),
                  key=lambda x: (x[1].get("at", ""), x[0]))


def _inks_of(doc: dict) -> list[str]:
    return [i["id"] if isinstance(i, dict) else i for i in doc.get("inks", [])]


def _ask_genes(kind: str, anchors: list[dict], k: int, rng: random.Random, scores: dict, totals: dict[str, int]):
    """One try at an ask's prompt: (style, subject, inks, fill, the anchor whose prompt it may reuse)."""
    a = anchors[k % len(anchors)]
    ga = a["genes"]
    subjects = scores.get("subject", {})
    if kind == "near" or (kind == "blend" and len(anchors) < 2):
        return ga["style"], _pick(rng, [x for x in SUBJECTS if x != ga["subject"]], subjects, k % 2 == 0), \
            _inks_of(a), ga.get("fill", "packed"), a
    if kind == "branch":
        styles = [x for x in TEMPLATES if x != ga["style"]]
        return _pick(rng, styles, scores.get("style", {}), k % 2 == 0), ga["subject"], _inks_of(a), \
            ga.get("fill", "packed"), None
    if kind == "blend":
        x, y = (anchors[0], anchors[1]) if k % 2 == 0 else (anchors[1], anchors[0])
        subject = y["genes"]["subject"] if k < 2 else _pick(rng, SUBJECTS, subjects, True)
        return x["genes"]["style"], subject, _inks_of(y), x["genes"].get("fill", "packed"), None
    weights = [1 + max(0, totals.get(m["id"], 0)) for m in anchors]  # cluster: sample the neighbourhood
    m = rng.choices(anchors, weights=weights)[0]
    other = rng.choices(anchors, weights=weights)[0]
    subject = other["genes"]["subject"] if rng.random() < 0.5 else _pick(rng, SUBJECTS, subjects, True)
    return m["genes"]["style"], subject, _inks_of(m), m["genes"].get("fill", "packed"), m


def _ask_text(style, subject, inks, catalog, like):
    if style in TEMPLATES:
        return prompt_text(style, subject, inks, catalog)
    # A style the templates do not have yet (a round's own overlay): the anchor's prompt, new subject.
    if like and like.get("prompt") and like["genes"]["subject"] in like["prompt"] and _inks_of(like) == inks:
        return like["prompt"].replace(like["genes"]["subject"], subject)
    return None


def ask_prompts(asks: dict | None, designs: dict[str, dict], votes: dict, catalog: dict, room: int,
                seen: set, rng: random.Random, tile: dict | None = None) -> list[dict]:
    """Prompts for the open asks, oldest first, while they fit in `room`; each names its ask and anchors. An ask the
    round's tile cannot draw (`fits_tile`) waits for a round that can."""
    scores, totals = gene_scores(designs, votes), votes_by_design(votes)
    out = []
    for aid, ask in open_asks(asks):
        anchors = [{**designs[a], "id": a} for a in ask["anchors"] if a in designs and "genes" in designs[a]]
        want = max(1, min(5, int(ask.get("n", 1))))
        if not anchors or len(out) + want > room:
            continue
        steer = " ".join(str(ask.get("steer") or "").split())[:STEER_CHARS]
        kept = sorted(anchors, key=lambda m: -totals.get(m["id"], 0))
        parents = [m["id"] for m in kept[:3]]
        made = 0
        for k in range(40 * want):
            if made == want:
                break
            style, subject, inks, fill, like = _ask_genes(ask["kind"], anchors, k, rng, scores, totals)
            key = (style, subject, palette_key(inks))
            text = _ask_text(style, subject, inks, catalog, like)
            if key in seen or text is None or not fits_tile(tile, style, inks):
                continue
            seen.add(key)
            made += 1
            out.append({"style": style, "subject": subject, "palette": palette_key(inks), "inks": inks, "fill": fill,
                        "separation": SEPARATION.get(style) or (like or {}).get("separation", "flash"),
                        "variant": f"ask-{ask['kind']}", "ask": aid, "parents": parents,
                        "text": f"{text} {steer}" if steer else text})
    return out


def propose(designs: dict[str, dict], votes: dict, catalog: dict, round_no: int, n: int = 9, explore: float = 0.34,
            seed: int = 0, asks: dict | None = None, tile: dict | None = None) -> list[dict]:
    """The next round's prompts. The owner's open asks come first; of the rest, `explore` are on genes not
    tried yet (random palettes from every owned ink) and the others are weighted toward the best-scoring
    genes. No style takes more than its share of a round. Each prompt names the kept designs it builds on.
    A `tile_round` composes every design for its tile: loads within its ink count, its size in every prompt."""
    rng = random.Random(seed * 1000 + round_no)
    scores, totals = gene_scores(designs, votes), votes_by_design(votes)
    max_inks = tile["max_inks"] if tile else 5
    palettes = [k for k, inks in PALETTES.items() if len(inks) <= max_inks]
    styles = [k for k in TEMPLATES if k not in (tile or {}).get("avoid_styles", {})]
    seen = {(d["genes"]["style"], d["genes"]["subject"], d["genes"]["palette"]) for d in designs.values() if "genes" in d}
    out = ask_prompts(asks, designs, votes, catalog, n, seen, rng, tile)
    cap = max(2, math.ceil(n / len(styles)))
    per_style, attempts = {}, 0
    while len(out) < n and attempts < 100 * n:
        attempts += 1
        exploring = rng.random() < explore
        style = _gene(rng, exploring, styles, scores.get("style", {}))
        if per_style.get(style, 0) >= cap:
            continue
        subject = _gene(rng, exploring, SUBJECTS, scores.get("subject", {}))
        inks = random_palette(rng, max_inks) if exploring else PALETTES[_gene(rng, False, palettes, scores.get("palette", {}))]
        palette = palette_key(inks)
        fill = _gene(rng, exploring, FILLS, scores.get("fill", {}))
        if (style, subject, palette) in seen:
            continue
        seen.add((style, subject, palette))
        per_style[style] = per_style.get(style, 0) + 1
        out.append({"style": style, "subject": subject, "palette": palette, "inks": inks, "fill": fill,
                    "separation": SEPARATION[style], "variant": "explore" if exploring else "exploit",
                    "parents": parents_for(style, subject, palette, designs, totals),
                    "text": prompt_text(style, subject, inks, catalog)})
    for k, p in enumerate(out, 1):
        p["id"] = f"r{round_no}-{k:02d}-{slug(p['style'])[:10]}-{slug(p['subject'])[:16]}"
        if tile:
            p["text"] = f"{p['text']} {tile['sentence']}"
    return [{"id": p.pop("id"), **p} for p in out]


def round_images(plan: dict, corpus: Path) -> list[tuple[Path, dict, dict]]:
    """A round's generated images in the corpus, in file order: (image, its tatbot.art-generation/1 sidecar, the
    plan's prompt it answers)."""
    by_id = {p["id"]: p for p in plan["prompts"]}
    found = []
    for side_path in sorted(corpus.glob("flow/*/*.json")):
        side = json.loads(side_path.read_text())
        prompt = by_id.get(side.get("prompt_id") or "")
        if side.get("schema") == "tatbot.art-generation/1" and prompt is not None:
            found.append((side_path.with_suffix(".jpg"), side, prompt))
    return found


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
    return path
