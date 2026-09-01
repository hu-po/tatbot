"""The drawing language: scene programs -> strokes + truthful prompts.

Design (see the brainstorm artifact): every episode samples a SCENE PROGRAM
that compiles two ways — down into strokes for the expert to trace (and for
the judge to score against), and up into a natural-language prompt through a
template realizer. The prompt is derived data and can never lie about the
drawing; the program is what run_meta stores, so prompts can be re-realized
later (paraphrase, tool/surface swap, normalization) without touching videos.

Three registries grow the language over time:

- MOTIFS   nouns: parameterized primitives compiling to polylines. Two
           families coexist (operator call, 2026-08-23): free-form motifs
           (circle, star, heart...) judged against their intended paths, and
           grid-locked motifs (squiggle, staircase, grid box) whose vertices
           snap to the sheet's printed 6 mm ruling — the stencil family.
- MODIFIERS adjectives that change geometry AND wording together: quantized
           size bins keep "small" truthful; "dashed" splits strokes (pen
           lifts); "wavy" perturbs; "nested" draws concentric copies.
- (composition lives in sample_scene: 1-3 motifs, collision-aware layout in
           the reach envelope, truthful joining words.)

The prompt frame is slotted: ``{verb} {scene} {tool} {surface}``, where the
tool slot is the fitted tool's own ``prompt_phrase`` from ``config/tools/``
and the surface is "on the paper pad" — so the same programs re-realize as
"using 3RL" / "on the 3mm silicon skin" when the hardware changes.

The verb slot exists because the removal laser runs the SAME scene programs
backwards: a program that compiles to strokes to draw also compiles to
strokes to erase, and only the wording (and what the sheet starts with)
differs. An erase scene is definite — you remove *the* star that is already
there, not *a* star — so the realizer swaps the determiner with the verb
rather than leaving the prompt subtly untruthful.

Episode time is a budget, not a constant: scenes are sampled to fit a step
budget (travel/settle/descend overhead per stroke plus ink time), and the
generator sizes each batch's horizon to its scenes (cap 900 steps = 30 s,
shorter preferred — operator call).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from tatbot_sim.strokes import MazeConfig, ShapeConfig, Stroke, pacing_estimate
from tatbot_sim.tools import active_substrate, active_tool

LEXICON_VERSION = "p0.8"  # p0.8: draw sentences drop the trailing period (fm2 parity)
# p0.6: removal on a blank substrate says "the ink"
# The tool slot comes from the fitted tool's datasheet, so the prompt cannot
# describe a tool other than the one the URDF was built from. Swapping pens
# re-realizes the same scene programs as "using 3RL" without touching video.
TOOL_PHRASE = active_tool().prompt_phrase
# The surface slot comes from the SUBSTRATE the fitted tool works on, for the
# same reason the tool slot comes from its datasheet: a prompt that named the
# paper pad while the laser worked a silicone skin would be the one part of an
# episode that could tell a policy which domain it was in.
_SUBSTRATE = active_substrate()
SURFACE_PHRASE = _SUBSTRATE.surface_phrase
# Nothing is printed on a blank substrate, so the motifs that put their
# vertices ON the ruling — and say so in the prompt — have no ruling to sit on.
RULED_SURFACE = _SUBSTRATE.ruled
# The verb slot. Adding one means adding the wording here and a branch in
# planning that knows what the sheet starts with — nothing else.
VERBS = {"draw": "draw", "erase": "remove"}

# metres; centred canvas frame. Flat-reach envelope is ±0.06 m.
REACH = 0.06
SIZES = {"tiny": (0.005, 0.008), "small": (0.009, 0.014), "": (0.015, 0.022), "big": (0.023, 0.030)}
NUM_WORDS = {2: "two", 3: "three"}

# pacing for time budgets, derived from the trajectory builder's own
# constants (strokes.pacing_estimate) so the estimate cannot drift from the
# pacing it estimates — this used to be a hand-mirrored copy
DRAW_SPEED, STROKE_OVERHEAD_S, EPISODE_OVERHEAD_S = pacing_estimate(
    ShapeConfig(), MazeConfig().draw_speed_range
)


# ---------------------------------------------------------------------------
# motif registry
# ---------------------------------------------------------------------------

@dataclass
class Motif:
    key: str
    names: tuple[str, ...]
    compile: Callable          # (params, grid) -> list[np.ndarray (N,2)]
    grid_locked: bool = False
    stylable: bool = False     # accepts wavy/dashed
    nestable: bool = False
    countable: bool = True     # can appear as "two ...s" / "three ...s"
    # Legibility floor: the ~3 mm ink stroke swallows features smaller than
    # itself (a "tiny zigzag" rendered as a blob and its prompt became a
    # lie). Detailed motifs exclude the smallest size bins.
    min_size: str = "tiny"
    det: str | None = None     # overrides the article ("the letter R")
    noun_fn: Callable | None = None   # params -> noun, for parameterized motifs


MOTIFS: dict[str, Motif] = {}


def register(**kw):
    def deco(fn):
        m = Motif(compile=fn, **kw)
        MOTIFS[m.key] = m
        return fn
    return deco


def _regular(n, r, rot=0.0):
    a = np.linspace(0, 2 * np.pi, n + 1) + rot
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


@register(key="circle", names=("circle", "ring"), stylable=True, nestable=True)
def _circle(p, grid):
    return [_regular(48, p["r"])]


@register(key="square", names=("square", "box"), stylable=True, nestable=True)
def _square(p, grid):
    return [_regular(4, p["r"] * 1.414, rot=np.pi / 4 + p["rot"])]


@register(key="triangle", names=("triangle",), stylable=True, nestable=True)
def _triangle(p, grid):
    return [_regular(3, p["r"], rot=np.pi / 2 + p["rot"])]


@register(key="star", names=("star", "five pointed star"), min_size="small")
def _star(p, grid):
    a = np.linspace(0, 2 * np.pi, 11) + np.pi / 2 + p["rot"]
    rr = np.where(np.arange(11) % 2 == 0, p["r"], p["r"] * 0.42)
    return [np.stack([rr * np.cos(a), rr * np.sin(a)], 1)]


@register(key="zigzag", names=("zigzag", "zigzag line"), stylable=True, min_size="small")
def _zigzag(p, grid):
    n = int(np.clip(2 * p["r"] / 0.007, 2, 6))  # period >= ~7 mm
    x = np.linspace(-p["r"], p["r"], 2 * n + 1)
    y = np.where(np.arange(2 * n + 1) % 2 == 0, -p["r"] / 2, p["r"] / 2)
    pts = np.stack([x, y], 1)
    c, s = np.cos(p["rot"]), np.sin(p["rot"])
    return [pts @ np.array([[c, s], [-s, c]])]


@register(key="wave", names=("wave", "wavy line"), min_size="small")
def _wave(p, grid):
    n = int(np.clip(2.4 * p["r"] / 0.010, 2, 4))  # period >= ~10 mm
    x = np.linspace(-p["r"] * 1.2, p["r"] * 1.2, 60)
    y = p["r"] * 0.45 * np.sin(x / (p["r"] * 1.2) * np.pi * n)
    pts = np.stack([x, y], 1)
    c, s = np.cos(p["rot"]), np.sin(p["rot"])
    return [pts @ np.array([[c, s], [-s, c]])]


@register(key="spiral", names=("spiral", "swirl"), min_size="small")
def _spiral(p, grid):
    turns = 2.5
    a = np.linspace(0, 2 * np.pi * turns, int(40 * turns))
    rr = p["r"] * a / a[-1]
    return [np.stack([rr * np.cos(a), rr * np.sin(a)], 1)]


@register(key="cross", names=("cross", "plus sign"))
def _cross(p, grid):
    r = p["r"]
    return [np.array([[-r, 0.0], [r, 0.0]]), np.array([[0.0, -r], [0.0, r]])]


@register(key="heart", names=("heart",), min_size="small")
def _heart(p, grid):
    t = np.linspace(0, 2 * np.pi, 60)
    s = p["r"] / 17
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return [np.stack([x * s, y * s], 1)]


# --- letters: single-stroke capitals, parameterized by character ------------

def _arc(cx, cy, rx, ry, a0, a1, n=22):
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def _build_letters() -> dict[str, list[np.ndarray]]:
    """A-Z as single-line print capitals in a unit box (height 1, width ~0.7),
    the skeleton style of CNC/engraving fonts — what a pen plotter draws."""
    arc = _arc
    letters = {
        "A": [[(0, 0), (.35, 1), (.7, 0)], [(.14, .4), (.56, .4)]],
        "B": [[(0, 0), (0, 1)],
              [(0, 1), (.35, 1), *arc(.35, .75, .3, .25, 90, -90), (0, .5)],
              [(0, .5), (.38, .5), *arc(.38, .25, .32, .25, 90, -90), (0, 0)]],
        "C": [[*arc(.38, .5, .38, .5, 55, 305)]],
        "D": [[(0, 0), (0, 1)], [(0, 1), (.25, 1), *arc(.25, .5, .45, .5, 90, -90), (0, 0)]],
        "E": [[(.6, 1), (0, 1), (0, 0), (.6, 0)], [(0, .5), (.45, .5)]],
        "F": [[(.6, 1), (0, 1), (0, 0)], [(0, .5), (.45, .5)]],
        "G": [[*arc(.38, .5, .38, .5, 55, 330), (.6, .35), (.42, .35)]],
        "H": [[(0, 0), (0, 1)], [(.7, 0), (.7, 1)], [(0, .5), (.7, .5)]],
        "I": [[(.35, 0), (.35, 1)], [(.15, 1), (.55, 1)], [(.15, 0), (.55, 0)]],
        "J": [[(.6, 1), (.6, .25), *arc(.38, .25, .22, .25, 0, -160)]],
        "K": [[(0, 0), (0, 1)], [(.65, 1), (0, .45)], [(.25, .62), (.65, 0)]],
        "L": [[(0, 1), (0, 0), (.6, 0)]],
        "M": [[(0, 0), (0, 1), (.35, .4), (.7, 1), (.7, 0)]],
        "N": [[(0, 0), (0, 1), (.7, 0), (.7, 1)]],
        "O": [[*arc(.35, .5, .35, .5, 0, 360)]],
        "P": [[(0, 0), (0, 1), (.4, 1), *arc(.4, .75, .3, .25, 90, -90), (0, .5)]],
        "Q": [[*arc(.35, .5, .35, .5, 0, 360)], [(.45, .25), (.75, -.05)]],
        "R": [[(0, 0), (0, 1), (.4, 1), *arc(.4, .75, .3, .25, 90, -90), (0, .5)],
              [(.3, .5), (.68, 0)]],
        "S": [[*arc(.35, .74, .3, .24, 60, 270), *arc(.35, .26, .3, .24, 90, -160)]],
        "T": [[(0, 1), (.7, 1)], [(.35, 1), (.35, 0)]],
        "U": [[(0, 1), (0, .3), *arc(.35, .3, .35, .3, 180, 360), (.7, 1)]],
        "V": [[(0, 1), (.35, 0), (.7, 1)]],
        "W": [[(0, 1), (.18, 0), (.35, .55), (.52, 0), (.7, 1)]],
        "X": [[(0, 1), (.7, 0)], [(.7, 1), (0, 0)]],
        "Y": [[(0, 1), (.35, .5), (.7, 1)], [(.35, .5), (.35, 0)]],
        "Z": [[(0, 1), (.7, 1), (0, 0), (.7, 0)]],
    }
    return {ch: [np.asarray(s, dtype=np.float64) for s in strokes]
            for ch, strokes in letters.items()}


LETTERS = _build_letters()


def _hex_centers(r, rings):
    """Centres of a hexagonal circle-packing, ``rings`` rings out from one.

    The flower of life is not a decorative squiggle but a construction: every
    circle passes through the centres of its neighbours, which is exactly a
    triangular lattice of spacing r. Getting that spacing wrong (r*1.5, say)
    produces a daisy that a tattooer would spot instantly.
    """
    out = []
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            # axial -> cartesian on a triangular lattice of pitch r
            x = r * (i + j * 0.5)
            y = r * j * np.sqrt(3) / 2
            if np.hypot(x, y) <= r * rings + 1e-9:
                out.append((x, y))
    return out


@register(key="seed_of_life", names=("seed of life", "seven circle rosette"),
          min_size="small", det="a")
def _seed_of_life(p, grid):
    """Seven circles: one centre, six around it. The flower's inner unit, and
    the affordable one — a full flower costs a minute and a half of drawing.

    ``r`` is the figure's OVERALL radius, as it is for every other motif, so
    the circles are half that: the outermost centre sits one circle-radius out
    and the circle reaches one more. Reading ``r`` as the circle radius made
    the figure twice its billed size and it could not be placed on the canvas.
    """
    cr = p["r"] / 2.0
    return [_regular(48, cr) + np.array(c) for c in _hex_centers(cr, 1)]


@register(key="flower_of_life", names=("flower of life",),
          min_size="big", det="a")
def _flower_of_life(p, grid):
    """The classic figure: 19 circles on the triangular lattice, closed by two
    concentric rings.

    EXPENSIVE, and unavoidably so: at the full 30 mm radius it is about 1.5 m
    of stroke, and 21 separate strokes cost another 38 s of pen-up between
    them — roughly 110 s in all, against a 30 s default episode. The scene
    sampler's budget therefore refuses it at ordinary episode lengths and it
    appears only when an episode is long enough to hold it. That is the honest
    gate: a figure this size either gets its time or is not drawn."""
    cr = p["r"] / 3.24        # overall radius -> circle radius (2 rings + border)
    circles = [_regular(56, cr) + np.array(c) for c in _hex_centers(cr, 2)]
    # the two containing rings that make it a flower rather than a lattice
    circles.append(_regular(72, 3 * cr))
    circles.append(_regular(72, 3 * cr * 1.08))
    return circles


@register(key="letter", names=("letter",), stylable=True, countable=False,
          min_size="small", det="the",
          noun_fn=lambda p: ("slanted " if p["slant"] else "") + "letter " + p["char"])
def _letter(p, grid):
    h = 2 * p["r"]  # letter height spans the size bin's full extent
    out = []
    for s in LETTERS[p["char"]]:
        q = s * h
        q[:, 0] += q[:, 1] * p["slant"]  # italic shear, named in the prompt
        out.append(q)
    allpts = np.concatenate(out)
    c = (allpts.min(0) + allpts.max(0)) / 2
    # letters stay upright to stay letters: jitter, not free rotation
    rot = (p["rot"] / (2 * np.pi) - 0.5) * 0.3  # +-0.15 rad
    cs, sn = np.cos(rot), np.sin(rot)
    rot_m = np.array([[cs, sn], [-sn, cs]])
    return [(q - c) @ rot_m for q in out]


# uniform choice would give the 26-character letter family the same share as
# one shape; weight it up so letters are a real slice of the distribution
MOTIF_WEIGHTS = {"letter": 3}


# --- grid-locked family: vertices on the sheet's printed ruling -------------

@register(key="gridbox", names=("box on the grid lines", "rectangle on the grid lines"),
          grid_locked=True)
def _gridbox(p, grid):
    xs = sorted(v for v in grid["xs"] if abs(v) <= REACH)
    ys = sorted(v for v in grid["ys"] if abs(v) <= REACH)
    w = min(p["cells_w"], len(xs) - 1)
    h = min(p["cells_h"], len(ys) - 1)
    i = p["ix"] % (len(xs) - w)
    j = p["iy"] % (len(ys) - h)
    x0, x1, y0, y1 = xs[i], xs[i + w], ys[j], ys[j + h]
    return [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])]


@register(key="staircase", names=("staircase", "staircase of grid steps"), grid_locked=True)
def _staircase(p, grid):
    xs = sorted(v for v in grid["xs"] if abs(v) <= REACH)
    ys = sorted(v for v in grid["ys"] if abs(v) <= REACH)
    n = min(p["steps"], len(xs) - 1, len(ys) - 1)
    i = p["ix"] % (len(xs) - n)
    j = p["iy"] % (len(ys) - n)
    pts = [[xs[i], ys[j]]]
    for k in range(n):
        pts.append([xs[i + k + 1], ys[j + k]])
        pts.append([xs[i + k + 1], ys[j + k + 1]])
    return [np.array(pts)]


# ---------------------------------------------------------------------------
# modifiers
# ---------------------------------------------------------------------------

def _resample(s, step):
    seg = np.diff(s, axis=0)
    arc_len = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
    if arc_len[-1] < 1e-9:
        return s
    t = np.linspace(0, arc_len[-1], max(int(arc_len[-1] / step), 8))
    return np.stack([np.interp(t, arc_len, s[:, i]) for i in range(2)], 1)


def mod_wavy(strokes, rng):
    out = []
    for s in strokes:
        d = _resample(s, 0.001)
        n = np.gradient(d, axis=0)
        n = np.stack([-n[:, 1], n[:, 0]], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        ph = np.linspace(0, 2 * np.pi * rng.integers(4, 8), len(d))
        out.append(d + n * np.sin(ph)[:, None] * rng.uniform(0.0008, 0.0016))
    return out


def mod_dashed(strokes, rng):
    out = []
    for s in strokes:
        d = _resample(s, 0.0005)
        n_pts = len(d)
        dash = max(8, n_pts // int(rng.integers(6, 10)))
        k = 0
        while k * dash < n_pts:
            piece = d[k * dash : k * dash + int(dash * 0.62)]
            if len(piece) > 1:
                out.append(piece)
            k += 1
    return out


def mod_nested(strokes, k):
    out = list(strokes)
    for i in range(1, k):
        out += [s * (1 - 0.32 * i) for s in strokes]
    return out


MODS = {"wavy": mod_wavy, "dashed": mod_dashed}


@dataclass(frozen=True)
class SceneStyle:
    """How ornate a scene is allowed to be.

    The defaults ARE the training draw and must stay that way: variety is the
    point of the data, and a policy that only ever saw clean geometry would be
    the poorer for it. A render is the opposite case — wavy modifiers, dashed
    strokes and hand-drawn-looking motifs read as scribble on camera — so a
    film asks for a narrower style rather than the code changing underneath
    the dataset.
    """

    style_prob: float = 0.30
    """Chance a stylable motif is given a wavy or dashed treatment."""
    nest_prob: float = 0.18
    """Chance a nestable motif is drawn as concentric copies."""
    motifs: tuple[str, ...] | None = None
    """Restrict the pool to these keys. None = the whole registry."""
    max_motifs: int = 3
    """Ceiling on how many motifs share a scene."""
    fill_budget: bool = False
    """Spend the whole episode on the scene instead of the short-skewed random
    draw. The training budget is deliberately `min` of two uniforms ("up to
    30 s but opt for shorter", operator call), which is right for a dataset and
    fatal for a film of one deliberate figure: a flower of life costs about
    110 s, and a budget that lands at 38 s two times in three refuses it however
    long the horizon is."""


DEFAULT_STYLE = SceneStyle()

# Motifs whose whole character is a wobble. Fine in training, wrong in a
# render: at a 3 mm stroke width on a 120 mm canvas they read as scribble.
SCRIBBLY = ("zigzag", "wave", "spiral", "staircase")

CLEAN_STYLE = SceneStyle(
    style_prob=0.0, nest_prob=0.0, max_motifs=2,
    motifs=tuple(k for k in ("circle", "square", "triangle", "star", "cross", "heart",
                             "seed_of_life", "flower_of_life", "letter")),
)
"""Deliberate geometry only: no wavy or dashed treatment, no nesting, and none
of SCRIBBLY. What a render should draw."""


# ---------------------------------------------------------------------------
# composition + budget + realizer
# ---------------------------------------------------------------------------

def _stroke_len(strokes):
    return sum(float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) for s in strokes)


def _cost_s(strokes):
    return _stroke_len(strokes) / DRAW_SPEED + len(strokes) * STROKE_OVERHEAD_S


def _plural(noun):
    words = noun.split()
    last = words[-1] + ("es" if words[-1].endswith(("s", "x", "sh", "ch")) else "s")
    return " ".join(words[:-1] + [last])


def _definite(phrase: str) -> str:
    """"a small star" -> "the small star"; "two circles" -> "the two circles".

    An erase episode acts on something already on the sheet, so its prompt has
    to be definite. Motifs that already name themselves definitely ("the
    letter R") are left alone rather than given a second determiner.
    """
    head, _, rest = phrase.partition(" ")
    if head == "the":
        return phrase
    if head in ("a", "an"):
        return f"the {rest}" if rest else phrase
    return f"the {phrase}"


@dataclass
class PlacedMotif:
    key: str
    params: dict
    mods: list
    nested: int
    count: int
    strokes: list = field(default_factory=list)
    phrase: str = ""


def _bounding_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and radius of a polyline set's bounding circle (from the axis-
    aligned box centre — cheap and tight enough for layout)."""
    c = (pts.min(0) + pts.max(0)) / 2
    return c, float(np.linalg.norm(pts - c, axis=1).max())


def sample_scene(rng: np.random.Generator, grid: dict, budget_s: float, verb: str = "draw",
                 reachable=None, style: "SceneStyle | None" = None):
    """One scene program that fits the time budget. Returns (strokes, program).

    ``reachable`` (optional) says where on the canvas the fitted tool can be
    held normal to the surface. A motif is placed only where it can actually be
    worked: on a mound the flanks ask the wrist for a lean it cannot make, and
    a scene laid across them would be a scene the arm quietly misses.

    ``verb`` picks what the episode does with the scene. "draw" lays it down;
    "erase" opens with it already on the sheet and takes it off, which is the
    same geometry read the other way — so removal costs a slot in the frame,
    not a second scene sampler.

    ``style`` narrows what may be drawn (tatbot_sim.language.SceneStyle);
    None is the training draw.

    Layout uses true bounding circles for EVERY motif. Grid-locked motifs sit
    at absolute sheet coordinates, so they re-roll their lattice placement
    until clear — before this they skipped collision entirely and could land
    on top of a placed motif while the prompt claimed "next to". A motif that
    exceeds the remaining budget re-rolls the slot (up to 3 draws) instead of
    ending the scene, which was quietly over-representing the cheap motifs.
    """
    style = style or DEFAULT_STYLE
    for _ in range(40):
        k_target = min(style.max_motifs, int(rng.choice([1, 1, 1, 2, 2, 3])))
        placed, occupied, spent = [], [], EPISODE_OVERHEAD_S
        for _slot in range(k_target):
            for _attempt in range(3):
                allow = style.motifs or tuple(MOTIFS)
                pool = [k for k in MOTIFS
                        if k in allow and (RULED_SURFACE or not MOTIFS[k].grid_locked)
                        for _ in range(MOTIF_WEIGHTS.get(k, 1))]
                if not pool:
                    raise ValueError(
                        f"no motif in {sorted(allow)} can be drawn on this substrate")
                key = str(rng.choice(pool))
                m = MOTIFS[key]
                order = list(SIZES)
                allowed = order[order.index(m.min_size):] if not m.grid_locked else [""]
                size = str(rng.choice(allowed)) if not m.grid_locked else ""
                r = float(rng.uniform(*SIZES[size or ""]))
                params = {"r": r, "rot": float(rng.uniform(0, 2 * np.pi)),
                          "cells_w": int(rng.integers(2, 6)), "cells_h": int(rng.integers(2, 6)),
                          "steps": int(rng.integers(3, 7)),
                          "ix": int(rng.integers(0, 64)), "iy": int(rng.integers(0, 64)),
                          "char": chr(65 + int(rng.integers(26))),
                          "slant": float(rng.choice([0.0, 0.0, 0.0, 0.28, -0.28]))}
                mods = ([str(rng.choice(list(MODS)))]
                        if (m.stylable and rng.random() < style.style_prob) else [])
                nested = (int(rng.integers(2, 4))
                          if (m.nestable and rng.random() < style.nest_prob) else 0)
                count = (int(rng.choice([1, 1, 1, 2, 3]))
                         if (k_target == 1 and m.countable and not m.grid_locked and not nested)
                         else 1)

                trial = m.compile(params, grid)
                if nested:
                    trial = mod_nested(trial, nested)
                for name in mods:
                    trial = MODS[name](trial, rng)
                cost = _cost_s(trial) * count
                if spent + cost <= budget_s:
                    break
            else:
                break  # nothing affordable in 3 draws: the scene is full

            ext = max(np.abs(np.concatenate(trial)).max(), 1e-4)
            all_strokes, new_occ = [], []
            ok = True
            for _c in range(count):
                for _try in range(60):
                    if m.grid_locked:
                        params["ix"], params["iy"] = int(rng.integers(0, 64)), int(rng.integers(0, 64))
                        copy_strokes = m.compile(params, grid)
                    else:
                        bound = max(REACH - ext - 0.004, 0.0)
                        pos = rng.uniform(-bound, bound, 2)
                        copy_strokes = [s + pos for s in trial]
                    pts = np.concatenate(copy_strokes)
                    if reachable is not None and not reachable.ok(pts):
                        continue
                    c0, r0 = _bounding_circle(pts)
                    if all(np.linalg.norm(c0 - cq) > r0 + rq + 0.006
                           for cq, rq in occupied + new_occ):
                        new_occ.append((c0, r0))
                        all_strokes += copy_strokes
                        break
                else:
                    ok = False
                    break
            if not ok:
                continue  # placement failed: leave the slot unfilled, no phantom occupancy
            occupied += new_occ
            spent += cost
            name = m.noun_fn(params) if m.noun_fn else str(rng.choice(m.names))
            adj = " ".join(filter(None, [size] + mods))
            if nested:
                det, noun = "", "nested " + _plural(name)
            elif count > 1:
                det, noun = NUM_WORDS[count], _plural(name)
            else:
                det = m.det or ("an" if (adj or name)[0] in "aeiou" else "a")
                noun = name
            phrase = " ".join(filter(None, [det, adj, noun]))
            placed.append(PlacedMotif(key, dict(params),
                                      mods, nested, count, all_strokes, phrase))
        if placed:
            break
    else:
        raise RuntimeError("could not sample a scene within the budget")

    strokes = [Stroke(s) for p in placed for s in p.strokes]
    joiner = str(rng.choice([" next to ", " and ", " beside "])) if len(placed) > 1 else ""
    phrases = [p.phrase if verb == "draw" else _definite(p.phrase) for p in placed]
    if verb == "erase" and not RULED_SURFACE:
        # The operator cannot name what they are removing: on a blank skin the
        # target was inked before the episode and there is nothing printed to
        # read it against. Naming the motif here would be the one part of an
        # episode that told a policy which domain it came from -- so removal on
        # a blank substrate says "the ink", in the order the recordings use.
        # Both slots still come from the datasheet and the substrate registry,
        # so the sentence cannot drift from the tool or the surface it names.
        prompt = f"{VERBS[verb]} the ink {SURFACE_PHRASE} {TOOL_PHRASE}"
    else:
        # no trailing period: the real fm2 draw recording types its sentence
        # without one ("draw a 6mm square using pen tip on the grid lines of
        # the paper pad"), and punctuation is exactly the kind of freebie a
        # policy can key a domain on.
        prompt = f"{VERBS[verb]} " + joiner.join(phrases) + f" {TOOL_PHRASE} {SURFACE_PHRASE}"
    program = {
        "lexicon": LEXICON_VERSION,
        "verb": verb,
        "tool": TOOL_PHRASE,
        "surface": SURFACE_PHRASE,
        "motifs": [
            {"key": p.key, "size_r": round(p.params["r"], 4), "mods": p.mods,
             "nested": p.nested, "count": p.count,
             "grid_locked": MOTIFS[p.key].grid_locked,
             **({"char": p.params["char"], "slant": p.params["slant"]}
                if p.key == "letter" else {}),
             # how many entries of the scene's stroke list belong to this
             # motif (strokes are emitted in motif order) — lets tooling map
             # strokes back to prompt words without re-compiling the program
             "n_strokes": len(p.strokes)}
            for p in placed
        ],
        "prompt": prompt,
        "est_cost_s": round(spent, 1),
    }
    program["hash"] = hashlib.sha1(
        json.dumps(program, sort_keys=True).encode()
    ).hexdigest()[:12]
    return strokes, program
