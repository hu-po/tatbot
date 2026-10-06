"""One portable design as the factory's artwork: ``--design FILE``.

``tatbot sim generate <distribution> -- --design design.json`` draws the
material strokes of a ``tatbot.inkmap-design/1`` — a plane or cylinder chart
design, as Inkmap's Paper and Cylinder workspaces and ``tatbot design place``
export it — in place of the shared collection, for the artwork and erase
tasks. Everything else is the factory's own: the trajectory builder, the
charge model and its dips, the reach mask, the cameras, the dataset writer.
Every episode records the design's digest beside its strokes.

The design's chart is mapped onto the simulator's canvas, and the two have to
be the same kind of surface:

* a plane design goes on a flat substrate (the paper pad, the silicone skin)
  as it is — Inkmap's u is canvas x and v is canvas y, both centred;
* a cylinder design goes on the paper cylinder, whose radius must match the
  design's, by a quarter turn. Inkmap runs u along the axis and v as arc from
  the crest; the simulator wraps canvas x and runs y along the axis
  (config.SurfaceDR.cylinder_axis). Both charts are right-handed about the
  outward normal, so the map is the proper rotation (x, y) = (-v, u) and the
  artwork keeps its handedness — a bare swap of the axes would mirror it.

Placement is the design's own by default (``authored``): its anchor is about
the chart centre, which is the canvas centre, so the strokes land where Inkmap
put them and the reach mask has the last word. ``sampled`` recentres the
artwork and draws an offset inside the flat reach envelope and the reach mask,
the way the shared collection is placed, for a run that wants the design in
many places. Rotation, mirroring and scale are the design's in both.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from tatbot_sim.strokes import Stroke

PLACEMENTS = ("authored", "sampled")
IDENTITY = ((1.0, 0.0), (0.0, 1.0))
# Inkmap chart (u along the axis, v arc from the crest) -> sim canvas (x arc,
# y along the axis): u -> +y, and a right-handed map then puts v on -x.
CYLINDER_QUARTER_TURN = ((0.0, -1.0), (1.0, 0.0))
RADIUS_TOLERANCE_M = 0.001
SAMPLED_TRIES = 200


class DesignSceneError(ValueError):
    """The design cannot be drawn on this substrate as asked."""


@dataclasses.dataclass(frozen=True)
class DesignScene:
    path: Path
    name: str
    artwork_names: tuple[str, ...]
    design_sha256: str
    target: dict
    substrate: str
    placement: str
    chart_to_canvas: tuple[tuple[float, float], tuple[float, float]]
    strokes_m: tuple[np.ndarray, ...]
    """Canvas metres, at the authored placement about the canvas centre."""
    width_mm: float
    dropped_short_strokes: int

    @property
    def length_m(self) -> float:
        return sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in self.strokes_m)

    @property
    def size_mm(self) -> list[float]:
        everything = np.concatenate(self.strokes_m)
        return ((everything.max(axis=0) - everything.min(axis=0)) * 1000.0).tolist()

    def est_cost_s(self, config=None) -> float:
        """The planner's own estimate of the episode: the language module's
        pacing, so the number agrees with the budget plan_batch will apply."""
        from tatbot_sim.language import _cost_s, pacing

        return pacing(config)[2] + _cost_s(list(self.strokes_m), config)

    def summary(self) -> dict:
        return {
            "path": str(self.path), "name": self.name, "artworks": list(self.artwork_names),
            "design_sha256": self.design_sha256, "target": self.target, "substrate": self.substrate,
            "placement": self.placement, "chart_to_canvas": [list(row) for row in self.chart_to_canvas],
            "stroke_count": len(self.strokes_m), "path_length_m": self.length_m,
            "stroke_width_mm": self.width_mm, "size_mm": self.size_mm,
            "dropped_short_strokes": self.dropped_short_strokes, "est_cost_s": self.est_cost_s(),
        }

    def sample(self, rng, sheet, budget_s: float, *, verb: str = "draw", reachable=None,
               split: str = "train", width_mm: float = .3, calibration: bool = False, config=None):
        """artwork_scene.sample_artwork_scene's contract, for this one design.

        Refuses, rather than shrinking, what does not fit: a design is not a
        motif the planner may resample smaller.
        """
        from tatbot_sim.language import REACH

        cost = self.est_cost_s(config)
        if cost > budget_s:
            raise SystemExit(f"design {self.name!r} needs ~{cost:.0f} s and this batch's scene budget "
                             f"is {budget_s:.0f} s; raise --horizon")
        offset = np.zeros(2)
        candidates = [np.zeros(2)]
        strokes = list(self.strokes_m)
        if self.placement == "sampled":
            everything = np.concatenate(strokes)
            centre = (everything.min(axis=0) + everything.max(axis=0)) / 2
            strokes = [p - centre for p in strokes]
            bound = REACH - float(np.abs(np.concatenate(strokes)).max()) - self.width_mm / 2000
            if bound <= 0:
                raise SystemExit(f"design {self.name!r} is wider than the flat reach envelope ({2 * REACH * 1000:.0f} mm)")
            candidates += [rng.uniform(-bound, bound, 2) for _ in range(SAMPLED_TRIES)]
        for offset in candidates:
            placed = [p + offset for p in strokes]
            if reachable is None or reachable.ok(np.concatenate(placed)):
                break
        else:
            where = ("at its authored placement" if self.placement == "authored"
                     else f"anywhere in {SAMPLED_TRIES} draws")
            raise SystemExit(f"design {self.name!r} has strokes where the tool cannot be held normal "
                             f"to the {self.substrate} {where}; move it in Inkmap or pass "
                             "--design-placement sampled")
        return [Stroke(p) for p in placed], {
            "schema": "tatbot.artwork-scene/1",
            "design_id": self.path.stem, "source_sha256": self.design_sha256,
            "family": "portable-design", "split": "external", "name": self.name,
            "prompt": f"{verb} the {' and '.join(n.lower() for n in self.artwork_names)} tattoo design",
            "size_mm": self.size_mm, "planning_width_mm": self.width_mm,
            "rotation_rad": 0.0, "mirrored": False, "offset_m": [float(v) for v in offset],
            "stroke_count": len(placed), "path_length_m": self.length_m,
            "est_cost_s": cost, "rejected_candidates": [],
            "portable_design": self.summary(),
        }


def _chart_to_canvas(target: dict, substrate) -> tuple[tuple[float, float], tuple[float, float]]:
    """The map from the design's chart to this substrate's canvas, or a refusal."""
    shape = getattr(substrate, "shape", "pad")
    kind = target["kind"]
    if shape == "cylinder":
        if kind != "cylinder":
            raise DesignSceneError(f"a {kind} design cannot be drawn on {substrate.name}, a rigid cylinder; "
                                   "place it on Inkmap's Cylinder workspace or pick a flat substrate")
        if abs(float(target["radius_m"]) - float(substrate.radius_m)) > RADIUS_TOLERANCE_M:
            raise DesignSceneError(f"the design's cylinder radius is {target['radius_m'] * 1000:.1f} mm and "
                                   f"{substrate.name} is {substrate.radius_m * 1000:.1f} mm")
        return CYLINDER_QUARTER_TURN
    if kind != "plane":
        raise DesignSceneError(f"a {kind} design cannot be drawn on {substrate.name}, a flat sheet; "
                               "place it on Inkmap's Paper workspace or select the paper cylinder")
    return IDENTITY


def load(path: Path, substrate, placement: str = "authored") -> DesignScene:
    """Read, validate and map one design onto ``substrate``'s canvas."""
    from tatbot_sim.inkmap.design_strokes import design_strokes

    if placement not in PLACEMENTS:
        raise DesignSceneError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get("schema") != "tatbot.inkmap-design/1":
        raise DesignSceneError(f"{path} is not a tatbot.inkmap-design/1 file")
    try:
        exported = design_strokes(value)
    except ValueError as exc:
        raise DesignSceneError(str(exc)) from exc
    rot = np.asarray(_chart_to_canvas(exported["target"], substrate), dtype=np.float64)
    strokes = tuple(np.asarray(s, dtype=np.float64) / 1000.0 @ rot.T for s in exported["strokes_mm"])
    everything = np.concatenate(strokes)
    half = np.asarray([substrate.width_m, substrate.height_m]) / 2
    if placement == "authored" and np.any(np.abs(everything) > half + 1e-9):
        over = (np.abs(everything).max(axis=0) - half) * 1000
        raise DesignSceneError(f"the authored placement leaves {substrate.name}'s canvas by "
                               f"{max(over):.1f} mm; move it in Inkmap or pass --design-placement sampled")
    names = tuple(value["artworks"][item["artwork_id"]]["name"] for item in value["placements"])
    return DesignScene(
        path=path, name=exported["name"], artwork_names=names, design_sha256=exported["design_sha256"],
        target=exported["target"], substrate=substrate.name, placement=placement,
        chart_to_canvas=tuple(tuple(float(v) for v in row) for row in rot.tolist()),
        strokes_m=strokes, width_mm=float(exported["stroke_width_mm"]),
        dropped_short_strokes=int(exported["dropped_short_strokes"]),
    )


def _field_default(args, name: str, fallback):
    for f in dataclasses.fields(type(args)):
        if f.name == name:
            return f.default
    return fallback


def from_args(args, substrate, *, config=None) -> DesignScene | None:
    """The design a generate/preview Args asks for, checked before anything
    builds — or None when it asks for none. Refusals are SystemExits that
    name the flag to change."""
    path = getattr(args, "design", None)
    if not path:
        return None
    task = getattr(args, "task", "artwork")
    if task not in ("artwork", "erase"):
        raise SystemExit(f"--design draws the artwork task (or erases it); --task {task} has its own scenes")
    if getattr(args, "scenario", None):
        raise SystemExit("--design and --scenario are two different sources of strokes; pass one")
    try:
        scene = load(Path(path), substrate, getattr(args, "design_placement", "authored"))
    except (DesignSceneError, OSError, ValueError) as exc:
        raise SystemExit(f"--design {path}: {exc}") from exc
    width = getattr(args, "artwork_width_mm", None)
    if width is not None and width != _field_default(args, "artwork_width_mm", width) \
            and abs(width - scene.width_mm) > 0.05:
        raise SystemExit(f"the design carries its own stroke width ({scene.width_mm:.2f} mm); "
                         f"--artwork-width-mm {width} applies to the shared collection only")
    n_app = int(args.dr.approach.duration_s[1] * 30)
    budget = ((args.horizon - n_app) / 30.0 - 0.5) * 0.96
    cost = scene.est_cost_s(config)
    if cost > budget:
        needed = math.ceil((cost / 0.96 + 0.5) * 30 + n_app)
        raise SystemExit(f"design {scene.name!r} needs ~{cost:.0f} s of episode ({len(scene.strokes_m)} strokes, "
                         f"{scene.length_m * 1000:.0f} mm) and --horizon {args.horizon} allows {budget:.0f} s; "
                         f"pass --horizon {needed} or more")
    print(f"[design] {scene.name!r}: {len(scene.strokes_m)} strokes, {scene.length_m * 1000:.0f} mm, "
          f"{scene.size_mm[0]:.1f} x {scene.size_mm[1]:.1f} mm at {scene.width_mm:.2f} mm on {substrate.name}, "
          f"{scene.placement} placement, ~{cost:.0f} s", flush=True)
    return scene


def width_mm_for(design: DesignScene | None, args) -> float:
    """The ink footprint: the design's own width, else the flag's."""
    return args.artwork_width_mm if design is None else design.width_mm


def sampler(design: DesignScene | None):
    """plan_batch's ``artwork_sampler`` for this design, or None for the collection."""
    return None if design is None else design.sample


def summary(design: DesignScene | None) -> dict | None:
    return None if design is None else design.summary()
