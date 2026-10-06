"""Score a drawn sheet against the design that was asked for.

Drawing is the rare robot task that ships with its own answer key: the planner
knows the polylines it asked for (``planning.BatchPlan.paths``) and the
environment knows the pigment that landed (``InkField.field``). Scoring is
therefore a raster comparison, and this module is the only place that comparison
is defined.

**One rasterizer.** The intended design is stamped through
:meth:`InkField.rasterize` — the same splat, the same kernel, the same texel
grid the pen deposits with. A second rasterizer written "just for scoring" would
measure its own disagreement with the first one, so there is not one.

**The headline number is an F-score at a stated tolerance**, not an IoU. Strokes
are a few texels wide, so IoU on a thin line collapses under a sub-millimetre
offset that a human would call a good drawing; precision/recall at a tolerance
band degrade smoothly and stay interpretable — precision is "ink that belongs",
recall is "design that got drawn". IoU is still reported, as a diagnostic.

The tolerance is one pen radius by default: ink within half a line width of the
intended path is on the line. It is an argument because it is a judgement call,
and it is recorded in every score.

This module measures the DRAWING. Mechanical episode facts (contact fraction,
floor clamping, chunk rejections, tip height) come from the rollout loop and are
reported alongside, not folded into this number.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch

from tatbot_sim.inkfield import InkField
from tatbot_sim.strokes import Stroke
from tatbot_sim.surface import Surface

# Any pigment at all counts as marked. The field saturates at 1.0 and a pen
# pass lands well above this, so the threshold separates "ink" from "the faint
# tail of a splat kernel" rather than trading off line weight.
INK_THRESHOLD = 0.05


@dataclass(frozen=True)
class DrawingScore:
    """How well one env's sheet matches the design it was asked for.

    ``f1`` is the headline. Everything else explains it: a low ``precision``
    with high ``recall`` is a policy that drew the design and kept going; the
    reverse is a policy that drew part of it and stopped.
    """

    f1: float
    precision: float                       # drawn ink within tolerance of the design
    recall: float                          # design covered by ink within tolerance
    iou: float                             # diagnostic; harsh on thin strokes
    chamfer_drawn_to_intended_mm: float    # mean stray distance of the ink laid down
    chamfer_intended_to_drawn_mm: float    # mean distance from the design to the nearest ink
    coverage_ratio: float                  # drawn texels / intended texels; >1 = over-inking
    tolerance_mm: float
    intended_texels: int
    drawn_texels: int
    blank: bool                            # nothing was drawn at all
    degenerate: bool                       # the design itself is empty; f1 is nan

    def as_dict(self) -> dict:
        # JSON has no NaN or infinity. Blank drawings legitimately have no
        # drawn-to-design distance, so carry those undefined diagnostics as
        # null rather than emitting Python's non-standard JSON constants.
        return {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in asdict(self).items()
        }


def strokes_from_plan_paths(path) -> list[Stroke]:
    """One env's ``planning.BatchPlan.paths`` entry -> strokes the judge can use.

    ``paths`` is plain JSON-able geometry and it is NOT one shape. The planner
    writes three, by task:

    * ``[[[x, y], ...], ...]`` — a list of strokes (shape, language, erase,
      body-tattoo scenes);
    * ``[[x, y], ...]`` — a single flat polyline (maze/squiggle, which has one
      stroke and stores its points directly);
    * ``[]`` — a dip episode, which draws nothing.

    The flat form is the trap: read as a list of strokes it becomes a pile of
    two-element "strokes" and the reference raster comes out as noise near the
    origin, which scores plausibly and is wrong. Normalizing in one place is
    cheaper than every caller remembering.
    """
    if path is None or len(path) == 0:
        return []
    first = np.asarray(path[0], dtype=np.float32)
    if first.ndim == 1:                       # flat polyline: one stroke
        return [Stroke(np.asarray(path, dtype=np.float32))]
    return [Stroke(np.asarray(stroke, dtype=np.float32))
            for stroke in path if len(stroke) >= 2]


def intended_field_like(
    ink_field: InkField,
    surface: Surface,
    strokes_per_env: list[list[Stroke]],
    opacity,
) -> torch.Tensor:
    """Rasterize the intended design with the field's own kernel.

    Takes the live :class:`InkField` rather than its parameters so the reference
    cannot drift from the field it will be compared against: same pen radii,
    same texel grid, same device.
    """
    reference = InkField(
        ink_field.num_envs,
        surface,
        pen_radius_m=ink_field.pen_radius_m,
        laser_radius_m=ink_field.laser_radius_m,
        ink_rgb=ink_field.ink_rgb,
        device=ink_field.device,
        max_stretch=ink_field.max_stretch,
    )
    reference.rasterize(surface, strokes_per_env, opacity)
    return reference.field


def _distance_to_mask_texels(mask: np.ndarray) -> np.ndarray:
    """Per texel, the Euclidean distance to the nearest True texel.

    ``cv2.distanceTransform`` measures distance to the nearest ZERO, so the
    complement goes in. ``DIST_MASK_PRECISE`` because a 3x3 chamfer mask carries
    up to ~5% error, which at 0.42 mm/texel is a tenth of a millimetre of
    invented accuracy in a number reported in millimetres.
    """
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    complement = (~mask).astype(np.uint8)
    return cv2.distanceTransform(complement, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def score_fields(
    drawn: torch.Tensor,
    intended: torch.Tensor,
    *,
    texel_per_m: float,
    tolerance_m: float,
    threshold: float = INK_THRESHOLD,
) -> list[DrawingScore]:
    """Score (B, rows, cols) pigment fields against (B, rows, cols) references.

    A blank sheet scores 0 rather than raising: "the policy never touched the
    paper" is a result, and the most common failure worth ranking.
    """
    if drawn.shape != intended.shape:
        raise ValueError(f"field shapes differ: drawn {tuple(drawn.shape)} vs "
                         f"intended {tuple(intended.shape)}")
    mm_per_texel = 1000.0 / texel_per_m
    tolerance_texels = tolerance_m * texel_per_m
    tolerance_mm = tolerance_m * 1000.0

    drawn_np = (drawn.detach().cpu().numpy() > threshold)
    intended_np = (intended.detach().cpu().numpy() > threshold)

    scores = []
    for d_mask, i_mask in zip(drawn_np, intended_np, strict=True):
        n_drawn, n_intended = int(d_mask.sum()), int(i_mask.sum())
        if n_intended == 0:
            scores.append(DrawingScore(
                f1=float("nan"), precision=float("nan"), recall=float("nan"),
                iou=float("nan"), chamfer_drawn_to_intended_mm=float("nan"),
                chamfer_intended_to_drawn_mm=float("nan"), coverage_ratio=float("nan"),
                tolerance_mm=tolerance_mm, intended_texels=0, drawn_texels=n_drawn,
                blank=n_drawn == 0, degenerate=True))
            continue
        if n_drawn == 0:
            scores.append(DrawingScore(
                f1=0.0, precision=0.0, recall=0.0, iou=0.0,
                chamfer_drawn_to_intended_mm=float("nan"),
                chamfer_intended_to_drawn_mm=float("inf"), coverage_ratio=0.0,
                tolerance_mm=tolerance_mm, intended_texels=n_intended, drawn_texels=0,
                blank=True, degenerate=False))
            continue

        dist_to_intended = _distance_to_mask_texels(i_mask)
        dist_to_drawn = _distance_to_mask_texels(d_mask)
        stray = dist_to_intended[d_mask]
        gap = dist_to_drawn[i_mask]

        precision = float((stray <= tolerance_texels).mean())
        recall = float((gap <= tolerance_texels).mean())
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        union = int((d_mask | i_mask).sum())

        scores.append(DrawingScore(
            f1=f1,
            precision=precision,
            recall=recall,
            iou=float((d_mask & i_mask).sum()) / union,
            chamfer_drawn_to_intended_mm=float(stray.mean()) * mm_per_texel,
            chamfer_intended_to_drawn_mm=float(gap.mean()) * mm_per_texel,
            coverage_ratio=n_drawn / n_intended,
            tolerance_mm=tolerance_mm,
            intended_texels=n_intended,
            drawn_texels=n_drawn,
            blank=False,
            degenerate=False,
        ))
    return scores


# --- deliberate degradation, for validating the judge itself -----------------
#
# A judge is only trustworthy if its score moves the right way when the drawing
# gets worse in a way whose sign we already know. These produce those drawings
# from a design, and the slice-A test asserts monotonicity over each one. They
# live here, not in the test, so a battery can re-run the same self-check
# against whatever tool and sheet it is about to score with.


def offset_strokes(strokes: list[Stroke], dx_m: float, dy_m: float) -> list[Stroke]:
    """Draw the right shape in the wrong place."""
    return [Stroke(s.points + np.array([dx_m, dy_m], dtype=np.float32)) for s in strokes]


def scale_strokes(strokes: list[Stroke], factor: float) -> list[Stroke]:
    """Right shape, wrong size, about the design's own centroid."""
    if not strokes:
        return []
    centroid = np.concatenate([s.points for s in strokes], axis=0).mean(axis=0)
    return [Stroke((s.points - centroid) * factor + centroid) for s in strokes]


def truncate_strokes(strokes: list[Stroke], keep: float) -> list[Stroke]:
    """Stop drawing partway — what a policy that loses the plot produces."""
    kept = []
    for stroke in strokes:
        n = max(2, int(round(len(stroke.points) * keep)))
        if n < len(stroke.points):
            kept.append(Stroke(stroke.points[:n]))
        elif keep >= 1.0:
            kept.append(stroke)
    return [s for s in kept if len(s.points) >= 2]


def jitter_strokes(strokes: list[Stroke], sigma_m: float, seed: int = 0) -> list[Stroke]:
    """A shaky hand: unbiased noise on every sample."""
    rng = np.random.default_rng(seed)
    return [Stroke(s.points + rng.normal(0.0, sigma_m, s.points.shape).astype(np.float32))
            for s in strokes]


def drop_strokes(strokes: list[Stroke], keep_fraction: float, seed: int = 0) -> list[Stroke]:
    """Miss whole strokes — the raster shadow of a pen held off the paper.

    A lift is a trajectory fact, so this is its consequence rather than the
    thing itself; the trajectory-level lift test belongs with the rollout loop.
    """
    if not strokes:
        return []
    rng = np.random.default_rng(seed)
    n_keep = max(0, int(round(len(strokes) * keep_fraction)))
    keep = sorted(rng.permutation(len(strokes))[:n_keep])
    return [strokes[i] for i in keep]


ENGAGED_MIN_COVERAGE_DELTA = 1e-5


def engaged(kind: str, start: float, end: float, dips: int = 0) -> bool:
    """Did the tool actually do to the sheet what the prompt says it did?

    An episode whose ink never moves is not a weak demonstration, it is a
    mislabelled one: it ships the same "remove the ink" sentence as the rest
    while showing the arm never touching the skin. A dip episode's claim is
    the dip, not a mark: it is engaged when its charge landed.
    """
    if kind == "dip":
        return dips > 0
    delta = float(end) - float(start)
    want = -1.0 if kind == "erase" else 1.0
    return want * delta > ENGAGED_MIN_COVERAGE_DELTA
