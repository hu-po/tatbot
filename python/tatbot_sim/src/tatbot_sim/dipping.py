"""Dips: leaving the drawing to charge the tool at the palette, and coming back.

A dip is the one motion in an episode that is not about the surface. The
canvas frame (strokes.py) is a chart of the SURFACE — z is height above the
skin at that xy — and the palette is not on the skin, so a dip cannot be
expressed there and go through canvas_to_world like a stroke. It is built in
the world frame and spliced into the world-frame trajectory at a stroke
boundary, where the tool is already hovering clear of the surface.

What is spliced (all at hover or above, until the plunge):

    hover over the next stroke start
      -> rise to the transit height        (max of both hovers)
      -> travel to above the cap
      -> descend to just above the rim, settle
      -> plunge to `plunge_m` below the rim, dwell   <- the only "down" part
      -> retract above the rim
      -> travel back over the stroke start at transit height
      -> descend to the hover it left from

so the segment closes on the exact point it opened from and the stroke that
follows is unchanged. The floor handed to the expert for the plunge steps is
the cap's FLOOR with a world-up normal, so the clamp lets the tool into the
cap and no deeper.

Which dips happen, and why, is scripts/lib/ink_spec.plan_dips — the same
charge arithmetic the real robot will run. This module only turns a plan
into motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tatbot_sim.strokes import ShapeConfig, Stroke


@dataclass(frozen=True)
class DipGeometry:
    """Everything a dip segment needs that is not in the plan itself."""

    rim_world: np.ndarray       # (3,) the cap rim centre, world frame
    plunge_m: float             # below the rim
    cap_depth_m: float          # rim to cap floor
    hover_m: float              # clearance above the rim on approach
    dwell_s: float
    plunge_speed: float         # m/s, in and out
    travel_speed: float         # m/s, transit
    settle_time: float          # s, holds at the rim and back at the stroke


def stroke_needs(strokes: list[Stroke], speed: float, cfg: ShapeConfig, ink_id=None):
    """What each stroke will cost, for the planner: its length in contact
    and the time the tip spends on the surface — the draw itself plus the
    settle it holds on the sheet before lifting (strokes.build_ee_trajectory)."""
    from tatbot_sim import tools

    need = tools.ink_registry().StrokeNeed
    out = []
    for st in strokes:
        pts = np.asarray(st.points, dtype=np.float64)
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
        out.append(need(contact_mm=length * 1000.0,
                        contact_s=length / max(speed, 1e-6) + cfg.settle_time,
                        ink_id=ink_id))
    return out


def dip_segment(from_world: np.ndarray, geo: DipGeometry, dt: float):
    """The world-frame steps of one dip, opening AND closing at ``from_world``.

    Returns (positions (n,3), floor_points (n,3), floor_normals (n,3),
    plunge_step) — the index of the first step at full depth, where the
    charge is credited.
    """
    rim = np.asarray(geo.rim_world, dtype=np.float64)
    start = np.asarray(from_world, dtype=np.float64)
    above_rim = rim + np.array([0.0, 0.0, geo.hover_m])
    transit_z = max(start[2], above_rim[2])
    up = np.array([start[0], start[1], transit_z])
    over = np.array([above_rim[0], above_rim[1], transit_z])
    bottom = rim - np.array([0.0, 0.0, geo.plunge_m])
    settle_n = max(1, int(round(geo.settle_time / dt)))
    dwell_n = max(1, int(round(geo.dwell_s / dt)))

    pos: list[np.ndarray] = []

    def extend(a, b, speed):
        dist = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(dist / (speed * dt))))
        for i in range(1, n + 1):
            pos.append(a + (b - a) * (i / n))

    extend(start, up, geo.travel_speed)
    extend(up, over, geo.travel_speed)
    extend(over, above_rim, geo.travel_speed)
    pos.extend([above_rim.copy() for _ in range(settle_n)])
    extend(above_rim, bottom, geo.plunge_speed)
    plunge_step = len(pos) - 1
    pos.extend([bottom.copy() for _ in range(dwell_n)])
    extend(bottom, above_rim, geo.plunge_speed)
    extend(above_rim, over, geo.travel_speed)
    extend(over, up, geo.travel_speed)
    extend(up, start, geo.travel_speed)
    pos.extend([start.copy() for _ in range(settle_n)])

    positions = np.stack(pos).astype(np.float32)
    n = len(positions)
    floor_pts = np.repeat((rim - np.array([0.0, 0.0, geo.cap_depth_m]))[None, :], n, axis=0)
    floor_nms = np.repeat(np.array([[0.0, 0.0, 1.0]]), n, axis=0)
    return positions, floor_pts.astype(np.float32), floor_nms.astype(np.float32), plunge_step


def dip_steps(from_world: np.ndarray, geo: DipGeometry, dt: float) -> int:
    """How many steps ``dip_segment`` will take from here — for budgeting the
    strokes against the horizon before anything is built."""
    return len(dip_segment(from_world, geo, dt)[0])


@dataclass
class Spliced:
    positions: np.ndarray       # (T', 3) world
    floor_points: np.ndarray    # (T', 3)
    floor_normals: np.ndarray   # (T', 3)
    dip_mask: np.ndarray        # (T',) True on every step of a dip segment
    credit_steps: list[int]     # the step each dip's charge lands on
    dips: list[dict]            # the plan, with the step it was placed at


def splice(positions: np.ndarray, floor_points: np.ndarray, floor_normals: np.ndarray,
           stroke_starts: list[int], plans, geometry_for, dt: float) -> Spliced:
    """Insert one dip segment per plan at that stroke's hover step.

    ``stroke_starts[k]`` is the step where the trajectory begins travelling
    to stroke k (strokes.EETrajectory.stroke_starts) — the tool is at hover
    there. ``geometry_for(plan)`` returns the DipGeometry for its slot.
    """
    plans = sorted(plans, key=lambda d: d.before_stroke)
    out_pos, out_pts, out_nms, out_mask = [], [], [], []
    credit_steps, dips = [], []
    cursor = 0
    for plan in plans:
        at = stroke_starts[plan.before_stroke]
        out_pos.append(positions[cursor:at])
        out_pts.append(floor_points[cursor:at])
        out_nms.append(floor_normals[cursor:at])
        out_mask.append(np.zeros(at - cursor, dtype=bool))
        base = sum(len(p) for p in out_pos)
        seg, pts, nms, plunge = dip_segment(positions[at], geometry_for(plan), dt)
        out_pos.append(seg)
        out_pts.append(pts)
        out_nms.append(nms)
        out_mask.append(np.ones(len(seg), dtype=bool))
        credit_steps.append(base + plunge)
        dips.append({
            "before_stroke": int(plan.before_stroke),
            "slot": plan.slot_id,
            "reason": plan.reason,
            "charge_before_ul": float(plan.charge_before_ul),
            "charge_after_ul": float(plan.charge_after_ul),
            "ink": plan.ink_id,
            "cap_fill_ul": float(getattr(plan, "cap_fill_ul", 0.0)),
            "why_slot": getattr(plan, "why_slot", ""),
            "step": int(base),
            "steps": int(len(seg)),
        })
        cursor = at
    out_pos.append(positions[cursor:])
    out_pts.append(floor_points[cursor:])
    out_nms.append(floor_normals[cursor:])
    out_mask.append(np.zeros(len(positions) - cursor, dtype=bool))
    return Spliced(
        positions=np.concatenate(out_pos).astype(np.float32),
        floor_points=np.concatenate(out_pts).astype(np.float32),
        floor_normals=np.concatenate(out_nms).astype(np.float32),
        dip_mask=np.concatenate(out_mask),
        credit_steps=credit_steps,
        dips=dips,
    )
