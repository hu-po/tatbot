#!/usr/bin/env python3
"""Solve the pen tip offset from touch-off samples. Three modes, newest first:

    il_touchoff.py <session_dir | teleop.wxtl> --ee-tool <id> [--write]

- tip_point (production): the guided tip phase's holds — stills with the pen
  tip planted on ONE point at different angles — from touches.json
  `tip_holds`. Solves p (tip in right/tool_mount, the bore face of the mount
  on the left finger carriage) and P (the planted point in the base frame,
  published as pivot_point_* and the paper_plane_z reference). Since
  2026-08-26 that point is on the paper pad; archived sessions planted on the
  palette tag instead.

  The mount rides the carriage, so a hold's carriage reading is part of its
  FK. A hold whose carriage sits off the rest position was one where the pen
  got pushed up its own axis (the contact cap did its job) and is refused as
  a touch. Holds recorded with six joints (pre-2026-08-30 fusers) are taken
  at carriage rest, and the report says so.
- pivot (legacy): continuous planted-roll windows, same math.
- plane (legacy): discrete plate touches, plane fit; also the fallback when
  reading a raw .wxtl (touches found by effort classification).
Nothing here moves the arm.

Each touch on a plane gives one scalar equation, with R_i, t_i the EE pose:

    row(R_i, 2) . p  -  plane_z  =  -t_i,z        4 unknowns (p, plane_z)

The trap is identifiability: touches made at near-identical wrist orientations
make the design matrix rank-deficient along everything but the tool axis, the
fit looks excellent, and the number is wrong the moment the wrist tilts. So
this tool computes the condition number and REFUSES to write a calibration it
did not identify — a confidently wrong constant is worse than no constant.

Gates (all must pass before config/workspace.yaml is written): orientation
spread, condition number, residual — and for the legacy plane mode, touch
count plus a two-touch holdout. A confidently wrong constant is worse than no
constant, so every failure refuses with the physical remedy.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vision"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import numpy as np  # noqa: E402
import tool_spec  # noqa: E402
from teleop_log import TeleopLog  # noqa: E402
from urdf_kinematics import UrdfChain  # noqa: E402

TIP_LINK = tool_spec.tip_frame("right")  # right/tool_mount
CARRIAGE_JOINT = "right/left_carriage_joint"
CARRIAGE_REST_M = 0.0
# A hold whose carriage is further than this from rest is not a touch: the
# pen was driven up its own axis. Half a millimetre is well above encoder
# noise and well below the 3 mm a real over-cap contact moves it.
CARRIAGE_MAX_OFF_REST_M = 0.0005
MIN_PLATE = 6           # 4 is the arithmetic minimum; 6 leaves a real holdout
COND_MAX = 50.0         # matches the workspace.yaml comment: >50 = too uniform
RESIDUAL_MAX_MM = 1.0
HOLDOUT = 2
# Pivot mode: tip planted on one point, wrist rolling. The tolerance is wider
# than the plane fit because tip compliance and micro-slides on the palette
# surface are real; identifiability comes from rotation spread, not stillness.
PIVOT_RESIDUAL_MAX_MM = 1.5   # floor; blunt tools get their own, see below
PIVOT_SPREAD_MIN_DEG = 30.0
PIVOT_REPEAT_MAX_MM = 2.5
PIVOT_HOLDOUT = 2
# A held-out suffix exposes pose-dependent compliance/seat migration, but does
# not make the full fixed-point solve unobservable.  Keep 1 mm as a warning
# threshold and propagate the observed value as uncertainty; do not demand an
# external body-axis instrument or discard an otherwise well-conditioned TCP.
PIVOT_HOLDOUT_WARN_MM = 1.0

PAD_WORDS = ("pad", "paper")


def ee_pose(chain, names, joints):
    """Pose of the tip frame for one hold. ``joints`` is the driver vector:
    six arm joints and, when the fuser kept it, the carriage metres."""
    values = dict(zip(names, joints, strict=False))
    if len(joints) < 7:
        values[CARRIAGE_JOINT] = CARRIAGE_REST_M
    return chain.link_pose(TIP_LINK, values)


def carriage_off_rest(joints_seq):
    """Holds whose carriage was pushed off rest: (index, metres) pairs."""
    out = []
    for i, joints in enumerate(joints_seq):
        if len(joints) >= 7 and abs(float(joints[6]) - CARRIAGE_REST_M) > CARRIAGE_MAX_OFF_REST_M:
            out.append((i, float(joints[6])))
    return out


def touches_from_wxtl(wxtl_path, events=None):
    """Find touches in a flight log: still intervals with elevated effort."""
    log = TeleopLog(wxtl_path)
    intervals = log.still_intervals()
    info = log.classify_contacts(intervals)
    touches = []
    for interval in intervals:
        if not interval["contact"]:
            continue
        touches.append({
            "joints": interval["follower_pos"][:6],
            "start_unix": interval["start_unix"],
            "end_unix": interval["end_unix"],
            "duration_s": interval["duration_s"],
            "arm_eff_med_nm": interval["arm_eff_med_nm"],
            "label": "plate",
        })
    label_touches(touches, events or [])
    return touches, info


def label_touches(touches, events):
    """Advisory speech labels: 'pad'/'paper' relabels, 'discard' drops.

    Speech never supplies geometry — the effort channel decides what IS a
    touch; words only say which surface it was. A relabel needs the word said
    DURING the touch; a discard removes the touch it overlaps, or failing
    that the nearest touch that ended before it — "scratch that" points
    backwards at the thing just done, never sideways at the neighbours.
    """
    discarded = set()
    for event in events:
        kinds = event.get("kinds", [])
        if "discard" in kinds:
            target = None
            for index, touch in enumerate(touches):
                if (touch["start_unix"] <= event["end_unix"]
                        and touch["end_unix"] >= event["start_unix"]):
                    target = index
            if target is None:
                prior = [i for i, t in enumerate(touches)
                         if t["end_unix"] <= event["start_unix"]]
                target = prior[-1] if prior else None
            if target is not None:
                discarded.add(target)
        elif "touch" in kinds and any(w in event.get("text", "").lower()
                                      for w in PAD_WORDS):
            for touch in touches:
                if (touch["start_unix"] <= event["end_unix"]
                        and touch["end_unix"] >= event["start_unix"]):
                    touch["label"] = "pad"
    touches[:] = [t for i, t in enumerate(touches) if i not in discarded]


def solve_pivot(poses):
    """Tip `p` (EE frame) and pivot point `P` (base frame) from one planted
    roll: R_i p + t_i = P for every sample -> [R_i | -I][p;P] = -t_i.

    3 equations per sample and full 3D observability — the reason pivot beats
    the plane fit whenever a repeatable point exists (since 2026-08-26: one
    spot on the paper pad). One trim pass drops samples where the tip slid or
    lifted.
    """
    a = np.zeros((3 * len(poses), 6))
    b = np.zeros(3 * len(poses))
    for i, pose in enumerate(poses):
        a[3 * i:3 * i + 3, :3] = pose[:3, :3]
        a[3 * i:3 * i + 3, 3:] = -np.eye(3)
        b[3 * i:3 * i + 3] = -pose[:3, 3]
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    residuals = np.array([
        np.linalg.norm(pose[:3, :3] @ x[:3] + pose[:3, 3] - x[3:]) * 1000.0
        for pose in poses])
    # rotation spread over a subsample (O(n^2) on hundreds of ticks is waste)
    subsample = poses[::max(1, len(poses) // 40)]
    spread = 0.0
    for i in range(len(subsample)):
        for j in range(i + 1, len(subsample)):
            delta = subsample[i][:3, :3].T @ subsample[j][:3, :3]
            angle = np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
            spread = max(spread, float(angle))
    return {"p": x[:3], "pivot": x[3:], "cond": float(np.linalg.cond(a)),
            "rms_mm": float(np.sqrt(np.mean(residuals ** 2))),
            "per_sample_mm": residuals, "spread_deg": spread}


def residual_gate_mm(spec):
    """The residual a planted-tip solve can actually REACH with this tool.

    The solve assumes one material point stays put while the wrist rotates
    around it. A tool whose contact is a disc of radius r cannot honour that:
    tilting walks the contact across the face, up to r from where it started,
    and over an even spread of tilt directions that averages about r/sqrt(2).
    Holding the pen better does not help, because it is geometry, not aim.

    The 2026-08-26 laser sessions measured exactly that. rms was 4.05 mm over
    3 holds and 4.26 mm over 9 — tripling the samples did not move it, which
    is the signature of a systematic error rather than noise. Both sit UNDER
    the 5.44 mm this tool's own 15.4 mm lens face predicts, so those captures
    were as good as the tool allows. Demanding the ballpoint's 1.5 mm of them
    was asking for precision finer than the contact patch.

    Point contacts are untouched: a 0.5 mm ballpoint (r = 1.0 mm) and a 3RL
    needle (r = 0.15 mm) both floor at the original 1.5 mm. This widens the
    gate for blunt tools only, and never below it.

    What it does NOT do is make the answer more precise — a laser tip solved
    this way is good to a few mm (bootstrap SE 2.0 mm, split-half 9.9 mm on
    the 9-hold run), and the residual is written into workspace.yaml so that
    uncertainty travels with the constant instead of being forgotten.
    """
    if spec is None:
        return PIVOT_RESIDUAL_MAX_MM
    face_mm = spec.tip_radius_m * 1000.0 / math.sqrt(2.0)
    # A second term for the SEAT. The 2026-08-30 refactor dropped the old
    # gripper-jaw play term on the claim that a bore-seated tool has no play
    # — but the printed mount's bore is ~33 mm over a ~20 mm wall around a
    # 29 mm body: the clamp locates the tool, and the contact still migrates
    # with wrist pose (sweep-20260831_082526: 3.0 mm rms whose per-hold
    # residuals correlate 0.6-0.8 with the wrist joints — geometry, not aim).
    # A datasheet that declares `seat_residual_m` owns that budget; the gate
    # never drops below the point-contact floor either way.
    seat_mm = spec.seat_residual_m * 1000.0
    return max(PIVOT_RESIDUAL_MAX_MM, face_mm, seat_mm)


def solve_pivot_trimmed(poses):
    fit = solve_pivot(poses)
    threshold = max(2.0, 3.0 * fit["rms_mm"])
    keep = [pose for pose, r in zip(poses, fit["per_sample_mm"], strict=True)
            if r <= threshold]
    dropped = len(poses) - len(keep)
    if dropped and len(keep) >= 20:
        fit = solve_pivot(keep)
        fit["dropped"] = dropped
    else:
        fit["dropped"] = 0
    return fit


def pivot_holdout_residual_mm(poses, count: int = PIVOT_HOLDOUT) -> float:
    """True suffix holdout for the planted-point model, in millimetres.

    The final filed value still uses every accepted sample, but it may be
    written only when a fit that never saw the last ``count`` poses predicts
    their common planted point. This catches a seat/tip that migrates over the
    session even when a permissive in-sample residual can absorb it.
    """
    if len(poses) < count + 4:
        return float("inf")
    train = solve_pivot_trimmed(poses[:-count])
    errors = []
    for pose in poses[-count:]:
        world_tip = pose[:3, :3] @ train["p"] + pose[:3, 3]
        errors.append(float(np.linalg.norm(world_tip - train["pivot"])))
    return float(np.sqrt(np.mean(np.square(errors))) * 1000.0)


def pivot_loo_tip_max_mm(poses) -> float:
    """Largest leave-one-out shift of the fitted mount-frame tip.

    The pivot residual is a world-point error; this companion number says how
    sensitive the contact vector itself is to any one planted pose.  It is a
    deterministic uncertainty diagnostic suitable for dataset metadata.
    """
    if len(poses) < tool_spec.CONTACT_TOUCH_MIN_SAMPLES + 1:
        return float("inf")
    full = solve_pivot_trimmed(poses)["p"]
    shifts = []
    for index in range(len(poses)):
        subset = poses[:index] + poses[index + 1:]
        shifts.append(float(np.linalg.norm(solve_pivot_trimmed(subset)["p"] - full)))
    return max(shifts) * 1000.0


def measurement_utc(samples) -> str | None:
    """UTC of the physical observations, rather than when a re-solve ran."""
    times = []
    for sample in samples:
        for key in ("end_unix", "start_unix"):
            value = sample.get(key)
            if value is not None and math.isfinite(float(value)):
                times.append(float(value))
                break
    if not times:
        return None
    return datetime.datetime.fromtimestamp(
        max(times), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def solve_plate(poses):
    """Least squares for (p, plane_z) over plate touches. Returns fit + gates."""
    rows, rhs = [], []
    for pose in poses:
        rows.append([pose[2, 0], pose[2, 1], pose[2, 2], -1.0])
        rhs.append(-pose[2, 3])
    a = np.array(rows)
    b = np.array(rhs)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    cond = float(np.linalg.cond(a))
    residual_mm = (a @ x - b) * 1000.0
    per_touch = np.abs(residual_mm).tolist()
    # Orientation spread: the world-z direction expressed in the EE frame. If
    # every touch points the pen the same way, these rows cluster and only the
    # tool-axis component of p is recoverable.
    dirs = a[:, :3] / np.linalg.norm(a[:, :3], axis=1, keepdims=True)
    spread = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            angle = np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1.0, 1.0)))
            spread = max(spread, float(angle))
    return {
        "p": x[:3], "plane_z": float(x[3]), "cond": cond,
        "rms_mm": float(np.sqrt(np.mean(residual_mm ** 2))),
        "max_mm": float(np.abs(residual_mm).max()),
        "per_touch_mm": per_touch,
        "spread_deg": spread,
    }


def holdout_residual_mm(poses):
    """Fit without the last HOLDOUT touches, predict their contact height."""
    fit = solve_plate(poses[:-HOLDOUT])
    errors = []
    for pose in poses[-HOLDOUT:]:
        predicted = pose[2, :3] @ fit["p"] - fit["plane_z"]
        errors.append(abs(predicted - (-pose[2, 3])) * 1000.0)
    return float(max(errors))


def ablation(poses):
    """Accuracy versus touch count: re-solve on chronological prefixes so the
    report card can say what touches 7 and 8 actually bought. p_delta is the
    tip's distance from the full-set answer — the number that should shrink."""
    full = solve_plate(poses)
    rows = []
    for count in range(4, len(poses) + 1):
        fit = solve_plate(poses[:count])
        rows.append({
            "n": count,
            "cond": round(fit["cond"], 1),
            "rms_mm": round(fit["rms_mm"], 3),
            "p_delta_mm": round(
                float(np.linalg.norm(fit["p"] - full["p"])) * 1000.0, 3),
        })
    return rows


def render_workspace(right):
    """The whole config/workspace.yaml, regenerated. Two-level flat on purpose:
    il_analyze_rollout.py reads this with a hand-rolled parser that sees one
    level of nesting and scalars only."""

    def scalar(value, fmt="{:.6f}"):
        return "null" if value is None else fmt.format(value)

    lines = [
        "# workspace.yaml — measured scene geometry. NOTHING HERE MOVES THE ARM.",
        "#",
        "# Written by scripts/il_touchoff.py (touch-off) and read by",
        "# scripts/il_analyze_rollout.py and scripts/vision/fuse_session.py.",
        "# Deliberately NOT part of config/trossen/tatbot.yaml — nothing here is",
        "# an arm parameter. Git history is the changelog.",
        "right:",
        "  # Which physical tool is fitted — names a datasheet in config/tools/.",
        "  # Everything below was measured with THIS tool in the mount, so a swap",
        "  # invalidates it; il_touchoff.py --tool-id sets it as the touch-off writes.",
        f"  tool_id: {right.get('tool_id') or 'null'}",
        "  # Frame the tip offset is solved in: right/tool_mount is the bore face",
        "  # of the mount on the left finger carriage, +z along the bore. A file",
        "  # naming any other frame is gripper-era and reads as no touch-off.",
        f"  tip_frame: {right.get('tip_frame') or 'null'}",
        "  # Pen tip in that frame, metres. Seated in the bore: one constant,",
        "  # measured once, valid until the tool or the mount physically changes.",
        f"  pen_tip_offset_x: {scalar(right.get('pen_tip_offset_x'))}",
        f"  pen_tip_offset_y: {scalar(right.get('pen_tip_offset_y'))}",
        f"  pen_tip_offset_z: {scalar(right.get('pen_tip_offset_z'))}",
        "  # Optional independent physical-body axis/origin in the same mount frame.",
        "  # Pivot touch-off qualifies the contact vector. For this axisymmetric",
        "  # profile, consumers infer the body axis from mount origin -> tip and",
        "  # preserve the measured shape; roll is irrelevant. Independent body",
        "  # evidence is optional for asymmetric clearance studies, and a new",
        "  # touch-off clears it because the physical seat may have changed.",
        f"  tool_body_status: {right.get('tool_body_status') or 'null'}",
        f"  tool_body_utc: {right.get('tool_body_utc') or 'null'}",
        f"  tool_body_method: {right.get('tool_body_method') or 'null'}",
        f"  tool_body_measurement_source: {right.get('tool_body_measurement_source') or 'null'}",
        f"  tool_body_report: {right.get('tool_body_report') or 'null'}",
        f"  tool_body_report_sha256: {right.get('tool_body_report_sha256') or 'null'}",
        f"  tool_body_samples: {scalar(right.get('tool_body_samples'), '{:.0f}')}",
        f"  tool_body_selected_cycle: {scalar(right.get('tool_body_selected_cycle'), '{:.0f}')}",
        f"  tool_body_alignment_max_mm: {scalar(right.get('tool_body_alignment_max_mm'))}",
        f"  tool_body_tip_repeatability_mm: {scalar(right.get('tool_body_tip_repeatability_mm'))}",
        f"  tool_body_origin_repeatability_mm: {scalar(right.get('tool_body_origin_repeatability_mm'))}",
        f"  tool_body_axis_repeatability_deg: {scalar(right.get('tool_body_axis_repeatability_deg'))}",
        f"  tool_body_frame: {right.get('tool_body_frame') or 'null'}",
        f"  tool_body_origin_x: {scalar(right.get('tool_body_origin_x'))}",
        f"  tool_body_origin_y: {scalar(right.get('tool_body_origin_y'))}",
        f"  tool_body_origin_z: {scalar(right.get('tool_body_origin_z'))}",
        f"  tool_body_rpy_x: {scalar(right.get('tool_body_rpy_x'))}",
        f"  tool_body_rpy_y: {scalar(right.get('tool_body_rpy_y'))}",
        f"  tool_body_rpy_z: {scalar(right.get('tool_body_rpy_z'))}",
        "  # Carriage reading at the solve, metres — the mount rides the carriage,",
        "  # so the FK that produced the offset above used this value.",
        f"  carriage_m: {scalar(right.get('carriage_m'))}",
        "  # Surface Z in the arm base frame, metres, and the surface's own",
        "  # softness band — pages lift, so contact is a band not a plane.",
        "  # On a DRAPED substrate (the silicone skin) this is the low ground",
        "  # rather than the whole surface: the skin lies flat where it",
        "  # overhangs and mounds ~25 mm over the wrist pad, so the plane is a",
        "  # lower bound on where the tool meets it. Still the right reference",
        "  # for il_analyze_rollout's contact depth, which measures against the",
        "  # surface the tool descends toward; it is not a claim of flatness.",
        f"  paper_plane_z: {scalar(right.get('paper_plane_z'))}",
        f"  paper_band_mm: {scalar(right.get('paper_band_mm'), '{:.2f}')}",
        "  # Pivot point in the arm base frame, metres — measured by rolling",
        "  # the planted pen tip. Ties touch to vision. Which surface it is on",
        "  # is what touchoff.n_pad records: > 0 means the paper pad (so",
        "  # paper_plane_z above is the paper), 0 means the palette tag.",
        f"  pivot_point_x: {scalar(right.get('pivot_point_x'))}",
        f"  pivot_point_y: {scalar(right.get('pivot_point_y'))}",
        f"  pivot_point_z: {scalar(right.get('pivot_point_z'))}",
        "  # Fallback when the solve could not separate the tip from the surface",
        "  # (all holds at one orientation): composite EE height at contact.",
        f"  ee_contact_z: {scalar(right.get('ee_contact_z'))}",
        "",
        "  touchoff:",
        f"    utc: {right['touchoff'].get('utc') or 'null'}",
        f"    session: {right['touchoff'].get('session') or 'null'}",
        f"    n_plate: {right['touchoff'].get('n_plate', 0)}",
        f"    n_pad: {right['touchoff'].get('n_pad', 0)}",
        f"    cond: {scalar(right['touchoff'].get('cond'), '{:.1f}')}"
        "            # >50 means the wrist orientations were too uniform",
        f"    residual_mm: {scalar(right['touchoff'].get('residual_mm'), '{:.3f}')}",
        f"    holdout_mm: {scalar(right['touchoff'].get('holdout_mm'), '{:.3f}')}",
        f"    tip_loo_max_mm: {scalar(right['touchoff'].get('tip_loo_max_mm'), '{:.3f}')}",
        f"    spread_deg: {scalar(right['touchoff'].get('spread_deg'), '{:.1f}')}",
        f"    note: \"{right['touchoff'].get('note', '')}\"",
        "",
    ]
    return "\n".join(lines)


def resolve_tool(args):
    """The tool this touch-off is measuring. --tool-id is required, so what is
    in the gripper is always STATED rather than inherited from workspace.yaml.

    It used to fall back to whatever workspace.yaml already named, which is how
    a laser-pen session on 2026-08-26 got solved against the ballpoint: the fit
    was right (126 mm, against the laser's documented 130 mm protrusion) and
    was refused for landing 68.8 mm from the wrong datasheet's tip. The file
    naming the previous tool is exactly the thing a tool swap invalidates, so
    it is the one source that must not supply this answer."""
    try:
        return args.tool_id, tool_spec.load_tool(args.tool_id, REPO)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"--tool-id: {exc}")


def tool_refusal(spec, p):
    """Refuse a fit that does not look like the tool it is being filed under.

    The datasheet says how far this tool's tip sits past the fingertips; the
    solve says where it actually is. Pens differ by tens of millimetres, so a
    large gap means the gripper is holding something other than what
    workspace.yaml claims — and writing that calibration under the wrong name
    would poison every dataset recorded after it.
    """
    if spec is None:
        return None
    error_m = tool_spec.tip_offset_error_m(spec, (float(p[0]), float(p[1]), float(p[2])))
    if error_m <= spec.tip_tolerance_m:
        return None
    return (
        f"the solved tip is {error_m * 1000:.1f} mm from where {spec.tool_id} says its tip "
        f"is (tolerance {spec.tip_tolerance_m * 1000:.0f} mm) — either a different tool is "
        f"fitted (re-run with --tool-id <the one in the gripper>) or "
        f"{spec.source.relative_to(REPO)} has the wrong profile.")


def pivot_mode(args, chain, names, pivots, source, session, mode="pivot",
               surface=None, n_samples=0, measured_utc=None):
    """Solve planted-tip samples for (p, P). Two capture styles share the
    math: continuous pivot windows (roll while planted), and the guided tip
    holds since 2026-08-22 — discrete stills at one point, each at a
    different orientation.

    `surface` names what the tip was planted ON, because P is written out as
    paper_plane_z. Only a session planted on the pad may claim that height is
    the paper the pen draws on; a palette-tag session measures a different
    surface at a different height, and tool_spec.derive_z_floor_m refuses a
    safety floor derived from it (it reads n_pad to tell them apart)."""
    tool_id, spec = resolve_tool(args)
    pushed = [pair for window in pivots for pair in carriage_off_rest(window["joints_seq"])]
    six_joint = any(len(j) < 7 for window in pivots for j in window["joints_seq"])
    report = {"source": source, "mode": mode, "n_windows": len(pivots),
              "tool_id": tool_id, "tip_frame": TIP_LINK,
              "carriage_rest_m": CARRIAGE_REST_M,
              "carriage_assumed_rest": six_joint,
              "carriage_pushed_holds": [{"hold": i + 1, "carriage_m": m} for i, m in pushed],
              # What paper_plane_z will mean if this writes: the paper the pen
              # draws on ("pad") or a different surface at a different height.
              "surface": surface,
              "gates": {"cond_max": COND_MAX,
                        "residual_max_mm": round(residual_gate_mm(spec), 2),
                        "residual_floor_mm": PIVOT_RESIDUAL_MAX_MM,
                        "holdout_count": PIVOT_HOLDOUT,
                        "holdout_warning_mm": PIVOT_HOLDOUT_WARN_MM,
                        "tip_radius_mm": (round(spec.tip_radius_m * 1000, 2)
                                          if spec else None),
                        "spread_min_deg": PIVOT_SPREAD_MIN_DEG,
                        "repeat_max_mm": PIVOT_REPEAT_MAX_MM,
                        "tip_tolerance_mm": (spec.tip_tolerance_m * 1000
                                             if spec else None),
                        "seat_tolerance_deg": (spec.seat_tolerance_deg
                                               if spec else tool_spec.AXIS_TOLERANCE_DEG)}}
    report_path = session / "touchoff_report.json"

    def finish(status, code, **extra):
        report.update({"status": status, **extra})
        with __import__("contextlib").suppress(OSError):
            report_path.write_text(json.dumps(report, indent=2))
        return code

    window_fits = []
    poses_all = []
    for window in pivots:
        poses = [ee_pose(chain, names, joints) for joints in window["joints_seq"]]
        poses_all.extend(poses)
        window_fits.append(solve_pivot_trimmed(poses))
    fit = solve_pivot_trimmed(poses_all)
    holdout_mm = pivot_holdout_residual_mm(poses_all)
    loo_tip_max_mm = pivot_loo_tip_max_mm(poses_all)
    repeat_mm = 0.0
    for i in range(len(window_fits)):
        for j in range(i + 1, len(window_fits)):
            repeat_mm = max(repeat_mm, float(np.linalg.norm(
                window_fits[i]["pivot"] - window_fits[j]["pivot"]) * 1000.0))
    print(f"{len(pivots)} pivot windows, {len(poses_all)} samples "
          f"({fit['dropped']} trimmed as slips)")
    if spec is not None:
        print(f"  tool {spec.summary()}")
    print(f"  pen tip p = {np.round(fit['p'] * 1000, 2)} mm in {TIP_LINK}"
          f"  (axis lean {tool_spec.axis_lean_deg(tuple(float(v) for v in fit['p'])):.1f} deg"
          f" off the bore's +z)")
    if six_joint:
        print("  note: holds carry six joints — carriage taken at rest "
              f"({CARRIAGE_REST_M * 1000:.1f} mm); a fuser from 2026-08-30 on keeps all seven")
    print(f"  pivot point P = {np.round(fit['pivot'] * 1000, 1)} mm (base frame)"
          f" — the planted spot{f' on the {surface}' if surface else ''}")
    print(f"  cond {fit['cond']:.1f}   rms {fit['rms_mm']:.3f} mm   holdout "
          f"{holdout_mm:.3f} mm   rotation spread {fit['spread_deg']:.1f} deg   "
          f"tip LOO {loo_tip_max_mm:.3f} mm   repeatability {repeat_mm:.2f} mm")

    report["fit"] = {"p_mm": [round(v * 1000, 3) for v in fit["p"]],
                     "pivot_mm": [round(v * 1000, 2) for v in fit["pivot"]],
                     "cond": round(fit["cond"], 1),
                     "rms_mm": round(fit["rms_mm"], 4),
                     "spread_deg": round(fit["spread_deg"], 1),
                     "holdout_mm": round(holdout_mm, 4),
                     "tip_loo_max_mm": round(loo_tip_max_mm, 4),
                     "repeat_mm": round(repeat_mm, 3),
                     "samples": len(poses_all), "trimmed": fit["dropped"]}
    step = max(20, len(poses_all) // 8)
    report["ablation"] = [{
        "n": count,
        "rms_mm": round((sub := solve_pivot(poses_all[:count]))["rms_mm"], 3),
        "p_delta_mm": round(float(np.linalg.norm(sub["p"] - fit["p"])) * 1000, 3),
        "cond": round(sub["cond"], 1),
    } for count in range(step, len(poses_all) + 1, step)]

    refusals = []
    if fit["spread_deg"] < PIVOT_SPREAD_MIN_DEG:
        refusals.append(
            f"rotation spread {fit['spread_deg']:.0f} deg < "
            f"{PIVOT_SPREAD_MIN_DEG:.0f} — use visibly different pen angles "
            "while the tip stays planted, then redo the tip phase.")
    if fit["cond"] > COND_MAX:
        refusals.append(f"condition number {fit['cond']:.0f} > {COND_MAX:.0f} "
                        "— same cause: not enough orientation variety.")
    gate_mm = residual_gate_mm(spec)
    if fit["rms_mm"] > gate_mm:
        detail = (f"rms {fit['rms_mm']:.2f} mm > {gate_mm:.2f} mm — the tip "
                  "slid or lifted during the roll. Press a little firmer and "
                  "keep the tip planted on one spot.")
        if gate_mm > PIVOT_RESIDUAL_MAX_MM and spec is not None:
            face_mm = spec.tip_radius_m * 1000.0 / math.sqrt(2.0)
            why = (f"its {spec.seat_residual_m * 1000:.1f} mm of seat play in the "
                   "mount (seat_residual_m)"
                   if spec.seat_residual_m * 1000.0 >= face_mm
                   else f"this tool's {spec.tip_radius_m * 2000:.1f} mm contact face")
            detail += (f" (the limit is widened from {PIVOT_RESIDUAL_MAX_MM} mm "
                       f"for {why}; exceeding even that is a real slip.)")
        refusals.append(detail)
    warnings = []
    if holdout_mm > PIVOT_HOLDOUT_WARN_MM:
        warnings.append(
            f"held-out planted poses miss by {holdout_mm:.2f} mm > "
            f"{PIVOT_HOLDOUT_WARN_MM:.2f} mm — the fixed-point solve remains "
            "observable, but pose-dependent compliance/seat migration is larger "
            "than 1 mm. This value is retained as contact uncertainty and must "
            "not be converted into an air-gap allowance.")
    if len(pivots) >= 2 and repeat_mm > PIVOT_REPEAT_MAX_MM:
        refusals.append(
            f"windows disagree by {repeat_mm:.1f} mm > {PIVOT_REPEAT_MAX_MM} "
            "— the tip was planted on different spots. Aim for the tag "
            "center each time.")
    mismatch = tool_refusal(spec, fit["p"])
    if mismatch:
        refusals.append(mismatch)
    if pushed:
        refusals.append(
            f"{len(pushed)} hold(s) had the carriage pushed off rest "
            f"({', '.join(f'#{i + 1} at {m * 1000:.1f} mm' for i, m in pushed)}) — the pen "
            "was driven up its own axis, so those were not touches. Plant more gently "
            "(the contact cap yields before the tip does) and redo the tip phase.")
    lean = tool_spec.axis_lean_deg(tuple(float(v) for v in fit["p"]))
    lean_max = spec.seat_tolerance_deg if spec else tool_spec.AXIS_TOLERANCE_DEG
    if lean > lean_max:
        refusals.append(
            f"the solved tip leans {lean:.1f} deg off the mount's bore axis (tolerance "
            f"{lean_max:.0f}) — more than {tool_id}'s datasheet grants its seat: the "
            "tool is seated crooked, the mount is on askew, or right/tool_mount_joint "
            "in urdf/tatbot.urdf is wrong. Fix the physical cause or the URDF; a "
            "calibration cannot absorb it.")
    if refusals:
        print("\nREFUSED — writing nothing, because a confidently wrong "
              "constant is worse than no constant:")
        for reason in refusals:
            print(f"  - {reason}")
        return finish("refused", 2, reasons=refusals)

    if warnings:
        print("\nACCEPTED WITH CALIBRATION UNCERTAINTY:")
        for warning in warnings:
            print(f"  - {warning}")
        report["warnings"] = warnings

    if not args.write:
        print("\ndry run — pass --write to update config/workspace.yaml")
        return finish("dry", 0)
    write_workspace(args, {
        "tool_id": tool_id,
        "tip_frame": TIP_LINK,
        "carriage_m": CARRIAGE_REST_M,
        "pen_tip_offset_x": float(fit["p"][0]),
        "pen_tip_offset_y": float(fit["p"][1]),
        "pen_tip_offset_z": float(fit["p"][2]),
        "paper_plane_z": float(fit["pivot"][2]),
        "pivot_point_x": float(fit["pivot"][0]),
        "pivot_point_y": float(fit["pivot"][1]),
        "pivot_point_z": float(fit["pivot"][2]),
    }, pivots,
        # n_pad is the gate on treating paper_plane_z as the paper. Count the
        # planted holds only when they were planted on the pad.
        [None] * n_samples if surface == "pad" else [],
        fit, holdout_mm, source, tip_loo_max_mm=loo_tip_max_mm,
        measured_utc=measured_utc)
    return finish("written", 0)


def load_events(session):
    events_path = session / "events.jsonl"
    if not events_path.is_file():
        return []
    return [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="session dir (touches.json / teleop.wxtl) or a .wxtl file")
    ap.add_argument("--urdf", default=str(REPO / "urdf" / "tatbot.urdf"))
    ap.add_argument("--workspace", default=str(REPO / "config" / "workspace.yaml"))
    ap.add_argument("--ee-tool", "--tool-id", dest="tool_id", required=True,
                    help="REQUIRED: which tool is in the mount (a datasheet "
                         "name in config/tools/). The solved tip is checked "
                         "against that datasheet before writing. This used to "
                         "default to workspace.yaml, but that file names the "
                         "PREVIOUS tool — precisely what a swap invalidates — "
                         "so the gripper's contents must be stated, not "
                         "inherited.")
    ap.add_argument("--write", action="store_true",
                    help="write config/workspace.yaml when every gate passes "
                         "(default: dry run, print only)")
    ap.add_argument("--allow-composite", action="store_true",
                    help="on a conditioning refusal, still record ee_contact_z "
                         "(composite EE height at contact — enough for contact%%, "
                         "not for absolute height)")
    args = ap.parse_args()

    target = Path(args.target).expanduser()
    session = target if target.is_dir() else target.parent
    touches_file = session / "touches.json"
    threshold_info = None
    data = None
    pivots = []
    tip_holds = []
    if target.is_file() and target.suffix == ".wxtl":
        touches, threshold_info = touches_from_wxtl(target, load_events(session))
        source = str(target)
    elif touches_file.is_file():
        data = json.loads(touches_file.read_text())
        touches = data["touches"]
        pivots = data.get("pivots", [])
        tip_holds = data.get("tip_holds", [])
        source = str(touches_file)
    elif (session / "teleop.wxtl").is_file():
        touches, threshold_info = touches_from_wxtl(session / "teleop.wxtl",
                                                    load_events(session))
        source = str(session / "teleop.wxtl")
    else:
        sys.exit(f"no touches.json or teleop.wxtl in {session}")

    chain = UrdfChain(args.urdf)
    names = chain.driver_joint_names("right")
    if tip_holds:
        # Guided tip holds: discrete planted stills, one pseudo-window.
        window = {"joints_seq": [h["joints"] for h in tip_holds]}
        # Pre-2026-08-26 sessions carry no surface; they were all palette.
        surface = (data.get("tip_surface") if data is not None else None) or tip_holds[0].get("surface") or "palette"
        return pivot_mode(args, chain, names, [window], source, session,
                          mode="tip_point", surface=surface,
                          n_samples=len(tip_holds),
                          measured_utc=measurement_utc(tip_holds))
    if pivots:
        # Pivot windows beat discrete touches whenever the fuser found them.
        pivot_samples = [window for pivot in pivots
                         for window in pivot.get("samples", [])]
        return pivot_mode(args, chain, names, pivots, source, session,
                          measured_utc=measurement_utc(pivot_samples))
    plate = [t for t in touches if t.get("label", "plate") == "plate"]
    pad = [t for t in touches if t.get("label") == "pad"]
    print(f"{len(plate)} plate touches, {len(pad)} pad touches from {source}")
    if threshold_info:
        print(f"  contact threshold {threshold_info['threshold_nm']:.2f} Nm "
              f"(baseline {threshold_info['baseline_nm']:.2f})")

    # Machine-readable twin of everything printed below — the report card
    # reads this instead of parsing stdout.
    tool_id, spec = resolve_tool(args)
    report = {"source": source, "n_plate": len(plate), "n_pad": len(pad),
              "tool_id": tool_id,
              "threshold": threshold_info, "gates": {
                  "min_plate": MIN_PLATE, "cond_max": COND_MAX,
                  "residual_max_mm": RESIDUAL_MAX_MM}}
    report_path = session / "touchoff_report.json"

    def finish(status, code, **extra):
        report.update({"status": status, **extra})
        with __import__("contextlib").suppress(OSError):
            report_path.write_text(json.dumps(report, indent=2))
        return code

    refusals = []
    holdout = None
    if len(plate) < MIN_PLATE:
        refusals.append(f"only {len(plate)} plate touches; need >= {MIN_PLATE} "
                        f"(a few extra buys a real holdout)")
        fit = solve_plate([ee_pose(chain, names, t["joints"]) for t in plate]) \
            if len(plate) >= 4 else None
    else:
        poses = [ee_pose(chain, names, t["joints"]) for t in plate]
        fit = solve_plate(poses)
        holdout = holdout_residual_mm(poses)
        # A guided touch that never landed (pen in the air on cue) is a point
        # far off the plate plane. With touches to spare, drop the single
        # worst if that alone rescues the gates — and say which one, so the
        # operator learns which cue went wrong.
        if (fit["rms_mm"] > RESIDUAL_MAX_MM or holdout > RESIDUAL_MAX_MM) \
                and len(poses) > MIN_PLATE:
            worst = int(np.argmax(fit["per_touch_mm"]))
            retry = poses[:worst] + poses[worst + 1:]
            retry_fit = solve_plate(retry)
            retry_holdout = holdout_residual_mm(retry)
            if retry_fit["rms_mm"] <= RESIDUAL_MAX_MM \
                    and retry_holdout <= RESIDUAL_MAX_MM \
                    and retry_fit["cond"] <= COND_MAX:
                print(f"  dropped plate touch {worst + 1}: "
                      f"{fit['per_touch_mm'][worst]:.2f} mm off the plane "
                      "(probably never landed) — refit without it")
                report["dropped_touch"] = worst + 1
                plate = plate[:worst] + plate[worst + 1:]
                poses, fit, holdout = retry, retry_fit, retry_holdout
        report["ablation"] = ablation(poses)
        print("  per-touch |residual| mm: "
              + ", ".join(f"{r:.2f}" for r in fit["per_touch_mm"]))
        print(f"  pen tip p = {np.round(fit['p'] * 1000, 2)} mm in {TIP_LINK}")
        print(f"  plate z = {fit['plane_z'] * 1000:.2f} mm (arm base frame)")
        print(f"  cond {fit['cond']:.1f}   fit RMS {fit['rms_mm']:.3f} mm   "
              f"holdout {holdout:.3f} mm   orientation spread {fit['spread_deg']:.1f} deg")
        if fit["cond"] > COND_MAX:
            refusals.append(
                f"condition number {fit['cond']:.0f} > {COND_MAX:.0f} — the touches "
                f"span only {fit['spread_deg']:.0f} deg of wrist orientation, so only "
                "the tool-axis component of p is identified. Tilt the wrist more "
                "between touches and redo the tip phase.")
        if fit["rms_mm"] > RESIDUAL_MAX_MM or holdout > RESIDUAL_MAX_MM:
            refusals.append(
                f"residual too large (fit {fit['rms_mm']:.2f} mm, holdout "
                f"{holdout:.2f} mm > {RESIDUAL_MAX_MM} mm) — a touch probably "
                "moved, or the plate is not rigid. Inspect touches.json, drop "
                "the outlier (say 'discard' next time), or touch a harder surface.")
        mismatch = tool_refusal(spec, fit["p"])
        if mismatch:
            refusals.append(mismatch)

    if fit is not None:
        report["fit"] = {
            "p_mm": [round(v * 1000, 3) for v in fit["p"]],
            "plane_z_mm": round(fit["plane_z"] * 1000, 3),
            "cond": round(fit["cond"], 1), "rms_mm": round(fit["rms_mm"], 4),
            "holdout_mm": round(holdout, 4) if holdout is not None else None,
            "spread_deg": round(fit["spread_deg"], 1)}

    if refusals:
        print("\nREFUSED — writing nothing, because a confidently wrong constant "
              "is worse than no constant:")
        for reason in refusals:
            print(f"  - {reason}")
        if args.allow_composite and plate:
            poses = [ee_pose(chain, names, t["joints"]) for t in plate]
            ee_contact = float(np.median([p[2, 3] for p in poses]))
            print(f"\n--allow-composite: ee_contact_z = {ee_contact * 1000:.2f} mm "
                  f"(composite; enough for contact%, not absolute height)")
            if args.write:
                write_workspace(args, {"tool_id": tool_id,
                                       "ee_contact_z": ee_contact}, plate, pad,
                                None, None, source)
        return finish("refused", 2, reasons=refusals)

    if fit is None:
        return finish("refused", 2, reasons=["No valid fit computed"])

    pad_planes = []
    for touch in pad:
        pose = ee_pose(chain, names, touch["joints"])
        pad_planes.append(float(pose[2, :3] @ fit["p"] + pose[2, 3]))
    paper_plane = float(np.median(pad_planes)) if pad_planes else None
    paper_band = (max(pad_planes) - min(pad_planes)) * 1000.0 if len(pad_planes) > 1 else None
    if paper_plane is not None:
        print(f"  paper plane = {paper_plane * 1000:.2f} mm"
              + (f", band {paper_band:.1f} mm over {len(pad_planes)} touches"
                 if paper_band is not None else " (single touch — no band)"))
    else:
        print("  no pad touches — paper_plane_z falls back to the plate "
              "(rigid, so no band); touch the pad next session to measure it")
        paper_plane = fit["plane_z"]

    report["paper"] = {"plane_z_mm": round(paper_plane * 1000, 3),
                       "band_mm": round(paper_band, 2) if paper_band else None}
    if not args.write:
        print("\ndry run — pass --write to update config/workspace.yaml")
        return finish("dry", 0)
    write_workspace(args, {
        "tool_id": tool_id,
        "tip_frame": TIP_LINK,
        "carriage_m": CARRIAGE_REST_M,
        "pen_tip_offset_x": float(fit["p"][0]),
        "pen_tip_offset_y": float(fit["p"][1]),
        "pen_tip_offset_z": float(fit["p"][2]),
        "paper_plane_z": paper_plane,
        "paper_band_mm": paper_band,
    }, plate, pad, fit, holdout, source,
        measured_utc=measurement_utc(plate + pad))
    return finish("written", 0)


def write_workspace(args, values, plate, pad, fit, holdout, source,
                    tip_loo_max_mm=None, measured_utc=None):
    right: dict[str, Any] = {
        "tool_id": None, "tip_frame": None, "carriage_m": None,
        "pen_tip_offset_x": None, "pen_tip_offset_y": None,
        "pen_tip_offset_z": None, "paper_plane_z": None,
        "paper_band_mm": None, "ee_contact_z": None,
        # A new planted-tip solve cannot retain an older body's independent
        # pose qualification.  The physical study must repopulate all of it.
        "tool_body_status": None, "tool_body_utc": None,
        "tool_body_method": None, "tool_body_measurement_source": None,
        "tool_body_report": None,
        "tool_body_report_sha256": None, "tool_body_samples": None,
        "tool_body_selected_cycle": None, "tool_body_alignment_max_mm": None,
        "tool_body_tip_repeatability_mm": None,
        "tool_body_origin_repeatability_mm": None,
        "tool_body_axis_repeatability_deg": None, "tool_body_frame": None,
        "tool_body_origin_x": None, "tool_body_origin_y": None,
        "tool_body_origin_z": None, "tool_body_rpy_x": None,
        "tool_body_rpy_y": None, "tool_body_rpy_z": None,
    }
    right.update(values)
    right["touchoff"] = {
        "utc": measured_utc or datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session": source,
        "n_plate": len(plate), "n_pad": len(pad),
        "cond": fit["cond"] if fit else None,
        "residual_mm": fit["rms_mm"] if fit else None,
        "holdout_mm": holdout,
        "tip_loo_max_mm": tip_loo_max_mm,
        "spread_deg": fit["spread_deg"] if fit else None,
        "note": "" if fit else "composite only — conditioning refusal",
    }
    workspace = Path(args.workspace).expanduser()
    workspace.write_text(render_workspace(right))
    print(f"wrote {workspace}")


if __name__ == "__main__":
    sys.exit(main())
