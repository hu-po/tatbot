#!/usr/bin/env python3
"""Score a rollout: did the pen draw the shape, and did the control loop keep time?

    il_analyze_rollout.py <run-dir | flight.csv> [--paper-z MM] [--settle S]
    il_analyze_rollout.py --compare <analysis.json>...      # the sweep table

Reads the follower's flight-recorder CSV, pushes the six arm joints through the
URDF with scripts/vision/urdf_kinematics.py, and reports what the pen tip did.
Writes analysis.json beside the CSV and prints one table.

WHAT THIS CAN AND CANNOT MEASURE. Forward kinematics stops at
`right/ee_gripper_link`. The tattoo pen extends past that and the URDF does
not model it; once scripts/il_touchoff.py has measured the tip offset it is
applied to the FK path here, and ABSOLUTE height above paper is meaningful.
Until then every height is relative.
Without one this still reports every height number, but marks contact_basis
"inferred" and valid=false — those numbers rank runs against EACH OTHER within
one session and must not be compared across setups. The 2026-08-21 DiT sweep
was scored that way and its draw/hover gap (2.5 mm) was smaller than the
unmodelled offset it was measured against.

Never imports lerobot: the torch import would make the post-rollout hook felt,
and this runs while the operator is waiting.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vision"))

try:
    import numpy as np
except ImportError:  # a bare system python3 has no numpy; uv always can get it.
    sys.exit("il_analyze_rollout needs numpy — run:\n"
             "  uv run --no-project --with numpy python scripts/il_analyze_rollout.py "
             + " ".join(sys.argv[1:]))

from urdf_kinematics import UrdfChain  # noqa: E402

# --- window ---------------------------------------------------------------
# Every run starts with the arm at staged_positions and the policy wanting a
# different pose; the traverse fires max_relative_target clamps for ~1.5 s and
# the first strokes are transit, not task. 8 s covered all five runs of the
# 2026-08-21 sweep with margin. FIXED, not auto-detected, on purpose: an
# adaptive window makes two runs incomparable, which defeats the point. The
# detected settle time is still reported, as a diagnostic — if it ever exceeds
# this, raise it for EVERY run, not one.
SETTLE_S = 8.0

# --- geometry -------------------------------------------------------------
# The pen is seated in the mount on the left finger carriage — it cannot shift
# in the mount, but the CARRIAGE moves it along its own axis, which is why the
# FK below carries the carriage column: at rest the tip is one fixed offset
# from the mount, and a retract shows up as the tip lifting. What is soft is
# the target: the paper pad's pages puff up and lift, giving roughly 5 mm of
# play in the surface itself. So "on the surface" is a band, not a plane, and
# a tolerance tighter than the pad's own movement would be measuring noise.
CONTACT_TOL_MM = 5.0
INFERRED_PLANE_PCT = 1.0  # uncalibrated fallback: the run's own 1st-pct Z

# --- path metrics ---------------------------------------------------------
# Turn and reversal are computed on the path RESAMPLED TO UNIFORM ARC LENGTH.
# Per-sample direction change is confounded by speed AND by tick rate — a slow
# policy takes small steps so its direction noise dominates, and a 135 ms stall
# stretches one "sample" into four. Resampling removes both. The per-sample
# numbers are kept as *_per_sample for continuity with the 2026-08-21 table.
RESAMPLE_MM = 0.5
MIN_STEP_MM = 0.05        # shorter than this is encoder noise, and its
                          # direction is meaningless — drop before any angle
REVERSAL_DEG = 90.0
STALL_S = 0.05

ARM_JOINTS = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
CARRIAGE = "left_carriage_joint"
TIP_LINK = "right/tool_mount"  # the frame workspace.yaml's tip offset lives in


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def pen_path(rows, chain, names, tip_offset_m=None):
    """EE positions in mm — with the measured pen tip offset applied, when one
    exists. The offset is a full 3-vector in the EE frame: applying only its z
    would go wrong the moment the wrist tilts, since the tip swings on the
    lever arm of its x/y components."""
    offset = np.zeros(3) if tip_offset_m is None else np.asarray(tip_offset_m, float)
    out = []
    for r in rows:
        q = {n: float(r["pos_" + j]) for n, j in zip(names, ARM_JOINTS, strict=True)}
        # the mount rides the carriage: a retracted pen is a lifted tip
        q["right/" + CARRIAGE] = float(r.get("pos_" + CARRIAGE) or 0.0)
        pose = chain.link_pose(TIP_LINK, q)
        out.append((pose[:3, :3] @ offset + pose[:3, 3]) * 1000.0)
    return np.array(out)


def resample_arclength(xy, step_mm=RESAMPLE_MM):
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < step_mm * 4:
        return xy[:1]
    grid = np.arange(0.0, s[-1], step_mm)
    return np.stack([np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])], 1)


def turn_stats(xy):
    d = np.diff(xy, axis=0)
    d = d[np.linalg.norm(d, axis=1) > MIN_STEP_MM]
    if len(d) < 2:
        return float("nan"), float("nan")
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    dot = np.einsum("ij,ij->i", d[:-1], d[1:])
    ang = np.degrees(np.abs(np.arctan2(cross, dot)))
    return float(ang.mean()), float(100.0 * (ang > REVERSAL_DEG).mean())


def hull_area(xy) -> float:
    """Convex-hull area — a bounding box lies about a diagonal scribble."""
    pts = np.unique(np.round(xy, 3), axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(points):
        stack = []
        for p in points:
            while len(stack) >= 2:
                a, b = stack[-2], stack[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0:
                    break
                stack.pop()
            stack.append(p)
        return stack

    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    if len(hull) < 3:
        return 0.0
    h = np.array(hull)
    return float(abs(np.dot(h[:, 0], np.roll(h[:, 1], -1))
                     - np.dot(h[:, 1], np.roll(h[:, 0], -1))) / 2.0)


def load_workspace() -> dict:
    path = REPO / "config" / "workspace.yaml"
    if not path.is_file():
        return {}
    # Deliberately not importing yaml: the analyzer has to run under bare
    # interpreters that have numpy and nothing else. Only reads the scalars it
    # needs — top-level section, one level of indent — and ignores anything
    # deeper (the touchoff: block) rather than flattening it into the parent.
    out, section = {}, None
    for line in path.read_text().splitlines():
        raw = line.rstrip()
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if indent == 0:
            section = raw.split(":")[0].strip()
            out[section] = {}
        elif section and indent == 2:
            key, _, val = raw.strip().partition(":")
            val = val.split("#")[0].strip()
            if val in ("null", ""):
                out[section][key] = None
            else:
                try:
                    out[section][key] = float(val)
                except ValueError:
                    out[section][key] = val.strip('"')
    return out


def sparc(speed, fs, fc=10.0, amp_th=0.05):
    """Spectral arc length of the speed profile. Less negative = smoother.

    The standard movement-smoothness measure (Balasubramanian et al.), and the
    one that separated these policies where per-sample direction change could
    not: it is amplitude- and duration-normalised, so a slow cautious policy and
    a fast confident one are judged on shape rather than on speed.
    """
    speed = np.asarray(speed, dtype=float)
    if len(speed) < 16 or np.allclose(speed, 0):
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + 4))
    freqs = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    if mag.max() == 0:
        return float("nan")
    mag = mag / mag.max()
    keep = (freqs <= fc) & (mag >= amp_th)
    if keep.sum() < 4:
        return float("nan")
    f_sel, m_sel = freqs[keep], mag[keep]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    return float(-np.sum(np.sqrt(df**2 + np.diff(m_sel) ** 2)))


def log_dimensionless_jerk(p3, dt):
    """Log dimensionless jerk over the 3D path. Less negative = smoother."""
    p3 = np.asarray(p3, dtype=float)
    if len(p3) < 8:
        return float("nan")
    vel = np.gradient(p3, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    peak = float(np.linalg.norm(vel, axis=1).max())
    if peak <= 0:
        return float("nan")
    dur = len(p3) * dt
    dj = (dur**3 / peak**2) * float(np.trapezoid(np.sum(jerk**2, axis=1), dx=dt))
    return float(-np.log(abs(dj))) if dj > 0 else float("nan")


def pad_dwell(z, pad_play=CONTACT_TOL_MM):
    """How much of the run is spent resting at the run's own lowest level.

    Deliberately self-referential, and honest without a calibration: it does not
    claim the pen touched paper, it measures whether the policy SETTLES at its
    floor or merely dips through it. That distinction is what separated the
    policies on 2026-08-21 — one dwelt at its floor 90-100% of the time while
    the other hovered ~45 mm up and darted down for 30-74%. An absolute
    contact threshold could not see it, because each run had a different floor.

    Returns the hover level (p75), the floor (p2), the drop between them, and
    the dwell fraction within one pad-thickness of the floor.
    """
    z = np.asarray(z, dtype=float)
    floor = float(np.percentile(z, 2))
    hover = float(np.percentile(z, 75))
    return {
        "hover_mm": round(hover, 2),
        "floor_mm": round(floor, 2),
        "drop_mm": round(hover - floor, 2),
        "dwell_pct": round(100.0 * float((z <= floor + pad_play).mean()), 1),
    }


def lift_stats(z, t, clearance_mm=6.0, min_s=0.3):
    """Lift-and-hop travel: sustained excursions above the run's own floor.

    The 2026-08-24 eval's defining behavioral difference was invisible to the
    in-band percentage: policies that drew (dwell 90%+, zero lifts) versus
    policies that hopped between patches (10+ lifts, half the window airborne)
    both scored 100% "contact" under the 5 mm band. Counts excursions more
    than clearance_mm above the run's own p2 floor sustained longer than
    min_s, and the total airborne time."""
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    floor = float(np.percentile(z, 2))
    lifted = z > floor + clearance_mm
    events, total, start = 0, 0.0, None
    for i in range(len(t)):
        if lifted[i] and start is None:
            start = t[i]
        elif not lifted[i] and start is not None:
            if t[i] - start > min_s:
                events += 1
                total += t[i] - start
            start = None
    if start is not None and t[-1] - start > min_s:
        events += 1
        total += t[-1] - start
    return {"lift_events": events, "lift_s": round(total, 1)}


def descent_stats(z_full, t_full, floor_mm, band_mm=3.0, min_start_mm=20.0):
    """Approach quality for runs that start high (staged-pose batteries).

    Reports null when the run starts within min_start_mm of the floor — a
    descent metric on an on-paper start is noise. Overshoot is depth below
    the run's own floor within 1.5 s of first band entry: the 2026-08-24
    paired approach study separated on exactly this (soft touch vs plunge)."""
    z = np.asarray(z_full, dtype=float)
    t = np.asarray(t_full, dtype=float)
    band = floor_mm + band_mm
    if z[0] < band + min_start_mm:
        return {"descent": None}
    below = z < band
    if not below.any():
        return {"descent": {"start_mm": round(float(z[0]), 1), "reached_band": False}}
    i = int(np.argmax(below))
    seg = z[: i + 1]
    vz = np.diff(seg) / np.maximum(np.diff(t[: i + 1]), 1e-6)
    after = z[(t >= t[i]) & (t <= t[i] + 1.5)]
    return {"descent": {
        "start_mm": round(float(z[0]), 1),
        "reached_band": True,
        "time_to_band_s": round(float(t[i]), 1),
        "max_down_mm_s": round(float(-vz.min()), 0) if len(vz) else None,
        "monotone_frac": round(float((np.diff(seg) <= 0.5).mean()), 2) if len(seg) > 1 else None,
        "overshoot_mm": round(float(floor_mm - after.min()), 1),
    }}


def analyze(target: Path, paper_z=None, settle=SETTLE_S, window_s=None) -> dict:
    run_dir = target if target.is_dir() else target.parent
    if target.is_dir():
        candidates = sorted(target.glob("flight-*.csv")) or sorted(target.glob("*.csv"))
        if not candidates:
            raise SystemExit(f"no flight CSV in {target}")
        csv_path = candidates[-1]
    else:
        csv_path = target

    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        with __import__("contextlib").suppress(Exception):
            meta = json.loads(meta_path.read_text())

    rows = load_rows(csv_path)
    if len(rows) < 30:
        raise SystemExit(f"{csv_path}: only {len(rows)} rows — nothing to analyze")

    t = np.array([float(r["t_mono"]) for r in rows])
    t -= t[0]
    dt = np.diff(t)
    chain = UrdfChain(str(REPO / "urdf" / "tatbot.urdf"))
    names = chain.arm_joint_names("right")

    win = t >= settle
    if window_s:
        win &= t < settle + window_s
    if win.sum() < 20:
        raise SystemExit(f"{csv_path}: run too short for a {settle}s settle window")
    rows_w = [r for r, keep in zip(rows, win, strict=True) if keep]
    tw = t[win]

    # --- contact basis ----------------------------------------------------
    # The tip offset is applied to the FK path itself whenever it is known —
    # zs are then PEN TIP heights, and the touch-off's paper plane is directly
    # comparable to them. Without an offset, zs are ee_gripper_link heights,
    # which is exactly what the ee_contact_z composite fallback measures.
    ws = load_workspace().get("right", {})
    tip_offset = None
    # tip_frame gates it: a gripper-era offset is in a frame the tool no
    # longer has any relation to, so it reads as no touch-off at all.
    if ws.get("pen_tip_offset_z") is not None and ws.get("tip_frame") == TIP_LINK:
        tip_offset = [float(ws.get("pen_tip_offset_x") or 0.0),
                      float(ws.get("pen_tip_offset_y") or 0.0),
                      float(ws["pen_tip_offset_z"])]
    tip_full = pen_path(rows, chain, names, tip_offset)
    tip = tip_full[win]
    xs, ys, zs = tip[:, 0], tip[:, 1], tip[:, 2]
    if paper_z is not None:
        plane, basis, valid, src = float(paper_z), "external", True, "--paper-z"
    elif ws.get("paper_plane_z") is not None and tip_offset is not None:
        plane = float(ws["paper_plane_z"]) * 1000.0
        basis, valid, src = "touchoff_tip", True, "config/workspace.yaml"
    elif ws.get("ee_contact_z") is not None and tip_offset is None:
        # Composite EE height at contact — only coherent while zs are also EE
        # heights, i.e. while no tip offset is being applied above.
        plane = float(ws["ee_contact_z"]) * 1000.0
        basis, valid, src = "touchoff_ee", True, "config/workspace.yaml"
    else:
        plane = float(np.percentile(zs, INFERRED_PLANE_PCT))
        basis, valid, src = "inferred", False, f"run p{INFERRED_PLANE_PCT:02.0f}"

    contact = zs <= plane + CONTACT_TOL_MM
    runs, cur = [], 0
    for c in contact:
        cur = cur + 1 if c else 0
        runs.append(cur)
    rate_hz = len(rows) / t[-1]
    longest_contact_s = (max(runs) / rate_hz) if runs else 0.0
    # What the ink charge model wants from this run (scripts/lib/ink_spec.py):
    # millimetres of path while in contact, and seconds in contact — the raw
    # quantities a stroke ledger event carries. Only meaningful against a
    # touch-off plane; an inferred plane is the run's own floor.
    xy_step = np.linalg.norm(np.diff(np.stack([xs, ys], 1), axis=0), axis=1)
    ink_contact = {
        "contact_mm": round(float(xy_step[contact[1:] & contact[:-1]].sum()), 2),
        "contact_s": round(float(contact.sum() / rate_hz), 2),
        "basis": basis, "valid": valid,
    }

    xy = np.stack([xs, ys], 1)
    rs = resample_arclength(xy)
    turn_mm, rev_mm = turn_stats(rs)
    turn_s, rev_s = turn_stats(xy)
    path_xy = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    bw, bh = float(xs.max() - xs.min()), float(ys.max() - ys.min())

    stalls = dt[dt > STALL_S]
    lost = float(((stalls - 1.0 / 30).sum()) * 30) if len(stalls) else 0.0

    def col(name):
        return np.array([abs(float(r[name])) for r in rows_w])

    # 2026-08-30: the column became the contact force the follower publishes
    # (carriage external effort, N); older CSVs carry the grip law's command.
    grip_cmd = col("contact_force_n" if "contact_force_n" in rows_w[0] else "grip_effort")
    grip_meas = col("eff_left_carriage_joint")
    arm_eff = np.max([col("eff_" + j) for j in ARM_JOINTS], axis=0)

    # Clamp ticks recomputed from the CSV, so throttling the WARNING display
    # costs no data (raw_* is the pre-clamp goal, goal_* the sent one).
    mrt = float(meta.get("key", {}).get("max_relative_target", 0.5) or 0.5)
    clamped = np.zeros(len(rows), dtype=bool)
    for j in ARM_JOINTS:
        raw = np.array([float(r["raw_" + j]) for r in rows])
        pos = np.array([float(r["pos_" + j]) for r in rows])
        clamped |= np.abs(raw - pos) >= (mrt - 1e-4)

    settle_detected = float(t[np.argmax(~clamped[10:]) + 10]) if clamped[:10].any() else 0.0

    m = {
        "ticks": len(rows), "duration_s": round(float(t[-1]), 2),
        "loop_hz_mean": round(rate_hz, 2),
        "dt_p50_ms": round(float(np.percentile(dt, 50)) * 1000, 1),
        "dt_p95_ms": round(float(np.percentile(dt, 95)) * 1000, 1),
        "dt_p99_ms": round(float(np.percentile(dt, 99)) * 1000, 1),
        "stall_count": int(len(stalls)),
        "stall_ms_median": round(float(np.median(stalls)) * 1000, 1) if len(stalls) else 0.0,
        "stall_ticks_lost_pct": round(100 * lost / (t[-1] * 30), 1),
        "contact_pct": round(100 * float(contact.mean()), 1),
        "contact_longest_s": round(longest_contact_s, 2),
        "z_min_mm": round(float(zs.min()), 2), "z_mean_mm": round(float(zs.mean()), 2),
        "z_std_mm": round(float(zs.std()), 2), "z_p05_mm": round(float(np.percentile(zs, 5)), 2),
        "bbox_x_mm": round(bw, 1), "bbox_y_mm": round(bh, 1),
        "aspect": round(max(bw, bh) / max(min(bw, bh), 1e-9), 2),
        "hull_area_mm2": round(hull_area(xy), 1),
        "path_len_xy_mm": round(path_xy, 1),
        "mean_speed_mm_s": round(path_xy / max(tw[-1] - tw[0], 1e-9), 1),
        "turn_deg_per_mm": round(turn_mm, 1), "reversal_pct": round(rev_mm, 1),
        "sparc": round(sparc(np.linalg.norm(np.gradient(xy, np.median(np.diff(tw)), axis=0), axis=1),
                             1.0 / float(np.median(np.diff(tw)))), 3),
        "ldj": round(log_dimensionless_jerk(tip, float(np.median(np.diff(tw)))), 2),
        **pad_dwell(zs),
        **lift_stats(zs, tw),
        **descent_stats(tip_full[:, 2], t, float(np.percentile(zs, 2))),
        "turn_deg_per_sample": round(turn_s, 1), "reversal_pct_per_sample": round(rev_s, 1),
        "grip_cmd_peak_n": round(float(grip_cmd.max()), 2),
        "grip_meas_peak_n": round(float(grip_meas.max()), 2),
        "arm_eff_peak_nm": round(float(arm_eff.max()), 2),
        "arm_eff_p95_nm": round(float(np.percentile(arm_eff, 95)), 2),
        "clamp_ticks": int(clamped.sum()),
        "clamp_ticks_post_settle": int(clamped[win].sum()),
    }

    out = {
        "schema": "tatbot.rollout.analysis/1",
        "run_id": meta.get("run_id") or run_dir.name,
        "flight_csv": str(csv_path),
        "git_sha": meta.get("git", {}).get("short"),
        "window": {"settle_s": settle, "end_s": round(float(tw[-1]), 1),
                   "run_duration_s": round(float(t[-1]), 1),
                   "scored_s": round(float(tw[-1] - tw[0]), 1),
                   "settle_detected_s": round(settle_detected, 1),
                   "ticks": int(win.sum())},
        "ink_contact": ink_contact,
        "geometry": {"contact_basis": basis, "valid": valid,
                     "plane_z_mm": round(plane, 2), "plane_source": src,
                     "pen_tip_offset_mm": ([round(v * 1000.0, 2) for v in tip_offset]
                                           if tip_offset is not None else None),
                     "link": TIP_LINK, "contact_tol_mm": CONTACT_TOL_MM,
                     "caveat": None if valid else
                     "no touch-off in right/tool_mount; absolute height is not measured"},
        "config": _run_config(meta),
        "metrics": m,
    }
    out["checks"] = build_checks(m, out["config"], out["geometry"])
    return out


def _run_config(meta: dict) -> dict:
    """Expected values come from the RUN, not today's repo config — re-analyzing
    an old run must check it against what actually ran."""
    key = meta.get("key", {}) or {}
    cfg = {"source": "meta.json" if key else "defaults", "fps": 30}
    for name, default in (("chunk", None), ("thresh", None), ("policy_type", None),
                          ("n_action_steps", None), ("chunk_source", None),
                          ("refill_budget_ms", None), ("infer_ms", None),
                          ("grip_force", 33.0), ("policy", None)):
        val = key.get(name, default)
        with __import__("contextlib").suppress(Exception):
            if isinstance(val, str) and val.replace(".", "", 1).isdigit():
                val = float(val) if "." in val else int(val)
        cfg[name] = val
    return cfg


def _check(name, value, expected, status, note=""):
    return {"name": name, "value": value, "expected": expected,
            "status": status, "note": note}


def build_checks(m, cfg, geom) -> list[dict]:
    """Checks whose notes carry the open questions, so they get re-asked with
    fresh numbers every run instead of being rediscovered from a journal."""
    checks = []
    fps = cfg.get("fps", 30)

    # Control-loop health. Resolved 2026-08-21: the stalls are ACTION QUEUE
    # STARVATION, not client-side observation upload as previously believed.
    # Client work measured 1.3 ms median with 1 tick over 50 ms in 978, while
    # the queue sat empty 23% of the time at chunk 24 and 3% at chunk 48 — and
    # the 48-chunk run held a clean 30.0 Hz with zero stalls.
    budget = cfg.get("refill_budget_ms")
    infer = cfg.get("infer_ms")
    hz = m["loop_hz_mean"]
    note = (f"{m['stall_ticks_lost_pct']}% of ticks lost to {m['stall_count']} stalls "
            f">50 ms (median {m['stall_ms_median']} ms).")
    if budget and infer:
        note += (f" Refill budget {budget} ms vs ~{infer} ms inference"
                 f"{' — TOO TIGHT, the queue drains' if budget < infer * 1.5 else ''}.")
    checks.append(_check("loop_rate", hz, f">={fps - 2}",
                         "ok" if hz >= fps - 2 else "warn", note))

    # SAFETY. The commanded value is the bounded-grip law's own output; if it
    # ever exceeds the ceiling, the saturation in send_action is broken.
    gf = cfg.get("grip_force") or 33.0
    checks.append(_check(
        "grip_cmd_ceiling", m["grip_cmd_peak_n"], f"<={gf}",
        "fail" if m["grip_cmd_peak_n"] > gf + 0.01 else "ok",
        "commanded, saturated by the bounded-grip law. A fail here means the "
        "saturation is broken — stop and fix it before running again."))

    # NOT a bug, and the note exists so nobody re-files it as one.
    checks.append(_check(
        "grip_measured", m["grip_meas_peak_n"], f"<={1.5 * gf:.0f} (informational)",
        "warn" if m["grip_meas_peak_n"] > 1.5 * gf else "ok",
        "MEASURED carriage effort — since 2026-08-30 this is the pen's contact "
        "force along its own axis (the mount rides the carriage), plus preload "
        "and drivetrain friction. Gripper-era runs read 45-51 N of grip "
        "against a 33 N ceiling and that was expected then."))

    if m["clamp_ticks_post_settle"]:
        checks.append(_check(
            "clamp_post_settle", m["clamp_ticks_post_settle"], "0", "warn",
            "max_relative_target still clamping after the settle window — the "
            "arm is not tracking. Expected only during the startup traverse."))

    checks.append(_check(
        "contact_basis", geom["contact_basis"], "touchoff_tip",
        "ok" if geom["valid"] else "warn",
        "" if geom["valid"] else
        "heights are RELATIVE ONLY — no pen touch-off in config/workspace.yaml, "
        "so the plane is this run's own contact floor. Ranks runs within a "
        "session; do not compare across setups."))
    return checks


STATUS_MARK = {"ok": "  ok ", "warn": "WARN ", "fail": "FAIL "}


def print_report(a: dict) -> None:
    m, g = a["metrics"], a["geometry"]
    print(f"\n{a['run_id']}   {a['flight_csv']}")
    print(f"window t={a['window']['settle_s']:.0f}..{a['window']['end_s']:.0f}s "
          f"({a['window']['ticks']} ticks)   basis={g['contact_basis']}"
          f"{'' if g['valid'] else ' (RELATIVE ONLY)'}   plane={g['plane_z_mm']} mm")
    # "in-band", not "contact": FK proximity to a calibrated plane is NOT a
    # touch measurement — the plane drifts with the base/table/pad, and on
    # 2026-08-24 this line said "contact 100%" while the pen never marked.
    print(f"\n  geometry   in-band {m['contact_pct']:>5.1f}% (FK, NOT touch)"
          f"  longest {m['contact_longest_s']:.1f}s"
          f"   Zmin {m['z_min_mm']:.1f}  Zstd {m['z_std_mm']:.2f}")
    print(f"             bbox {m['bbox_x_mm']:.1f} x {m['bbox_y_mm']:.1f} mm  "
          f"aspect {m['aspect']:.2f}  hull {m['hull_area_mm2']:.0f} mm2")
    print(f"  motion     {m['mean_speed_mm_s']:.1f} mm/s  path {m['path_len_xy_mm']:.0f} mm"
          f"   turn {m['turn_deg_per_mm']:.1f} deg/step  reversals {m['reversal_pct']:.1f}%")
    print(f"  smoothness SPARC {m['sparc']:.2f}  LDJ {m['ldj']:.1f}"
          f"   (less negative = smoother)")
    print(f"  dwell      {m['dwell_pct']:.0f}% at its floor {m['floor_mm']:.1f} mm"
          f"   hover {m['hover_mm']:.1f}  drop {m['drop_mm']:.1f} mm")
    print(f"  lifts      {m['lift_events']} sustained (>6 mm, >0.3 s)"
          f"   airborne {m['lift_s']:.1f}s of {a['window']['scored_s']:.0f}s")
    d = m.get("descent")
    if d is not None:
        if d.get("reached_band"):
            print(f"  descent    {d['start_mm']:.0f} mm -> band in {d['time_to_band_s']:.1f}s"
                  f"   monotone {d['monotone_frac']:.2f}  first-touch overshoot"
                  f" {d['overshoot_mm']:+.1f} mm")
        else:
            print(f"  descent    started {d['start_mm']:.0f} mm up and NEVER reached the band")
    print(f"  timing     {m['loop_hz_mean']:.1f} Hz  dt p95 {m['dt_p95_ms']:.0f} ms"
          f"   {m['stall_count']} stalls  {m['stall_ticks_lost_pct']:.0f}% ticks lost")
    print(f"  force      grip cmd {m['grip_cmd_peak_n']:.1f} N (ceiling "
          f"{a['config'].get('grip_force')})  measured {m['grip_meas_peak_n']:.1f} N"
          f"   arm p95 {m['arm_eff_p95_nm']:.1f} Nm")
    print()
    for c in a["checks"]:
        print(f"  {STATUS_MARK.get(c['status'], '  ?  ')}{c['name']:<18} "
              f"{str(c['value']):<10} expected {c['expected']}")
        if c["note"] and c["status"] != "ok":
            for line in _wrap(c["note"], 68):
                print(f"        {line}")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def compare(paths: list[Path]) -> None:
    rows, seen = [], set()
    for p in paths:
        with __import__("contextlib").suppress(Exception):
            row = json.loads(Path(p).read_text())
            # A glob over the run root also matches the `latest` symlink, which
            # would silently double-count the newest run in the table.
            key = row.get("run_id") or str(Path(p).resolve())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    if not rows:
        raise SystemExit("no readable analysis.json files")
    inferred = [r for r in rows if not r["geometry"]["valid"]]
    # An "inferred" plane is each run's OWN contact floor, so contact% answers
    # "how long did this policy sit near its own lowest point" — which is not
    # the question, and ranks a policy that never reached the paper above one
    # that did. It is therefore not printed at all without a touch-off. Z min
    # IS comparable (same arm, same frame, same session) and takes its place.
    show_contact = not inferred
    head = f"{'run':<26}" + (f"{'contact':>9}" if show_contact else f"{'Zmin(rel)':>11}")
    print("\n" + head + f"{'SPARC':>8}{'dwell':>8}{'rev%':>7}{'Zstd':>7}"
          f"{'bbox':>16}{'aspect':>8}{'mm/s':>7}{'Hz':>7}")
    print("-" * (len(head) + 68))
    key = ((lambda x: -x["metrics"]["contact_pct"]) if show_contact
           else (lambda x: x["metrics"]["z_min_mm"]))
    for r in sorted(rows, key=key):
        m = r["metrics"]
        name = (r["config"].get("policy") or r["run_id"] or "")
        name = str(name).rstrip("/").split("/")[-1][:25]
        first = (f"{m['contact_pct']:>8.1f}%" if show_contact
                 else f"{m['z_min_mm']:>11.1f}")
        print(f"{name:<26}{first}{m.get('sparc', float('nan')):>8.2f}"
              f"{m.get('dwell_pct', float('nan')):>7.0f}%{m['reversal_pct']:>7.1f}"
              f"{m['z_std_mm']:>7.2f}"
              f"{m['bbox_x_mm']:>7.1f} x{m['bbox_y_mm']:>6.1f}{m['aspect']:>8.2f}"
              f"{m['mean_speed_mm_s']:>7.1f}{m['loop_hz_mean']:>7.1f}")
    if inferred:
        print(f"\n!! contact% withheld: {len(inferred)}/{len(rows)} runs have no pen "
              "touch-off, so each\n!! run's plane is its own contact floor and the number "
              "would rank a policy that\n!! never reached the paper above one that did. "
              "Z min is relative but comparable.\n!! Run scripts/il_touchoff.py to make "
              "contact measurable.")
    # With n>=3 the SPREAD is the result, not the mean: on 2026-08-21 three runs
    # of one checkpoint spanned 11x in drawn area, so a single number per policy
    # is not a measurement.
    if len(rows) >= 3:
        print()
        for key, label in (("sparc", "SPARC"), ("dwell_pct", "dwell%"),
                           ("reversal_pct", "rev%")):
            vals = [r["metrics"].get(key) for r in rows]
            vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
            if len(vals) >= 3:
                print(f"  {label:<8} mean {np.mean(vals):8.2f}   "
                      f"spread {max(vals) - min(vals):7.2f}   n={len(vals)}")

    hz = {round(r["metrics"]["loop_hz_mean"]) for r in rows}
    if len(hz) > 1:
        print("!! these runs did NOT share a control rate "
              f"({sorted(hz)} Hz) — smoothness is confounded by tick loss.")
    # Footprint and path length grow with observation time, so runs of
    # different length cannot be compared on them at all.
    wins = sorted({round(r["window"].get("scored_s",
                                        r["window"]["end_s"] - r["window"]["settle_s"]))
                   for r in rows})
    if len(wins) > 1:
        print(f"!! these runs did NOT share a window ({wins} s) — bbox, hull and "
              "path length\n!! grow with observation time. Re-run at one duration, "
              "or pass --settle/--window.")


# --- ink: debit the session ----------------------------------------------

def _node_from_run_id(run_id: str | None) -> str | None:
    """`20260821T164233Z-<node>-a3f1` -> `<node>` (tatbot_runlog run ids)."""
    if not run_id:
        return None
    parts = run_id.split("-")
    return parts[1] if len(parts) >= 3 and parts[0].endswith("Z") else None


def _short_host(name: str | None) -> str | None:
    return name.split(".")[0].lower() if name else None


def run_ink_tracking(run_dir: Path) -> tuple[bool, str]:
    """Did this run take part in ink accounting? The run's own stamp
    ($RUN_DIR/ink.json, written by scripts/lib/dip_hook.sh) outranks the
    environment: TATBOT_INK lives one shell, the stamp lives with the run."""
    import os

    stamp = run_dir / "ink.json"
    if stamp.is_file():
        try:
            tracking = bool(json.loads(stamp.read_text()).get("tracking", True))
            return tracking, "run stamp ink.json"
        except Exception:
            pass
    return os.environ.get("TATBOT_INK", "1") != "0", "TATBOT_INK (no run stamp)"


def run_ink_session_id(run_dir: Path) -> str | None:
    """The session the run's mirrored ink events (ink.jsonl) belong to, if any."""
    mirror = run_dir / "ink.jsonl"
    if not mirror.is_file():
        return None
    for line in mirror.read_text().splitlines():
        try:
            sid = json.loads(line).get("session_id")
        except Exception:
            continue
        if sid:
            return sid
    return None


def debit_ink_session(run_dir: Path, a: dict, log=print) -> dict | None:
    """After the fact, the run's contact becomes a `stroke` in the open ink
    session (scripts/lib/ink_session.py): the robot has no stroke executor,
    so this is where the charge on the needle gets spent. Skipped, and says
    why, when the run was launched --no-ink (its ink.json stamp, else
    TATBOT_INK=0), when no session is open, when the run was made on another
    node or under another session than the one open here (the session is
    node-local; debiting the wrong needle is worse than not debiting), when
    the contact plane is inferred (the µL would be fiction), or when the
    session already holds this run. Never a gate."""
    import socket

    tracking, basis = run_ink_tracking(run_dir)
    if not tracking:
        return {"skipped": f"--no-ink ({basis})"}
    try:
        sys.path.insert(0, str(REPO / "scripts" / "lib"))
        import ink_session
        import ink_spec
        import tool_spec
    except Exception as exc:  # the analysis must not fail on the ink stack
        return {"skipped": f"ink stack unavailable: {exc}"}
    sess = ink_session.current()
    if sess is None:
        return {"skipped": "no open ink session"}
    meta = {}
    with contextlib.suppress(Exception):
        meta = json.loads((run_dir / "meta.json").read_text())
    run_node = _short_host((meta.get("node") or {}).get("hostname")) \
        or _short_host(_node_from_run_id(a.get("run_id") or run_dir.name))
    here = _short_host(socket.gethostname())
    if run_node and run_node != here:
        return {"skipped": f"run was made on {run_node}; the open session here is {here}'s — analyze it there"}
    run_sid = run_ink_session_id(run_dir)
    if run_sid and run_sid != sess.session_id:
        return {"skipped": f"run belongs to session {run_sid}, not the open session {sess.session_id}"}
    ic = a.get("ink_contact") or {}
    if not ic.get("valid"):
        return {"skipped": f"contact basis {ic.get('basis')!r} is not a touch-off; not debiting"}
    run_id = a.get("run_id") or run_dir.name
    try:
        tool = tool_spec.load_tool(sess.tool_id, REPO)
        policy = ink_spec.policy_for(tool)
        inks = ink_spec.load_inks(REPO)
        policy = ink_spec.policy_with_ink(policy, inks.get(sess.ink_id) if sess.ink_id else None)
        ev = ink_session.apply_stroke(sess, policy, ic["contact_mm"], ic["contact_s"], run_id=run_id,
                                      basis=ic.get("basis"), mirror=run_dir / "ink.jsonl")
    except Exception as exc:
        return {"skipped": f"debit failed: {exc}"}
    if ev is None:
        return {"skipped": "run already debited in this session"}
    out = {"session_id": sess.session_id, "ul": ev["ul"], "taken_ul": ev["taken_ul"],
           "charge_after_ul": sess.charge_ul, "capacity_ul": sess.capacity_ul}
    log(f"  ink        session {sess.session_id}: -{ev['taken_ul']:.3f} uL for "
        f"{ic['contact_mm']:.0f} mm / {ic['contact_s']:.1f} s on the surface; "
        f"charge {sess.charge_ul:.2f}/{sess.capacity_ul:.2f} uL")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="*", help="run directory or flight CSV")
    ap.add_argument("--paper-z", type=float, default=None,
                    help="paper plane in mm (overrides the workspace/inferred basis)")
    ap.add_argument("--settle", type=float, default=SETTLE_S)
    ap.add_argument("--window", type=float, default=None,
                    help="seconds of run to score after the settle window; use this "
                         "to align runs of different duration")
    ap.add_argument("--compare", action="store_true",
                    help="treat targets as analysis.json files and tabulate them")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if not args.target:
        ap.error("need a run directory, flight CSV, or --compare analysis.json...")
    if args.compare:
        compare([Path(p) for p in args.target])
        return 0
    rc = 0
    for target in args.target:
        a = analyze(Path(target).expanduser(), paper_z=args.paper_z,
                    settle=args.settle, window_s=args.window)
        run_dir = Path(target) if Path(target).is_dir() else Path(target).parent
        a["ink"] = debit_ink_session(run_dir, a, log=(lambda *_: None) if args.json else print)
        out = run_dir / "analysis.json"
        with __import__("contextlib").suppress(Exception):
            out.write_text(json.dumps(a, indent=2))
        if args.json:
            print(json.dumps(a, indent=2))
        else:
            print_report(a)
            if a["ink"] and a["ink"].get("skipped"):
                print(f"  ink        not debited: {a['ink']['skipped']}")
            print(f"\n  wrote {out}")
        if any(c["status"] == "fail" for c in a["checks"]):
            rc = 0  # a report, never a gate — see il_rollout_async.sh
    return rc


if __name__ == "__main__":
    sys.exit(main())
