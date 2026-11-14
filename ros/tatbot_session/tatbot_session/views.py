"""The lab's views (docs/lab.md): frames of page points taken before a draw and again after it, which `tatbot lab
read` pairs by name and differences.

A view holds the pen over a page point the way a close-up hold does (client._closeup_hold): a joint move to the close-up
approach height over the point, straight down to its hover height, frames, straight back up. Every wrist D405 the arm
carries grabs `frames` frames there and its per-pixel median is kept, as v<NN>-<role>.png. With no points the arm stays
where it stands and the cameras take that one view (a rest view, 2026-11-05's).

Before and after must stand at the same poses, so both aim by one page record: the after capture names the run whose
page the before capture used (views.json's `page_run`). Every hold is planned, its wrist checked against the model's
range, before the arm moves at all (2026-11-05: a hold over a bad page turned the wrist into its cables).

Without a page set-up the views can aim from the overhead instead (`sheets`): one tracker sample of a print gives
every sheet of it in view, each its own page through the arm's adopted registration (tatbot_bridge.page), so a hold
can stand over each mount carrying one print, or with `sheet_at` over the one nearest a point. The tracker reads a page
several millimetres off, so such holds hover higher than a set-up's.

The overhead D555 is a view of its own, overhead.png: the median of `frames` exposures from its owner's capture over
the bus (tatbot_bridge.capture), each newer than the last. It is taken before any hold, with the arm where it stands
(at rest, once a capture has woken it), so the arm hides the same part of the table before and after; it moves
nothing and sees every mount at about 0.86 px/mm, enough for a cell's mean change, never a line.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

WORKFLOWS = ("ros-draw", "ros-touch")   # the runs that leave a page set-up (page.json with `used`)
OVERHEAD = "overhead.png"
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}")


def name(index: int, role: str) -> str:
    return f"v{index:02d}-{role}.png"


def points_mm(text: str, flag: str = "--points") -> list[tuple[float, float]]:
    """Page points `x,y;x,y` in mm (the page frame: about the page centre, x right, y toward the print's top)."""
    out = []
    for item in filter(None, (part.strip() for part in text.split(";"))):
        x, sep, y = item.partition(",")
        if not sep:
            raise ValueError(f"{flag}: expected x,y in mm, got {item!r}")
        out.append((float(x), float(y)))
    if not np.all(np.isfinite(np.asarray(out, float))):
        raise ValueError(f"{flag}: every coordinate must be finite")
    return out


def page_run(log_root: Path, run: str = "") -> Path:
    """The run whose page set-up the views aim by: `run` (a ros-draw or ros-touch run id), else the newest of
    either with a finished set-up."""
    roots = [Path(log_root).expanduser() / workflow for workflow in WORKFLOWS]
    if run:
        found = [root / run for root in roots if (root / run / "page.json").is_file()]
        if not found:
            raise ValueError(f"no ros-draw or ros-touch run {run} with a page set-up under {log_root}")
        return found[0]
    runs = sorted((p for root in roots for p in root.glob("2*")
                   if (p / "page.json").is_file() and "used" in json.loads((p / "page.json").read_text())),
                  key=lambda p: p.name)
    if not runs:
        raise ValueError(f"no run under {log_root} has a page set-up: set one up first (tatbot ros touch)")
    return runs[-1]


def page_record(run: Path) -> tuple[np.ndarray, dict]:
    """base_from_page of a run's finished set-up, and its record."""
    page = json.loads((run / "page.json").read_text())
    if "used" not in page:
        raise ValueError(f"{run.name}: its page set-up never finished, so there is no page to aim by")
    return np.array(page["used"], dtype=float), page


def heading(page: dict, fk_now: np.ndarray) -> np.ndarray:
    """The tcp pose a hold keeps its wrist turned as: the set-up's first gauge hold or touch, else where it stands."""
    rows = page.get("gauge") or page.get("touches") or []
    return np.array(rows[0]["tcp"], float) if rows else np.asarray(fk_now, float)


def resolve_print(references: Path, given: str) -> str:
    """The installed print a pattern id or its leading hex digits name (the observer's references)."""
    hexpart = given.removeprefix("stencil-").lower()
    found = sorted(p.name for p in Path(references).glob("stencil-*")
                   if (p / "tracking.json").is_file() and p.name.removeprefix("stencil-").startswith(hexpart))
    if len(found) != 1:
        raise ValueError(f"{given!r} names {len(found)} of the prints installed in {references}; it must name one")
    return found[0]


def sheets(sample: dict, world_from_base) -> list[dict]:
    """Every sheet of a measured tracker sample (tatbot_bridge.page.copies, the most markers first), each a page in
    the arm's base and in the tracker's world, the overhead's colour optical frame (where `lab read` finds it in the
    overhead's frame): {"sheet", "base_from_page", "world_from_page", "sigma_m"}."""
    from tatbot_bridge import page

    return [{"sheet": k, "base_from_page": page.base_from_page(world_from_base, sheet["world_from_target"]),
             "world_from_page": page.world_from_page(sheet["world_from_target"]),
             "sigma_m": float(sheet["translation_sigma_m"])} for k, sheet in enumerate(page.copies(sample))]


def sheet_at(found: list[dict], xy_m, same_m: float) -> list[dict]:
    """The one of `found` (sheets) whose centre stands nearest xy_m in the arm's base, if within same_m (stack.yaml
    page.copies, as the bridge chooses); ValueError saying where each stands."""
    centres = [np.asarray(s["base_from_page"], float)[:2, 3] for s in found]
    off = [float(np.linalg.norm(at - np.asarray(xy_m, float))) for at in centres]
    k = int(np.argmin(off)) if off else -1
    if k < 0 or off[k] > same_m:
        where = ", ".join(f"{x * 1e3:.0f},{y * 1e3:.0f}" for x, y in centres) or "none"
        raise ValueError(f"--sheet-at: no sheet within {same_m * 1e3:.0f} mm of {xy_m[0] * 1e3:.0f},"
                         f"{xy_m[1] * 1e3:.0f} mm; the overhead reads {len(found)}, at {where} mm")
    return [found[k]]


def targets(pages: list[np.ndarray], points_m) -> list[tuple[np.ndarray, tuple[float, float]]]:
    """(page, point) for every point on every page, page by page."""
    return [(np.asarray(used, float), xy) for used in pages for xy in points_m]


def plan(kin, aims, turn: np.ndarray, q_seed, *, height_m: float) -> list[np.ndarray]:
    """Joint positions over every (page, point) aim at height_m along its page's normal, each from the last;
    ValueError naming the first whose wrist would leave the model's range, before anything moves."""
    from tatbot_session import geometry
    from tatbot_session import inspect as ins
    from tatbot_session.ready import solve_ik_seeded

    out, q = [], np.asarray(q_seed, float)
    for k, (used, xy) in enumerate(aims):
        q = solve_ik_seeded(kin, geometry.tool_down_pose(used, xy, height_m, turn), q)
        if not ins.wrist_in_range(kin, q):
            wrist = np.round(q[list(kin.arm_model.wrist)], 2).tolist()
            raise ValueError(f"view {k} over ({xy[0] * 1e3:+.1f} {xy[1] * 1e3:+.1f}) mm: the wrist would stand at "
                             f"{wrist} rad, outside {kin.arm_model.wrist_range_rad}: is the page right?")
        out.append(q)
    return out


def grab(cameras: dict, frames: int) -> dict[str, np.ndarray]:
    """role -> the per-pixel median of `frames` fresh colour frames from that camera."""
    return {role: np.median(np.stack(cam.grab_both(frames, fresh=True)[0]), axis=0).astype(np.uint8)
            for role, cam in cameras.items()}


def capture(io, kin, cameras: dict, pages: list[np.ndarray], over: list[np.ndarray], *, approach_m: float, motion,
            frames: int, settle_s: float, speed_rad_s: float, out_dir: Path, write) -> list[dict]:
    """Each planned view: the joint move over it, straight down approach_m along its page's normal (`pages`, one a
    view), frames, straight back up. `write(path, image)` saves a frame. Returns one record a view; RuntimeError on a
    move that does not end ok."""
    from tatbot_motion import plan_lift, to_knots

    from tatbot_session.ready import joint_move

    records = []
    for k, (q_over, used) in enumerate(zip(over, pages, strict=True)):
        normal = np.asarray(used, float)[:3, 2]
        moves = [("over", lambda q=q_over: joint_move(kin.joint_names, io.measured(), q, max_rad_s=speed_rad_s,
                                                      max_m_s=0.002, rate_hz=100.0, kin=kin)),
                 ("down", lambda n=normal: to_knots(plan_lift(q_seed=io.measured(), direction=-n,
                                                              distance_m=approach_m, kin=kin, motion=motion), 100.0))]
        for what, make in moves:
            status, _, text = io.execute(make())
            if status != "ok":
                raise RuntimeError(f"view {k} {what}: {status} {text}")
        try:
            time.sleep(settle_s)
            images = grab(cameras, frames)
            q_at = np.asarray(io.measured(), float)
            for role, image in images.items():
                write(out_dir / name(k, role), image)
            records.append({"view": k, "names": [name(k, role) for role in images], "q": q_at.tolist(),
                            "base_from_tcp": np.asarray(kin.fk(q_at), float).tolist()})
        finally:
            status, _, text = io.execute(to_knots(plan_lift(q_seed=io.measured(), direction=normal,
                                                            distance_m=approach_m, kin=kin, motion=motion), 100.0))
            if status != "ok":
                raise RuntimeError(f"view {k} up: {status} {text}")
    return records


def rest_view(io, kin, cameras: dict, *, frames: int, out_dir: Path, write) -> list[dict]:
    """The one view where the arm stands: no motion."""
    images = grab(cameras, frames)
    q_at = np.asarray(io.measured(), float)
    for role, image in images.items():
        write(out_dir / name(0, role), image)
    return [{"view": 0, "names": [name(0, role) for role in images], "q": q_at.tolist(),
             "base_from_tcp": np.asarray(kin.fk(q_at), float).tolist()}]


def overhead(camera, frames: int, *, out_dir: Path, write, now_ns) -> dict:
    """`frames` overhead exposures, each newer than the last (camera.capture(after_ns), tatbot_bridge.capture), their
    per-pixel median written as overhead.png; returns its record: the exposures' stamps and the owner's optics."""
    after, images, stamps, metadata = int(now_ns()), [], [], {}
    for _ in range(frames):
        shot = camera.capture(after)
        images.append(np.asarray(shot["image"]))
        stamps.append(int(shot["stamp_ns"]))
        metadata, after = shot["metadata"], int(shot["stamp_ns"]) + 1
    write(out_dir / OVERHEAD, np.median(np.stack(images), axis=0).astype(np.uint8))
    optics = (metadata.get("attributes") or {}).get("intrinsics")
    return {"name": OVERHEAD, "stamps_ns": stamps, "intrinsics": json.loads(optics) if optics else None}
