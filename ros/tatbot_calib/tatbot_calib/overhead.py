"""The overhead D555 registered through the wrist D405 instead of the wrist's fiducial cube (ros/README.md section 7).

The cube's registration (register.py) places the D555 by the cube's tags through FK and their layout on the carriage.
The cube gives under load and its stickers stand millimetres off their layout: that fit carried a ~2 deg tilt, which
read the palette station 8.5 mm low against the wrist camera and the measured tool alike (2026-10-22). This places
the camera by what it and the wrist D405 both see, through FK and the URDF's chain to the D405 (Trossen's rigid
mount, the same in every view):

- the palette's roof tag, its pose in the arm base from the wrist's holds over it (wristcam.HandEye's views, the tag
  left free), and the fiducial tile on the holder, its markers' corners in the base by bundle adjustment of the
  wrist's straight-down views (every corner free, each view where FK puts it); each against its corners in one D555
  frame taken with the arm at rest;
- the wrist's depth of the station and the holder against the D555's own, point to plane on its flat patches.

A tile marker the fit misses past OUTLIER_M is a bad read on one side and is left out once (2026-10-22: 6 of 17, each
2.2-4.6 mm off, the rest 0.2-1.5 mm). Each region (station, holder) is left out and predicted by the other: one alone
extrapolates past ten millimetres to the other, so the run needs both. A tile the D555 cannot read (the 12 mm forearm
tile) has its views aimed at the bridge's page and rests on the wrist's depth there (ros/README.md section 7). The D555's optics are held as it reports them
(it sees the tile ~1 % larger than the wrist puts it; its depth agrees there within 0.1 mm). The run writes the holds,
candidate.json (a tatbot.arm-registration/1 record) and report.md; --adopt installs it where register.adopt does, the
robot-world golden's camera with it, and the station touch must be made again (its registration changed).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tatbot_calib import program, reach, register, station, sweep, wristcam

STATION_HOLDS = 12
# Straight-down views over the tile (inspect.top_down): the camera this far over the tile's centre moved this much
# along the tile's x and y (2026-10-22, the silicone tile on the holder: 10 of 12 such views reached).
TILE_VIEWS = [(d, xy) for d in (0.19, 0.23, 0.27) for xy in ((-0.035, 0.0), (-0.03, 0.04), (-0.03, -0.04))] + \
    [(0.23, (0.0, -0.05)), (0.27, (0.0, -0.05)), (0.27, (0.0, 0.0))]
FRAMES = 5
SETTLE_S = 1.0
NEAR_M, FAR_M = 0.17, 0.5      # the wrist's depth kept: past the tool, within the scene
CLOUD_POINTS = 4000            # sampled from each hold's depth
MATCH_M = 0.010                # a wrist point pairs with the D555's nearest point within this
FLAT_M = 0.0015                # ... on a patch flat to this (its 12 neighbours' least spread)
DEPTH_SIGMA_M = 0.002
DEPTH_WEIGHT = 400             # all the depth together weighs as this many points, so it cannot drown the tags
TAG_SIGMA_PX = 0.5             # the roof tag and the tile's 40 mm anchors at ~0.6 m; its 18 mm markers 0.8
OUTLIER_M = 0.002
BAD_READ_PX = 12.0             # a wrist read the bundle misses this far is a wrong read or one the needle cut
# The straight-down views' baselines leave a marker's height to 1-2 mm by parallax alone, and the D555 sees the tile
# obliquely, so each read's corners carry the D405's depth there (within +-0.5 mm of triangulation, 2026-10-22).
CORNER_DEPTH_SIGMA_M = 0.001
REGIONS = ("station", "holder")
HOLDER_DEPTH_VIEWS = 3         # a holder no D555 read places needs the wrist's depth from this many views


def pose(x) -> np.ndarray:
    """A 4x4 pose from a rotation vector and a translation (6)."""
    from scipy.spatial.transform import Rotation

    out = np.eye(4)
    out[:3, :3], out[:3, 3] = Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6]
    return out


def project(points, k, dist) -> np.ndarray:
    import cv2

    return cv2.projectPoints(np.asarray(points, float).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, dist)[0][:, 0]


@dataclass
class Target:
    """A fiducial both cameras read: its corners in the arm base (the wrist's) and in the D555 image."""
    region: str
    name: str
    base: np.ndarray            # (n, 3) m
    pixels: np.ndarray          # (n, 2) D555 px
    sigma_px: float = TAG_SIGMA_PX


@dataclass
class Scene:
    """The D555's depth as points in its colour optical frame, their normals and which lie on flat patches."""
    points: np.ndarray
    normals: np.ndarray
    flat: np.ndarray

    def __post_init__(self):
        from scipy.spatial import cKDTree

        self.tree = cKDTree(self.points)

    @classmethod
    def from_depth(cls, depth_m, k, dist, step: int = 2) -> Scene:
        """From an aligned depth image (m), every `step`-th pixel 0.3-1 m out."""
        import cv2
        from scipy.spatial import cKDTree

        v, u = np.nonzero((depth_m > 0.3) & (depth_m < 1.0))
        keep = (v % step == 0) & (u % step == 0)
        v, u = v[keep], u[keep]
        z = depth_m[v, u].astype(float)
        rays = cv2.undistortPoints(np.c_[u, v].astype(np.float64).reshape(-1, 1, 2), k, dist)[:, 0]
        points = np.c_[rays * z[:, None], z]
        _, near = cKDTree(points).query(points, k=12)
        spread = points[near] - points[near].mean(axis=1, keepdims=True)
        _, s, vt = np.linalg.svd(spread, full_matrices=False)
        return cls(points, vt[:, -1, :], s[:, -1] < FLAT_M)

    def pairs(self, world_from_base, cloud) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The cloud's points (base) with their nearest flat D555 points and normals (camera)."""
        seen = cloud @ world_from_base[:3, :3].T + world_from_base[:3, 3]
        dist, idx = self.tree.query(seen, distance_upper_bound=MATCH_M)
        ok = np.isfinite(dist)
        ok[ok] &= self.flat[idx[ok]]
        return cloud[ok], self.points[idx[ok]], self.normals[idx[ok]]


def tag_in_base(views, model, prior) -> tuple[np.ndarray, np.ndarray]:
    """base <- tag from the wrist's views of it, each (base <- camera by FK, k, dist, its corners' px): the tag's
    pose that projects its model where every view saw it. Returns (base <- tag, each view's rms px)."""
    from scipy.optimize import least_squares

    corners = np.c_[model, np.ones(len(model))].T

    def residuals(x):
        b_t = pose(x) @ prior
        return np.concatenate([project((np.linalg.inv(b_c) @ b_t @ corners).T[:, :3], k, dist) - px
                               for b_c, k, dist, px in views]).ravel()

    sol = least_squares(residuals, np.zeros(6), loss="soft_l1", f_scale=1.0)
    return pose(sol.x) @ prior, np.sqrt(np.mean(sol.fun.reshape(len(views), -1) ** 2, axis=1))


def tile_in_base(views, start) -> tuple[dict, float, int]:
    """The tile's markers' corners in the base from the wrist's views of them, each (base <- camera by FK, k, dist,
    {id: (4, 2) px}, {id: (4,) the D405's depth at those corners, m}), every corner free from `start` ({id: (4, 3)}):
    ({id: (4, 3)}, rms px, reads left out). A read missed past BAD_READ_PX is left out once and the adjustment
    repeated."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    ids = sorted(start)
    at = {i: n for n, i in enumerate(ids)}
    obs = [(n, i) for n, view in enumerate(views) for i in view[3] if i in at]

    def adjust(x0, obs):
        def residuals(x):   # per read: its 4 corners' px (8), then their depths over CORNER_DEPTH_SIGMA_M in px units
            pts = x.reshape(-1, 4, 3)
            out = []
            for n, i in obs:
                b_c, k, dist, reads, depths = views[n]
                cam = (np.linalg.inv(b_c)[:3, :3] @ pts[at[i]].T).T + np.linalg.inv(b_c)[:3, 3]
                out += [(project(cam, k, dist) - reads[i]).ravel(),
                        np.nan_to_num(cam[:, 2] - depths.get(i, np.full(4, np.nan))) / CORNER_DEPTH_SIGMA_M]
            return np.concatenate(out)

        sparsity = lil_matrix((12 * len(obs), len(x0)), dtype=int)
        for r, (_, i) in enumerate(obs):
            sparsity[12 * r:12 * r + 12, 12 * at[i]:12 * at[i] + 12] = 1
        return least_squares(residuals, x0, jac_sparsity=sparsity, loss="soft_l1", f_scale=1.0, x_scale="jac")

    def px_rms(sol):
        return np.sqrt(np.mean(sol.fun.reshape(-1, 12)[:, :8] ** 2, axis=1))

    sol = adjust(np.concatenate([np.asarray(start[i], float) for i in ids]).ravel(), obs)
    bad = px_rms(sol) > BAD_READ_PX
    if bad.any():
        sol = adjust(sol.x, [o for o, b in zip(obs, bad, strict=True) if not b])
    pts = sol.x.reshape(-1, 4, 3)
    return {i: pts[at[i]] for i in ids}, float(np.sqrt(np.mean(px_rms(sol) ** 2))), int(bad.sum())


def _residuals(x, start, use, pairs, depth_scale, k, dist) -> np.ndarray:
    """fit's misses at world <- base = start . pose(x): the targets' px over their sigma, the depth's point-to-plane
    metres over depth_scale."""
    w = start @ pose(x)
    out = [((project(t.base @ w[:3, :3].T + w[:3, 3], k, dist) - t.pixels) / t.sigma_px).ravel() for t in use]
    out += [np.einsum("ij,ij->i", p @ w[:3, :3].T + w[:3, 3] - q, normal) / depth_scale for p, q, normal in pairs]
    return np.concatenate(out)


def fit(start, targets, clouds, scene: Scene, k, dist, *, regions=REGIONS, tags=True, depth=True) -> np.ndarray:
    """world <- base from `start`: the targets' corners where the D555 saw them, and the clouds ([(region, (n, 3)
    base)]) on its flat patches (point to plane, re-paired as the fit moves)."""
    from scipy.optimize import least_squares

    use = [t for t in targets if t.region in regions] if tags else []
    x = np.zeros(6)
    for _ in range(8):
        pairs = [scene.pairs(start @ pose(x), c) for r, c in clouds if r in regions] if depth else []
        scale = DEPTH_SIGMA_M * math.sqrt(max(1, sum(len(p) for p, _, _ in pairs)) / DEPTH_WEIGHT)
        sol = least_squares(_residuals, x, args=(start, use, pairs, scale, k, dist), loss="soft_l1", f_scale=2.0,
                            x_scale="jac")
        moved, x = float(np.linalg.norm(sol.x - x)), sol.x
        if moved < 1e-6:
            break
    return start @ pose(x)


def misses(world_from_base, targets, clouds, scene: Scene, k, dist) -> dict:
    """Per region: its targets' rms in the D555 (px, and mm at their range) and its depth's median miss (m)."""
    out = {}
    for region in REGIONS:
        use = [t for t in targets if t.region == region]
        row = {}
        if use:
            base = np.concatenate([t.base for t in use])
            seen = base @ world_from_base[:3, :3].T + world_from_base[:3, 3]
            e = np.linalg.norm(project(seen, k, dist) - np.concatenate([t.pixels for t in use]), axis=1)
            row.update(tags_px=float(np.sqrt(np.mean(e ** 2))),
                       tags_m=float(np.sqrt(np.mean(e ** 2)) * np.median(seen[:, 2]) / k[0, 0]))
        pairs = [scene.pairs(world_from_base, c) for r, c in clouds if r == region]
        if pairs and sum(len(p) for p, _, _ in pairs):
            p, q, nrm = (np.concatenate(a) for a in zip(*pairs, strict=True))
            d = np.einsum("ij,ij->i", p @ world_from_base[:3, :3].T + world_from_base[:3, 3] - q, nrm)
            row.update(depth_median_m=float(np.median(d)), depth_points=len(d))
        out[region] = row
    return out


def target_miss_m(world_from_base, target: Target, k, dist) -> float:
    seen = target.base @ world_from_base[:3, :3].T + world_from_base[:3, 3]
    e = np.linalg.norm(project(seen, k, dist) - target.pixels, axis=1)
    return float(np.sqrt(np.mean(e ** 2)) * np.median(seen[:, 2]) / k[0, 0])


def solve(installed, targets, clouds, scene: Scene, k, dist) -> tuple[np.ndarray, dict]:
    """The registration and its report: the fit of both regions after the tile's outliers, the installed one's and
    each variant's misses (each region left out, tags alone, depth alone)."""
    fitted = fit(installed, targets, clouds, scene, k, dist)
    first = {t.name: target_miss_m(fitted, t, k, dist) for t in targets if t.region == "holder"}
    page = [miss for name, miss in first.items() if name.startswith("page")]
    # stencild's page may stand millimetres off whole; a wrist read off its layout stands off alone (2026-10-23: one
    # forearm marker 17.5 mm off the print's surface, the rest within 2 mm of their layout)
    bound = {"tile": OUTLIER_M, "page": max(OUTLIER_M, 2.0 * float(np.median(page))) if page else 0.0}
    dropped = [name for name, miss in first.items() if miss > bound[name.split()[0]]]
    targets = [t for t in targets if t.name not in dropped]
    fitted = fit(installed, targets, clouds, scene, k, dist)
    variants = {"installed": installed, "fitted": fitted}
    for region in REGIONS:
        variants[f"without the {region}"] = fit(installed, targets, clouds, scene, k, dist,
                                                regions=tuple(r for r in REGIONS if r != region))
    variants["tags alone"] = fit(installed, targets, clouds, scene, k, dist, depth=False)
    variants["depth alone"] = fit(installed, targets, clouds, scene, k, dist, tags=False)
    report = {"dropped": dropped, "first_misses_m": first, "targets": len(targets),
              "holder_by": ", ".join(sorted({t.name.split()[0] for t in targets if t.region == "holder"})) or "depth alone",
              "variants": {name: {"turn_deg": turn_deg(installed, w), **misses(w, targets, clouds, scene, k, dist)}
                           for name, w in variants.items()}}
    return fitted, report


def turn_deg(a, b) -> float:
    return math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0))))


def record(installed: dict, world_from_base: np.ndarray, report: dict, run_id: str) -> dict:
    """The installed registration with this world <- base, its camera bundle and frames kept."""
    root_from_base = np.asarray(installed["root_from_arm_base"], float)
    fitted = report["variants"]["fitted"]
    return {**installed, "world_from_arm_base": world_from_base.tolist(),
           "world_from_root": (world_from_base @ np.linalg.inv(root_from_base)).tolist(),
           "fit": {"observations": report["targets"], "distinct_poses": None,
                   "median_px": fitted.get("station", {}).get("tags_px"), "max_px": None, "p95_px": None,
                   "per_region": fitted},
           "withheld": {"observations": 0, "median_px": None, "max_px": None},
           "hold_out": {name: report["variants"][f"without the {name}"][name] for name in REGIONS},
           "per_tag": {}, "rig": {"capture": run_id, "method": "wrist",
                                  "note": "the D555 placed through the wrist D405: the palette's roof tag and the "
                                          "holder's fiducial tile in one D555 frame, the wrist's depth against its "
                                          "own; the wrist cube not used; the camera's optics as it reports them"}}


def adopt(run_dir: Path, arm: str) -> list[str]:
    """The candidate where register.adopt installs a registration (the robot-world golden's world <- base with it,
    its tag layout kept), each one it replaces kept under run_dir/replaced."""
    candidate = run_dir / "candidate.json"
    done = []
    for directory in (register.VISION, register.STACK_CALIB):
        destination = directory / f"arm-registration-{arm}-current.json"
        register.install(candidate, destination, run_dir / "replaced" / directory.name)
        done.append(str(destination))
    world = register.VISION / "robot-world-current.json"
    if arm == "right" and world.is_file():
        golden = json.loads(world.read_text())
        golden["world_from_base"] = json.loads(candidate.read_text())["world_from_root"]
        (run_dir / "robot-world.json").write_text(json.dumps(golden, indent=2) + "\n")
        register.install(run_dir / "robot-world.json", world, run_dir / "replaced")
        done.append(str(world))
    return done


def _report(report: dict, ball_gap: dict | None) -> str:
    lines = ["# The overhead D555 through the wrist D405\n", f"- the holder placed by its {report['holder_by']}",
             f"- tile markers left out past {OUTLIER_M * 1000:.0f} mm: {', '.join(report['dropped']) or 'none'}", "",
             "| registration | turn from installed | station tag | station depth | holder tile | holder depth |",
             "|---|---|---|---|---|---|"]
    for name, row in report["variants"].items():
        cells = []
        for region in REGIONS:
            r = row.get(region, {})
            cells.append(f"{r['tags_m'] * 1000:.1f} mm ({r['tags_px']:.2f} px)" if "tags_m" in r else "-")
            cells.append(f"{r['depth_median_m'] * 1000:+.1f} mm" if "depth_median_m" in r else "-")
        lines.append(f"| {name} | {row['turn_deg']:.2f} deg | " + " | ".join(cells) + " |")
    if ball_gap:
        lines += ["", "The station touch's ball less the overhead fix's, through each registration (mm): " +
                  ", ".join(f"{k} {np.round(np.asarray(v) * 1000, 1).tolist()}" for k, v in ball_gap.items())]
    return "\n".join(lines) + "\n"


# --- the run ----------------------------------------------------------------------------------------------------
class Capture:
    """One run's views: the static D555 frame, the wrist's holds over the roof tag and its views of the tile."""

    def __init__(self, cal: program.Calibration, camera, detector, model, tile):
        from tatbot_session import inspect as ins

        self.cal, self.camera, self.detector, self.model, self.tile = cal, camera, detector, model, tile
        self.kin, self.frame = cal.kin, ins.camera_frame(cal.arm)
        self.depth_frame = ins.DEPTH_FRAME.format(arm=cal.arm)
        self.dir = cal.run.dir / "holds"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.holds: list[dict] = []

    def hold(self, label: str, region: str, tcp: np.ndarray) -> dict | None:
        """One guarded move (Calibration.go_to) and the wrist's frames there (median colour and raw depth), or None
        when refused."""
        try:
            self.cal.go_to(label, tcp)
        except RuntimeError as error:
            if program._latched(self.cal):
                raise
            self.cal.say(f"{label}: skipped ({error})")
            return None
        self.cal.rig.spin(SETTLE_S)
        before = self.cal.rig.joints()
        colors, _ = self.camera.grab_both(FRAMES, fresh=True)
        after = self.cal.rig.joints()
        if float(np.max(np.abs(after - before)[:6])) > 2e-3:
            self.cal.say(f"{label}: the arm moved during the grab")
            return None
        row = {"label": label, "region": region, "q": (before + after) / 2.0,
               "color": np.median(np.asarray(colors), axis=0).astype(np.uint8),
               "depth_raw": np.median(np.asarray(self.camera.raw_depth), axis=0).astype(np.uint16)}
        np.savez_compressed(self.dir / f"{label}.npz", **row)
        self.holds.append(row)
        return row

    def station(self, base_from_tag: np.ndarray) -> None:
        hand = wristcam.HandEye(self.cal, base_from_tag, self.camera, self.detector, self.model, self.frame)
        pool = hand.candidates()
        self.cal.say(f"station: {len(pool)} views of the roof tag clear of the station; {STATION_HOLDS} wanted")
        def wanted() -> int:
            return STATION_HOLDS - sum(h["region"] == "station" for h in self.holds)

        for i, tcp in reach.nearest_first(self.kin, self.cal.rig.joints, pool[:3 * STATION_HOLDS], wanted):
            self.hold(f"station{i:02d}", "station", tcp)

    def from_rest(self, wake=None) -> None:
        """Land and wake the arm between the regions, so the holder's views plan from rest: from where the station's
        holds ended, every one met joint 3's guarded limit (the station agent's setup, 2026-10-22). `wake` wakes the
        landed arm (the session client's, by default)."""
        import subprocess

        program.retreat(self.cal.rig, self.kin)
        (wake or (lambda arm: subprocess.run(["ros2", "run", "tatbot_session", "client", "wake", "--arm", arm],
                                             check=True, timeout=90, capture_output=True)))(self.cal.arm)
        for _ in range(60):
            self.cal.rig.spin(0.5)
            if self.cal.rig.safety is not None and not self.cal.rig.safety.landed:
                return
        raise RuntimeError(f"the {self.cal.arm} arm did not wake after landing between the station and the holder")

    def holder(self, base_from_tile: np.ndarray) -> None:
        from tatbot_session import inspect as ins
        from tatbot_session.ready import solve_ik

        for i, (distance, xy) in enumerate(TILE_VIEWS):
            aimed = ins.top_down(self.kin, solve_ik, base_from_tile, xy, self.cal.rig.joints(), frame=self.frame,
                                 clearance_m=0.015, max_y=float(base_from_tile[1, 3]), distances_m=(distance,))
            if aimed is None:
                self.cal.say(f"holder{i:02d}: no reachable view {distance:.2f} m over {xy}")
                continue
            self.hold(f"holder{i:02d}", "holder", self.kin.fk(aimed[0]))


def tile_frame(tile, reads: dict, depth_m, k, dist, installed: np.ndarray) -> np.ndarray | None:
    """base <- page, roughly, from the D555's reads of the tile through the installed registration: each read
    marker's corners along their rays at the D555's depth there, the tile's layout fitted to them (in
    stencil.target_corners' frame, the page's with y and z turned over, turned back). For aiming the views only; None
    under three markers."""
    import cv2
    from fiducials import stencil

    layout, seen = [], []
    for i, corners in reads.items():
        depths = corner_depths(depth_m, np.asarray(corners, float), half=2)
        if np.isfinite(depths).all():
            rays = cv2.undistortPoints(np.asarray(corners, float).reshape(-1, 1, 2), k, dist)[:, 0]
            world = np.c_[rays, np.ones(4)] * depths[:, None]
            seen.append((world - installed[:3, 3]) @ installed[:3, :3])
            layout.append(stencil.target_corners(tile, i))
    if len(seen) < 3:
        return None
    return stencil.kabsch(np.concatenate(layout), np.concatenate(seen)) @ np.diag([1.0, -1.0, -1.0, 1.0])


def corner_depths(depth_m, corners, half: int = 3) -> np.ndarray:
    """The depth (m) about each corner's pixel, NaN where under half the patch reads (the D405's native depth grid
    stands within a few pixels of its colour's; the D555's depth is aligned to its colour)."""
    out = np.full(len(corners), np.nan)
    for n, (u, v) in enumerate(np.round(corners).astype(int)):
        patch = depth_m[max(v - half, 0):v + half + 1, max(u - half, 0):u + half + 1]
        if (patch > 0).sum() >= (2 * half + 1) ** 2 // 2:
            out[n] = float(np.median(patch[patch > 0]))
    return out


def wrist_cloud(depth_raw, meta: dict, base_from_depth: np.ndarray, seed: int) -> np.ndarray:
    """A hold's depth (NEAR_M-FAR_M) as CLOUD_POINTS points in the base."""
    di = meta["intrinsics"]
    d = np.asarray(depth_raw, np.float32) * float(meta["depth_units_m"])
    v, u = np.nonzero((d > NEAR_M) & (d < FAR_M))
    pick = np.random.default_rng(seed).choice(len(v), min(len(v), CLOUD_POINTS), replace=False)
    v, u = v[pick], u[pick]
    z = d[v, u]
    points = np.stack([(u - di["ppx"]) / di["fx"] * z, (v - di["ppy"]) / di["fy"] * z, z], axis=1)
    return points @ base_from_depth[:3, :3].T + base_from_depth[:3, 3]


def roof_tag_px(detector, image) -> np.ndarray | None:
    """The palette's roof tag's (ID 0) corners in an image, or None."""
    seen = [d for d in detector.detect("tag", image, 0) if int(d.tag_id) == 0]
    return np.asarray(seen[0].corners_px, float).reshape(4, 2) if seen else None


def page_targets(tile, corners: dict, world_from_page, sigma_m: float, k, dist) -> list[Target]:
    """The wrist's tile corners where stencild's page (the D555 tracking the print whole) puts them: one rigid
    observation, so each corner's sigma grows with the markers' count."""
    from fiducials import stencil

    w_t = np.asarray(world_from_page, float) @ np.diag([1.0, -1.0, -1.0, 1.0])   # the target frame: y and z turned
    sigma_px = max(1.0, sigma_m * k[0, 0] / float(w_t[2, 3])) * math.sqrt(len(corners))
    return [Target("holder", f"page {i}", corners[i],
                   project((w_t[:3, :3] @ stencil.target_corners(tile, i).T).T + w_t[:3, 3], k, dist), sigma_px)
            for i in sorted(corners)]


def tile_targets(capture: Capture, static_image, meta, page=None) -> tuple[list[Target], dict]:
    """The tile's markers both cameras read: their corners in the base from the wrist's holder views (tile_in_base,
    each read's D405 depth with it), against the D555's reads of them, and with `page` (world <- page, its sigma,
    k, dist) against stencild's page."""
    import cv2
    from fiducials import stencil

    views, start = [], {}
    for h in (h for h in capture.holds if h["region"] == "holder"):
        b_c = capture.kin.frame(h["q"], capture.frame)
        reads = {int(i): np.asarray(c, float) for i, c in stencil.find(h["color"], capture.tile).items()}
        depth = np.asarray(h["depth_raw"], float) * float(meta["depth_units_m"])
        depths = {i: d for i, c in reads.items() if np.isfinite(d := corner_depths(depth, c)).all()}
        views.append((b_c, capture.camera.k, capture.camera.dist, reads, depths))
        for i in (i for i in depths if i not in start):   # a start along each corner's ray at the depth there
            rays = cv2.undistortPoints(reads[i].reshape(-1, 1, 2), capture.camera.k, capture.camera.dist)[:, 0]
            start[i] = (np.c_[rays, np.ones(4)] * depths[i][:, None]) @ b_c[:3, :3].T + b_c[:3, 3]
    if not start:
        return [], {"tile_views": len(views)}
    corners, rms, out = tile_in_base(views, start)
    d555 = {}
    for det in (stencil.detector(capture.tile), stencil.detector(capture.tile, upsample=True)):
        for i, c in stencil.find(static_image, capture.tile, det).items():
            d555.setdefault(int(i), np.asarray(c, float))
    targets = [Target("holder", f"tile {i}", corners[i], d555[i], TAG_SIGMA_PX if capture.tile.sides_m[i] > 0.03 else 0.8)
               for i in sorted(set(d555) & set(corners))]
    shared = len(targets)
    if page is not None:
        targets += page_targets(capture.tile, corners, *page)
    return targets, {"tile_views": len(views), "tile_markers": len(corners), "tile_px": round(rms, 2),
                     "tile_reads_out": out, "tile_shared": shared}


def targets_from(capture: Capture, prior_tag, static_image, meta, page=None) -> tuple[list[Target], list, dict]:
    """The run's targets and clouds from its holds and static frame, and what they said along the way."""
    roof = [(capture.kin.frame(h["q"], capture.frame), capture.camera.k, capture.camera.dist, px)
            for h in capture.holds if h["region"] == "station"
            and (px := roof_tag_px(capture.detector, h["color"])) is not None]
    base_from_tag, per_view = tag_in_base(roof, capture.model, prior_tag)
    seen = roof_tag_px(capture.detector, static_image)
    targets = [] if seen is None else [
        Target("station", "roof tag", (base_from_tag @ np.c_[capture.model, np.ones(4)].T).T[:, :3], seen)]
    tile, said = tile_targets(capture, static_image, meta, page)
    clouds = [(h["region"], wrist_cloud(h["depth_raw"], meta, capture.kin.frame(h["q"], capture.depth_frame), n))
              for n, h in enumerate(capture.holds)]
    return targets + tile, clouds, {"roof_tag_views": len(roof), "roof_tag_px": per_view.round(2).tolist(), **said}


def page_in_base(page, world_from_base: np.ndarray) -> np.ndarray:
    """base <- page from the bridge's Page: stencild's (world) through `world_from_base`, or the fixed one (base)."""
    from tatbot_description.transforms import quat_matrix

    p = page.pose.pose.pose   # Page.pose is a PoseWithCovarianceStamped
    m = quat_matrix([p.position.x, p.position.y, p.position.z],
                    [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
    return m if page.pose.header.frame_id.endswith("base_link") else np.linalg.inv(world_from_base) @ m


def unmeasured(targets, clouds) -> str | None:
    """Why a fit cannot place the camera, or None: the station needs its roof tag; the holder a tile or its depth."""
    regions = {t.region for t in targets}
    if "station" not in regions:
        return "the D555 or the wrist reads no roof tag at the station"
    if "holder" not in regions and sum(region == "holder" and len(c) > 0 for region, c in clouds) < HOLDER_DEPTH_VIEWS:
        return f"the holder has no tile marker both cameras read and under {HOLDER_DEPTH_VIEWS} views of its depth"
    return None


def placed_station(cal, fix: station.StationFix, tool_id: str, urdf: Path):
    """(the station touch or None, the fix placed by it): the views are placed by the touch, whose height against the
    overhead's is what this registration mends; by the overhead fix when no touch speaks for now."""
    try:
        touch = station.load_touch(cal.arm)
        palette = station.touched_palette(touch, fix, tool_id=tool_id, tcp_m=cal.tip0, palette_urdf=urdf,
                                          joint_offsets=station.joint_offsets(cal.repo, cal.arm), check_height=False)
    except station.StaleStationError as error:
        cal.say(f"S0: the views are placed by the overhead fix: {error}")
        return None, fix
    ball = (palette @ np.append(station.ball_in_palette(urdf), 1.0))[:3]
    return touch, dataclasses.replace(fix, base_from_palette=palette, ball=ball)


def write(cal, installed: dict, targets, clouds, shot: dict, optics, said: dict, touch, fix, adopting: bool) -> int:
    """The fit, candidate.json, report.json and report.md, and with `adopting` its installation: 0, or 1 when a
    region is unmeasured."""
    if why := unmeasured(targets, clouds):
        cal.say(f"{why}: nothing fitted")
        return 1
    k, dist = optics
    start = np.asarray(installed["world_from_arm_base"], float)
    fitted, report = solve(start, targets, clouds, Scene.from_depth(np.asarray(shot["depth_m"], float), k, dist),
                           k, dist)
    gap = None
    if touch is not None:   # the touched ball less the overhead fix's, through each registration
        in_world = start[:3, :3] @ np.asarray(fix.ball, float) + start[:3, 3]
        gap = {name: (np.asarray(touch.ball, float) - (np.linalg.inv(w)[:3, :3] @ in_world + np.linalg.inv(w)[:3, 3]))
               .tolist() for name, w in (("installed", start), ("fitted", fitted))}
    run_dir = cal.run.dir
    (run_dir / "candidate.json").write_text(json.dumps(record(installed, fitted, report, run_dir.name), indent=2) + "\n")
    (run_dir / "report.json").write_text(json.dumps({**report, "captured": said, "ball_gap_m": gap}, indent=1) + "\n")
    text = _report(report, gap)
    (run_dir / "report.md").write_text(text)
    print(text, flush=True)
    if adopting:
        for path in adopt(run_dir, cal.arm):
            cal.say(f"adopted: {path}")
        cal.say("the station touch's registration changed: touch it again (tatbot ros calib run --arm "
                f"{cal.arm} --station-only, then tatbot ros calib height)")
    return 0


def run(args) -> int:
    from tatbot_bridge import capture as overhead
    from tatbot_description import repo_root, robot_description
    from tatbot_motion import Kinematics, load_motion
    from tatbot_session import inspect as ins

    from tatbot_calib.tool import ToolRefusedError, fitted_tool

    repo = repo_root(None)
    try:
        tool, halo_tip = fitted_tool(repo, args.arm, args.tool)
    except ToolRefusedError as error:
        print(json.dumps({"ok": False, "message": str(error)}))
        return 3
    sys.path.insert(0, str(repo / "scripts" / "lib"))
    sys.path.insert(0, str(repo / "scripts" / "vision"))
    import tatbot_runlog
    from fiducials import load_inventory, stencil, tag_model_corners
    from fiducials.detector import DetectorConfig, FiducialDetector

    fix = station.StationFix.from_dict(json.loads(Path(args.station).expanduser().read_text()))
    station.require_fresh(fix, args.run_started, max_age_s=args.max_age_s)
    tile = ins.print_artwork(args.pattern)
    if tile is None:
        print(json.dumps({"ok": False, "message": f"no fiducial tile {args.pattern!r} is installed"}))
        return 3
    kin = Kinematics(robot_description(None, arms=(args.arm,)), args.arm)
    run_log = tatbot_runlog.init(program.WORKFLOW, meta={"arm": args.arm, "method": "overhead", "station": args.station,
                                                         "pattern": args.pattern},
                                 attach_logging=False, argv=["tatbot_calib", "overhead", "--arm", args.arm])
    cal = program.Calibration(None, run_log, repo, args.arm, halo_tip, 0.0, kin)
    sweep.equip(cal, fix, load_motion())
    urdf = repo / "urdf" / "palette.urdf"
    touch, placed = placed_station(cal, fix, tool.tool_id, urdf)
    program.place_station(cal, placed)
    registration = register.STACK_CALIB / f"arm-registration-{args.arm}-current.json"
    installed = json.loads(registration.read_text())
    world_from_base0 = np.asarray(installed["world_from_arm_base"], float)
    inventory = load_inventory(repo / "config" / "fiducials.json")
    roof = inventory.target("palette")
    detector = FiducialDetector.from_inventory(inventory, DetectorConfig(min_side_px=8.0, refinement="edges",
                                                                        cell_margin=0.30), target="palette")
    camera = ins.WristCamera(depth=True)
    camera.size = (1280, 720)   # wristcam.HandEye's margins
    cal.rig, code = program.Rig(args.arm), 1
    try:
        program._preflight(cal)
        d555 = overhead.Camera(repo)
        try:   # the arm stands at rest, clear of the station and the holder
            shot = d555.capture(time.time_ns())
        finally:
            d555.close()
        k, dist = overhead.k_dist(overhead.intrinsics_of(shot["metadata"]))
        np.savez_compressed(run_log.dir / "static.npz", overhead_rgb_bgr=shot["image"], overhead_depth_m=shot["depth_m"])
        reads = {int(i): c for i, c in stencil.find(shot["image"], tile).items()}
        base_from_tile = tile_frame(tile, reads, shot["depth_m"], k, dist, world_from_base0)
        prior_tag = placed.base_from_palette @ station.palette_from_tag(urdf, require_confirmed=True)
        capture = Capture(cal, camera, detector, tag_model_corners(roof.edge_m), tile)
        capture.station(prior_tag)
        seen = None   # stencild's page, the D555's sight of the print whole: the holder's lateral place
        if base_from_tile is None and (page := cal.rig.page()) is not None:
            base_from_tile = page_in_base(page, world_from_base0)
            if not page.pose.header.frame_id.endswith("base_link"):
                seen = (world_from_base0 @ base_from_tile, page.translation_sigma_m or 0.002, k, dist)
            cal.say(f"holder: the D555 reads {len(reads)} of the tile's markers: the views aimed at the bridge's page")
        if base_from_tile is None:
            cal.say(f"holder: the D555 reads {len(reads)} of the tile's markers, 3 needed, and no page: no views")
        else:
            capture.from_rest()
            capture.holder(base_from_tile)
        targets, clouds, said = targets_from(capture, prior_tag, shot["image"], camera.depth_metadata, seen)
        cal.say("captured: " + json.dumps(said))
        code = write(cal, installed, targets, clouds, shot, (k, dist), said, touch, fix, args.adopt)
    except RuntimeError as error:
        cal.say(f"stopped: {error}")
    finally:
        camera.close()
        if not program._latched(cal):
            program.retreat(cal.rig, kin)
        cal.rig.close()
        run_log.finalize(code, status="ok" if code == 0 else "fail")
    print(json.dumps({"ok": code == 0, "run_dir": str(run_log.dir)}), flush=True)
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tatbot_calib overhead", description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tool", default=None)
    parser.add_argument("--station", required=True, help="this run's station.json (tatbot ros station)")
    parser.add_argument("--run-started", type=float, required=True)
    parser.add_argument("--max-age-s", type=float, default=station.MAX_AGE_S)
    parser.add_argument("--pattern", required=True, help="the fiducial tile on the holder (an installed stencil id)")
    parser.add_argument("--adopt", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
