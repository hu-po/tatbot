"""Registration residuals: shared wrist-tag seat correction and joint_1..joint_4 offsets at FK(q + dq).

Base yaw and wrist roll are absorbed by camera registration and tag seat; the
carriage is not fitted. Model selection uses left-out runs, or left-out holds
for one run. Layout output is an adoption candidate; adopting it requires new
registrations. Joint offsets supply priors for tip and wrist-camera calibration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _pose(v) -> np.ndarray:
    """A 4x4 pose from a rotation vector and a translation (6)."""
    import cv2

    out = np.eye(4)
    out[:3, :3], out[:3, 3] = cv2.Rodrigues(np.asarray(v[:3], float))[0], v[3:6]
    return out


def _vector(pose: np.ndarray) -> np.ndarray:
    import cv2

    pose = np.asarray(pose, float)
    return np.concatenate([cv2.Rodrigues(pose[:3, :3])[0].ravel(), pose[:3, 3]])


FREE_JOINTS = (1, 2, 3, 4)   # controller order; joint_0 is a base yaw, joint_5 the seat's roll, index 6 the carriage
MODELS = {"rigid": ((), False), "seat": ((), True), "seat+joints": (FREE_JOINTS, True)}
ROBUST_PX = 2.0              # soft-L1 scale: a corner this far off counts less than squared
STILL_RAD = 2e-3             # a hold whose joints moved more than this during its capture is not used (register's)


@dataclass
class Run:
    """One registration's sightings ({"hold", "q" (7), "tag", "pixels" (4x2)}), the camera's optics and the
    camera <- base it adopted (the start of every fit)."""
    name: str
    k: np.ndarray
    dist: np.ndarray
    sightings: list
    camera_from_base: np.ndarray


def run_from_dir(path: Path) -> Run:
    """A ros-register run directory's holds, optics and camera <- base."""
    path = Path(path)
    rows = [json.loads(line) for line in (path / "holds.jsonl").read_text().splitlines() if line.strip()]
    optics = next((row["intrinsics"] for row in rows if row.get("intrinsics")), None)
    if optics is None:
        raise ValueError(f"{path.name}: no hold carries the camera's optics")
    k = np.array([[optics["fx"], 0.0, optics["ppx"]], [0.0, optics["fy"], optics["ppy"]], [0.0, 0.0, 1.0]])
    dist = np.array((list(optics.get("distortion_coefficients") or []) + [0.0] * 5)[:5], float)
    camera = np.asarray(json.loads((path / "report.json").read_text())["camera_from_base"], float)
    return Run(path.name, k, dist, sightings_of(rows), camera)


def sightings_of(rows) -> list:
    """A register run's hold rows (holds.jsonl) as sightings: every tag each reached, still hold saw."""
    return [{"hold": row["hold"], "q": np.asarray(row["q"], float), "tag": int(tag), "pixels": np.asarray(px, float)}
            for row in rows if row.get("reached", True) and row.get("q") is not None
            and float(row.get("still_rad", 0.0)) <= STILL_RAD
            for tag, px in (row.get("tags") or {}).items()]


@dataclass
class Chain:
    """The arm's kinematics, its wrist target (fiducials inventory) and the tags' layout on the parent frame."""
    kin: object
    arm: str
    target: object

    def __post_init__(self):
        from fiducials import tag_model_corners

        zero = np.zeros(7)
        parent = np.linalg.inv(self.kin.frame(zero, self.target.parent_frame))
        self.parent_from_tag = {t: parent @ self.kin.frame(zero, f"{self.arm}/wrist_tag{t}") for t in self.target.ids}
        self.model = np.c_[tag_model_corners(self.target.edge_m), np.ones(4)]

    def corners(self, s, dq, seat) -> np.ndarray:
        """A sighting's four corners in the base at its joints moved by dq, the tags on their seat."""
        tag = self.kin.frame(s["q"] + dq, self.target.parent_frame) @ seat @ self.parent_from_tag[s["tag"]]
        return (tag @ self.model.T).T[:, :3]

    def errors(self, run: Run, camera6, dq, seat) -> np.ndarray:
        """Each sighting's corners as projected less as detected (px, 8 per sighting)."""
        import cv2

        out = [(cv2.projectPoints(self.corners(s, dq, seat).reshape(-1, 1, 3), camera6[:3], camera6[3:6], run.k,
                                  run.dist)[0].reshape(4, 2) - s["pixels"]).ravel() for s in run.sightings]
        return np.concatenate(out) if out else np.zeros(0)


def _norms(errors) -> np.ndarray:
    return np.linalg.norm(errors.reshape(-1, 2), axis=1)


def _shared(x, n: int, free, seat: bool) -> tuple[np.ndarray, np.ndarray]:
    dq = np.zeros(7)
    dq[list(free)] = x[6 * n:6 * n + len(free)]
    return dq, (_pose(x[6 * n + len(free):]) if seat else np.eye(4))


def _solve(chain: Chain, runs, free, seat: bool, x0=None):
    """Each run's camera and the shared terms, soft-L1 on every corner."""
    from scipy.optimize import least_squares

    n = len(runs)

    def residuals(x):
        dq, pose = _shared(x, n, free, seat)
        return np.concatenate([chain.errors(r, x[6 * i:6 * i + 6], dq, pose) for i, r in enumerate(runs)])
    start = np.r_[np.concatenate([_vector(r.camera_from_base) for r in runs]), np.zeros(len(free) + 6 * seat)]
    return least_squares(residuals, start if x0 is None else x0, loss="soft_l1", f_scale=ROBUST_PX)


def _camera_alone(chain: Chain, run: Run, dq, seat):
    """A run's camera <- base alone, the shared terms held."""
    from scipy.optimize import least_squares

    return least_squares(lambda x: chain.errors(run, x, dq, seat), _vector(run.camera_from_base), loss="soft_l1",
                         f_scale=ROBUST_PX)


def fit(chain: Chain, runs: list[Run], free=FREE_JOINTS, seat: bool = True) -> dict:
    """The shared terms over the runs, each run its camera <- base: dq (7, rad), the seat (parent <- parent as
    measured), their 1-sigma from the Jacobian scaled by the median corner error, the cameras and the corner error."""
    n = len(runs)
    result = _solve(chain, runs, free, seat)
    norms = _norms(result.fun)
    cov = np.linalg.pinv(result.jac.T @ result.jac) * float(np.median(norms)) ** 2
    sd = np.sqrt(np.clip(np.diag(cov)[6 * n:], 0.0, None))
    dq, pose = _shared(result.x, n, free, seat)
    names = [chain.kin.joint_names[j].split("/")[-1] for j in free]
    return {"dq": dq, "seat": pose, "sigma": {**dict(zip(names, sd[:len(free)].tolist(), strict=True)),
                                              **({"seat": sd[len(free):].tolist()} if seat else {})},
            "cameras": {r.name: _pose(result.x[6 * i:6 * i + 6]) for i, r in enumerate(runs)},
            "median_px": float(np.median(norms)), "p95_px": float(np.percentile(norms, 95)),
            "corners": int(len(norms)), "x": result.x}


def hold_out(chain: Chain, runs: list[Run], free=FREE_JOINTS, seat: bool = True, full=None) -> dict:
    """The median corner error of what the fit was not given. Several runs: each left out in turn, the shared terms
    fitted to the others and its camera alone to them. One run: each hold left out, its corners at the camera and
    terms of the rest (warm-started from `full`, the fit of all)."""
    n = len(runs)
    if n > 1:
        each = {}
        for i, run in enumerate(runs):
            others = [r for j, r in enumerate(runs) if j != i]
            dq, pose = _shared(_solve(chain, others, free, seat).x, n - 1, free, seat)
            each[run.name] = float(np.median(_norms(_camera_alone(chain, run, dq, pose).fun)))
        return {"by": "run", "median_px": float(np.median(list(each.values()))), "each": each}
    run, x0 = runs[0], (full or fit(chain, runs, free, seat))["x"]
    held = []
    for hold in sorted({s["hold"] for s in run.sightings}):
        rest = Run(run.name, run.k, run.dist, [s for s in run.sightings if s["hold"] != hold], run.camera_from_base)
        left = Run(run.name, run.k, run.dist, [s for s in run.sightings if s["hold"] == hold], run.camera_from_base)
        x = _solve(chain, [rest], free, seat, x0).x
        dq, pose = _shared(x, 1, free, seat)
        held.append(float(np.median(_norms(chain.errors(left, x[:6], dq, pose)))))
    return {"by": "hold", "median_px": float(np.median(held)), "p90_px": float(np.percentile(held, 90)),
            "holds": len(held)}


def compare(chain: Chain, runs: list[Run], models=None) -> tuple[dict, dict]:
    """Each model fitted and held out; the best is the least held-out median. Returns (the report, JSON-ready;
    each model's fit)."""
    report, fits = {}, {}
    for name, (free, seat) in (models or MODELS).items():
        fits[name] = fit(chain, runs, free, seat)
        report[name] = {"fit": summary(chain, fits[name], free, seat),
                        "held_out": hold_out(chain, runs, free, seat, fits[name])}
    best = min(report, key=lambda name: report[name]["held_out"]["median_px"])
    return {"runs": [r.name for r in runs], "best": best, "models": report}, fits


def summary(chain: Chain, fitted: dict, free, seat: bool) -> dict:
    """A fit for the report: corner error, dq (mrad) and the seat (mm, deg) with their sigma."""
    names = [chain.kin.joint_names[j].split("/")[-1] for j in free]
    out = {"median_px": fitted["median_px"], "p95_px": fitted["p95_px"], "corners": fitted["corners"],
           "camera_from_base": {name: pose.tolist() for name, pose in fitted["cameras"].items()}}
    if free:
        out["dq_mrad"] = {name: round(float(fitted["dq"][j]) * 1e3, 2) for name, j in zip(names, free, strict=True)}
        out["dq_sigma_mrad"] = {name: round(fitted["sigma"][name] * 1e3, 2) for name in names}
    if seat:
        x6, sd = _vector(fitted["seat"]), fitted["sigma"]["seat"]
        out["seat"] = {"parent_frame": chain.target.parent_frame, "parent_from_measured": fitted["seat"].tolist(),
                       "mm": np.round(x6[3:] * 1e3, 2).tolist(), "deg": np.round(np.degrees(x6[:3]), 3).tolist(),
                       "sigma_mm": np.round(np.asarray(sd[3:]) * 1e3, 2).tolist(),
                       "sigma_deg": np.round(np.degrees(sd[:3]), 3).tolist()}
    return out


def layout_record(chain: Chain, runs: list[Run], fitted: dict) -> dict:
    """The seated layout as the wrist solve scripts/vision/export_wrist_tags.py adopts: each tag on the parent
    frame, the sightings behind it and their corner error, in px and as millimetres at the corner's depth."""
    import cv2

    px, mm, poses = [], [], {}
    for run in runs:
        camera6 = _vector(fitted["cameras"][run.name])
        rotation = cv2.Rodrigues(camera6[:3])[0]
        for s in run.sightings:
            points = chain.corners(s, fitted["dq"], fitted["seat"])
            err = _norms(chain.errors(Run(run.name, run.k, run.dist, [s], None), camera6, fitted["dq"],
                                      fitted["seat"]))
            depth = (rotation @ points.T).T[:, 2] + camera6[5]
            px += err.tolist()
            mm += (err * depth / float(np.mean(np.diag(run.k)[:2])) * 1e3).tolist()
            poses.setdefault(s["tag"], set()).add((run.name, s["hold"]))
    return {"link": chain.target.parent_frame, "mode": "corner_reprojection",
            "link_from_tag": {str(t): (fitted["seat"] @ m).tolist() for t, m in chain.parent_from_tag.items()},
            "observations": sum(len(r.sightings) for r in runs),
            "pose_observations_by_tag": {str(t): len(v) for t, v in sorted(poses.items())},
            "corner_px_median": float(np.median(px)), "residual_mm_median": float(np.median(mm)),
            "residual_mm_max": float(np.max(mm)), "runs": [r.name for r in runs]}


NOISE_PX = 0.8   # the corner error a plan is judged at: the rigid fits' median left about this


def information(c: Chain, k, dist, camera_from_base, q, tags, step: float = 1e-5) -> np.ndarray:
    """A hold's rows of the Jacobian at the measured chain: d(corner px)/d(camera 6, seat 6, joints 1-4) for the
    tags it would show from camera_from_base (forward differences)."""
    import cv2

    sightings = [{"q": np.asarray(q, float), "tag": t} for t in tags]
    x0 = np.r_[_vector(camera_from_base), np.zeros(6 + len(FREE_JOINTS))]

    def pixels(x):
        dq = np.zeros(7)
        dq[list(FREE_JOINTS)] = x[12:]
        return np.concatenate([cv2.projectPoints(c.corners(s, dq, _pose(x[6:12])).reshape(-1, 1, 3), x[:3], x[3:6], k,
                                                 dist)[0].ravel() for s in sightings])
    base = pixels(x0)
    columns = []
    for i in range(len(x0)):
        x = x0.copy()
        x[i] += step
        columns.append((pixels(x) - base) / step)
    return np.array(columns).T


def predicted_sigma(jacobians, noise_px: float = NOISE_PX) -> dict:
    """The 1-sigma the holds would give the seat (mm, deg) and joints 1-4 (mrad) at `noise_px` per corner."""
    sd = np.sqrt(np.clip(np.diag(np.linalg.pinv(sum(j.T @ j for j in jacobians))), 0.0, None)) * noise_px
    return {"seat_mm": np.round(sd[9:12] * 1e3, 2).tolist(), "seat_deg": np.round(np.degrees(sd[6:9]), 3).tolist(),
            "dq_mrad": np.round(sd[12:] * 1e3, 2).tolist()}


def describe(report: dict) -> list[str]:
    """The comparison as summary lines: each model's held-out error, the seat and the joint offsets."""
    models = report["models"]
    by = next(iter(models.values()))["held_out"]["by"]
    lines = [f"# chain over {len(report['runs'])} run(s), held out by {by}: " + ", ".join(
        f"{name} {row['held_out']['median_px']:.2f} px" + (" (best)" if name == report["best"] else "")
        for name, row in models.items())]
    seat = models.get("seat", {}).get("fit", {}).get("seat")
    if seat:
        lines.append(f"# the tags' seat on {seat['parent_frame']} (seat alone): "
                     f"{', '.join(f'{v:+.1f}' for v in seat['mm'])} mm +- {', '.join(f'{v:.1f}' for v in seat['sigma_mm'])}"
                     f"; {', '.join(f'{v:+.2f}' for v in seat['deg'])} deg")
    joints = models.get("seat+joints", {}).get("fit", {})
    if joints.get("dq_mrad"):
        lines.append("# joint offsets (with the seat): " + ", ".join(
            f"{name} {value:+.1f} +- {joints['dq_sigma_mrad'][name]:.1f}" for name, value in joints["dq_mrad"].items())
                     + " mrad")
    return lines


def report_for(c: Chain, runs: list[Run], out_dir: Path, models=None) -> dict:
    """compare, with the seat alone's layout written as out_dir/wrist-layout.json: the candidate for a stack that
    carries no joint model."""
    report, fits = compare(c, runs, models)
    if "seat" in fits:
        record = {**layout_record(c, runs, fits["seat"]), "model": "seat"}
        (Path(out_dir) / "wrist-layout.json").write_text(json.dumps(record, indent=2) + "\n")
        report["layout"] = "wrist-layout.json"
    return report


def main(argv=None) -> int:
    """`ros2 run tatbot_calib chain RUN...`: the chain over earlier registrations of one arm, pooled; moves nothing
    and adopts nothing. Writes chain.json and wrist-layout.json into ~/tatbot-logs/ros-chain/<run-id>/."""
    import argparse
    import sys

    from tatbot_calib import register

    parser = argparse.ArgumentParser(prog="chain", description=main.__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="ros-register run ids or directories, all of one arm")
    parser.add_argument("--arm", default="right", choices=sorted(register.TARGETS))
    ns = parser.parse_args(argv)
    from tatbot_description import repo_root, robot_description
    repo = repo_root()
    register._lib(repo)
    import tatbot_runlog
    from fiducials import load_inventory
    from tatbot_motion import Kinematics

    paths = [register.run_path(value) for value in ns.runs]
    other = [p.name for p in paths if json.loads((p / "report.json").read_text()).get("arm") != ns.arm]
    if other:
        print(json.dumps({"ok": False, "message": f"runs {other} did not register the {ns.arm} arm"}))
        return 2
    kin = Kinematics(robot_description(None, arms=(ns.arm,)), ns.arm)
    c = Chain(kin, ns.arm, load_inventory(repo / "config" / "fiducials.json").target(register.TARGETS[ns.arm]))
    runs = [run_from_dir(p) for p in paths]
    run = tatbot_runlog.init("ros-chain", meta={"arm": ns.arm, "runs": [r.name for r in runs]}, attach_logging=False,
                             argv=["tatbot_calib", "chain", *(argv or sys.argv[1:])])
    try:
        report = report_for(c, runs, Path(run.dir))
        (Path(run.dir) / "chain.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        run.finalize(1, status="fail")
        raise
    for line in describe(report):
        print(line, flush=True)
    run.finalize(0, status="ok")
    print(json.dumps({"ok": True, "run_dir": str(run.dir), "best": report["best"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
