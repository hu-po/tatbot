"""Stencil tracking (ros/README.md 4.4): the pen tip on the print, measured with the wrist D405.

With `page.track.mode` log or correct, the wrist D405 stays open through a draw's op loop. A worker process fits
frames to the print (scripts/vision/stencil_tip.py), so the fit's ~0.3 s never holds the executor:
- the goal's own gauge holds, before the first stroke (mode correct);
- with `hover_stop`, a fresh frame at each stroke's hover before it descends, fitted at once (at_hover);
- a fresh frame at the end of each stroke's final lift.
Each fit is a page event and a row of <run>/track.jsonl, its frame kept under <run>/track/.

An accepted fit gives a residual: where the pen inks (the gauge's tip point carried onto the print, plus the dot
test's offset at the tip's height, turned with the camera) less where the stack believes the TCP is (FK in the run's page frame). With
mode correct the executor moves each stroke's points by the residual field at them (Field), so the ink lands on
the plan however the sheet has slid or turned and wherever the arm's error is.
"""
from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tatbot_interfaces.msg import Event

from tatbot_session import gauge, geometry
from tatbot_session import inspect as ins

_LAYOUT = None


def _init(repo: str, pattern_id: str) -> None:
    global _LAYOUT
    from pathlib import Path

    from tatbot_bridge import capture

    capture._lib(Path(repo))
    import stencil_tip

    _LAYOUT = stencil_tip.load_layout(pattern_id)


def _measure(job: dict) -> dict:
    """In the worker: keep the frame under track/ (unless it is kept already), fit it, return the log row. A fit
    the run's page cannot find is tried again from the overhead's page when it has seen the sheet move: the
    wrist fit reaches about two lattice steps from its prior, and a sheet slid 50 mm is farther."""
    if job.get("png"):
        cv2.imwrite(job["png"], job["bgr"])
        np.savez_compressed(job["npz"], depth_raw=job["depth"], q=job["q"], meta=json.dumps(job["meta"]))
    t0 = time.monotonic()
    out = _fit(job, job["used"])
    if not out.get("accepted") and job.get("overhead") is not None:
        again = _fit(job, job["overhead"])
        if again.get("fit") and again["margin"] > out.get("margin", -1):
            out = {**again, "prior": "overhead"}
    out["fit_s"] = round(time.monotonic() - t0, 3)
    return out


def _fit(job: dict, base_from_page) -> dict:
    import stencil_tip

    try:
        return stencil_tip.measure(job["bgr"], job["depth"], job["meta"], job["base_from_camera"], base_from_page,
                                   _LAYOUT, job["tip_cam"])
    except ValueError as error:
        return {"fit": None, "error": str(error)}


class Field:
    """The residual field r(p) (page metres) from samples (t, u, r): u where the stack believed the TCP was, r where
    the pen inks less u. Two parts, both weighted by age against the newest sample (tau_s):
    - global: an affine fit r = a + B u once three samples span spread_m both ways (else their mean). A slide and a
      turn of the sheet are affine, so it carries them exactly, past the samples too;
    - local: what the global part leaves at the samples, averaged by distance (sigma_m) and shrunk toward 0
      (prior_w) where none is near: the arm's position-dependent error.
    Samples spanning spread_m one way only fit a slide and a turn instead (the two of a confirmed move).
    A stroke's correction is capped at cap_m. A sample farther from the field than jump_m (plus `turn` per metre to
    its nearest sample: a turned sheet's residual grows with distance) is held. A second jump consistent with it
    as one rigid move of the sheet (the jumps keep the two samples' distance within confirm_m; a lattice misfit
    does not) makes it a real move, and the older samples go. The goal's gauge holds go in together, unchecked:
    they set the field."""

    def __init__(self, sigma_m=0.03, tau_s=180.0, prior_w=0.05, spread_m=0.005, jump_m=0.00325, turn=0.1,
                 confirm_m=0.0015, cap_m=0.060):
        self.sigma, self.tau, self.prior_w, self.spread = sigma_m, tau_s, prior_w, spread_m
        self.jump, self.turn, self.confirm, self.cap = jump_m, turn, confirm_m, cap_m
        self.samples: list[tuple[float, np.ndarray, np.ndarray]] = []
        self.held = None

    def _global(self, u, r, wt):
        """(3, 2) coefficients [a; B] of r = a + B u: affine with the spread both ways, a slide and a turn with it
        one way, else the weighted mean (B = 0)."""
        mean_u = (wt @ u) / wt.sum()
        cov = ((u - mean_u) * wt[:, None]).T @ (u - mean_u) / wt.sum()
        spread = np.linalg.eigvalsh(cov)
        if len(u) >= 3 and spread[0] > self.spread ** 2:
            x = np.c_[np.ones(len(u)), u] * np.sqrt(wt)[:, None]
            return np.linalg.lstsq(x, r * np.sqrt(wt)[:, None], rcond=None)[0]
        mean_r = (wt @ r) / wt.sum()
        if spread[1] <= self.spread ** 2:
            return np.vstack([mean_r, np.zeros((2, 2))])
        a, b = u - mean_u, u + r - mean_u - mean_r            # where the stack believed, where the pen inks
        turn = np.arctan2(wt @ (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), wt @ (a * b).sum(1))
        rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        return np.vstack([mean_u + mean_r - rot @ mean_u, (rot - np.eye(2)).T])

    def predict(self, points) -> tuple[np.ndarray, np.ndarray]:
        """(r at each point (N, 2), the distance weight behind it (N,)), uncapped."""
        p = np.atleast_2d(np.asarray(points, float))
        if not self.samples:
            return np.zeros_like(p), np.zeros(len(p))
        t = np.array([s[0] for s in self.samples])
        u = np.array([s[1] for s in self.samples])
        r = np.array([s[2] for s in self.samples])
        wt = np.exp(-(t.max() - t) / self.tau)
        coef = self._global(u, r, wt)
        resid = r - np.c_[np.ones(len(u)), u] @ coef
        w = np.exp(-((p[:, None, :] - u[None, :, :]) ** 2).sum(-1) / (2 * self.sigma ** 2)) * wt[None, :]
        local = (w @ resid) / (w.sum(1) + self.prior_w)[:, None]
        return np.c_[np.ones(len(p)), p] @ coef + local, w.sum(1)

    def add(self, t: float, u, r, check: bool = True) -> str:
        """'added', 'held' (far from the field: wait for a second), or 'moved' (the second agreed)."""
        u, r = np.asarray(u, float), np.asarray(r, float)
        if self.samples and check:
            pred, weight = self.predict(u)
            near = min(float(np.linalg.norm(s[1] - u)) for s in self.samples)
            jump = r - pred[0]
            if weight[0] > 0.3 and np.linalg.norm(jump) > self.jump + self.turn * near:
                if self.held is not None and self._one_move(u, jump):
                    self.samples, self.held = [self.held[0], (t, u, r)], None
                    return "moved"
                self.held = ((t, u, r), jump)
                return "held"
        self.samples.append((t, u, r))
        self.held = None
        return "added"

    def _one_move(self, u, jump) -> bool:
        """Whether this jump and the held one are one rigid move: it keeps their samples' distance."""
        (_, held_u, _), held_jump = self.held
        du = u - held_u
        return abs(np.linalg.norm(du + jump - held_jump) - np.linalg.norm(du)) < self.confirm

    def snapshot(self) -> Field:
        snap = Field(self.sigma, self.tau, self.prior_w, self.spread, self.jump, self.turn, self.confirm, self.cap)
        snap.samples = list(self.samples)
        return snap

    def corrected(self, op: dict) -> tuple[dict, np.ndarray]:
        """A copy of the op with its points moved by -r (capped at cap_m), and the corrections (N, 2)."""
        pts = np.asarray(op["points_m"], float)
        corr, _ = self.predict(pts)
        corr *= np.minimum(1.0, self.cap / np.maximum(np.linalg.norm(corr, axis=1), 1e-12))[:, None]
        return {**op, "points_m": (pts - corr).tolist()}, corr


class Tracker:
    """One draw goal's wrist-camera tracking. `start` returns None when tracking is off or cannot run."""

    def __init__(self, ex, cam, pool, tip_cam, cfg):
        self.ex, self.cam, self.pool, self.tip_cam = ex, cam, pool, tip_cam
        self.correcting = cfg.get("mode") == "correct"
        hover, contact = cfg.get("tip_offset_m", [0.0, 0.0]), cfg.get("tip_offset_contact_m") or cfg.get("tip_offset_m", [0.0, 0.0])
        self.offsets = [(float(cfg.get("hover_height_m", 0.011)), np.asarray(hover, float)),
                        (float(cfg.get("contact_height_m", 0.0032)), np.asarray(contact, float))]
        self.field = Field(**{k: float(v) for k, v in (cfg.get("field") or {}).items()})
        self.pending = []
        (ex.run_dir / "track").mkdir(exist_ok=True)

    @classmethod
    def start(cls, ex, since: float = 0.0) -> Tracker | None:
        """Open the camera and the worker; with mode correct, also fit the goal's gauge holds (written after
        `since`, a time.time()) and wait for them, so the first stroke has a field."""
        cfg = ex.stack["page"].get("track") or {}
        if cfg.get("mode", "off") not in ("log", "correct") or ex.stack.get("hardware") != "real" or ex.run_dir is None:
            return None
        path = gauge.calibration_path(ex.arm)
        if not path.exists():
            ex.event(Event.KIND_ERROR, "track: no wrist gauge calibration; not tracking")
            return None
        tip_cam = np.asarray(json.loads(path.read_text())["tip_cam_m"], float)
        try:
            cam = ins.WristCamera(serial=str((ex.stack["page"].get("locate") or {}).get("serial", "")), depth=True)
        except RuntimeError as error:
            ex.event(Event.KIND_ERROR, f"track: wrist camera: {error}; not tracking")
            return None
        pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"), initializer=_init,
                                   initargs=(str(ex.repo), ex.stack["page"]["pattern_id"]))
        tracker = cls(ex, cam, pool, tip_cam, cfg)
        ex.event(Event.KIND_PAGE, "track: the pen tip on the print after every lift" +
                 (", each stroke corrected by it" if tracker.correcting else " (logged, no correction)"))
        if tracker.correcting:
            tracker._seed(since)
        return tracker

    def _seed(self, since: float) -> None:
        for npz in sorted((self.ex.run_dir / "touches").glob("gauge-*.npz")):
            if npz.stat().st_mtime < since:
                continue
            z = np.load(npz, allow_pickle=True)
            self._submit(npz.stem, None, cv2.imread(str(npz.with_suffix(".png"))), z["depth_raw"],
                         json.loads(str(z["meta"])), np.asarray(z["q"], float), keep=False)
        self.drain(wait=True)

    def _submit(self, name, end, bgr, depth, meta, q, keep=True, op_id="") -> None:
        used = self.ex.used()
        job = {"bgr": bgr, "depth": depth, "meta": meta, "q": q, "used": used, "tip_cam": self.tip_cam,
               "overhead": self._overhead(used),
               "base_from_camera": self.ex.kin.frame(q, ins.camera_frame(self.ex.arm)),
               "png": str(self.ex.run_dir / "track" / f"{name}.png") if keep else None,
               "npz": str(self.ex.run_dir / "track" / f"{name}.npz")}
        tcp = (np.linalg.inv(used) @ self.ex.kin.fk(q))[:2, 3]
        self.pending.append((name, op_id, end, tcp, time.monotonic(), self.pool.submit(_measure, job)))

    def _overhead(self, used: np.ndarray) -> np.ndarray | None:
        """The run's page moved as the overhead measures the sheet moved since setup; None without a measurement
        within page.max_lost_s or when it moved under 2 mm and 1 deg (the same prior, near enough)."""
        newest = self.ex.node.camera_page(self.ex.arm)
        lost = float(self.ex.stack["page"].get("max_lost_s", 5.0))
        if newest is None or geometry.stale_page(newest[1], lost) or self.ex.camera is None:
            return None
        page = newest[0] @ np.linalg.inv(self.ex.camera) @ used
        return page if geometry.page_moved(used, page, 0.002, 0.0175) else None

    def after_lift(self, op: dict) -> None:
        """A fresh frame now (the arm at the stroke's lifted end), fitted in the worker."""
        self.drain()
        end = np.asarray(op["points_m"][-1 if not op.get("closed") else 0], float)
        self._grab(op["id"], op["id"], end)

    def at_hover(self, name: str, op_id: str, plan_xy) -> str | None:
        """A fresh frame now (the arm at rest over plan_xy, about to descend), fitted and logged before returning:
        the field's verdict on it ('added', 'held' or 'moved'), None when the print was not found or refused."""
        self.drain(wait=True)
        self._grab(name, op_id, np.asarray(plan_xy, float))
        return self.drain(wait=True)

    def _grab(self, name: str, op_id: str, end: np.ndarray) -> None:
        while self.cam.pipe.poll_for_frames():   # drop queued frames: the next one is taken where the arm is now
            pass
        colors, _ = self.cam.grab_both(1)
        self._submit(name, end, colors[0], self.cam.raw_depth[0], self.cam.depth_metadata, self.ex.io.q.copy(),
                     op_id=op_id)

    def snapshot(self) -> Field:
        self.drain()
        return self.field.snapshot()

    def drain(self, wait: bool = False) -> str | None:
        """Log every finished measurement (all of them with wait); the field's verdict on the last."""
        keep, verdict = [], None
        for item in self.pending:
            if wait or item[-1].done():
                verdict = self._log(*item[:-1], item[-1].result())
            else:
                keep.append(item)
        self.pending = keep
        return verdict

    def _offset(self, height: float, cam_from_print) -> np.ndarray:
        """The dot test's offset at the tip's height (the hover's, the contact's, linear between), in the print. It
        rides with the pen, so it is kept in the camera's axes and turned by the camera's yaw on the print."""
        (h1, o1), (h0, o0) = self.offsets
        a = min(max((height - h0) / (h1 - h0), 0.0), 1.0) if h1 != h0 else 1.0
        x = np.linalg.inv(np.asarray(cam_from_print, float))[:2, 0]   # the camera's x axis, in the print
        c, s = x / np.linalg.norm(x)
        return np.array([[c, -s], [s, c]]) @ (o0 + a * (o1 - o0))

    def _log(self, name: str, op_id: str, end, tcp, t: float, row: dict) -> str | None:
        row = {"op": name, "op_id": op_id, "plan_end_m": None if end is None else end.tolist(),
               "tcp_page_m": tcp.tolist(), **row}
        if not row.get("fit"):
            text = f"track {name}: the print was not found ({row.get('error', 'too few knots')})"
        else:
            tip = np.asarray(row["tip_m"][:2]) + self._offset(row["tip_m"][2], row["cam_from_print"])
            text = f"track {name}: tip on print ({tip[0] * 1e3:+.2f}, {tip[1] * 1e3:+.2f}) mm"
            if end is not None:
                off = tip - end
                row["off_plan_m"] = off.tolist()
                text += f", {off[0] * 1e3:+.2f} {off[1] * 1e3:+.2f} mm from the plan"
            text += f"; {row['inliers']} knots, margin {row['margin']}"
            text += " (from the overhead's page)" if row.get("prior") == "overhead" else ""
            if row["accepted"]:
                resid = tip - tcp
                row.update(residual_m=resid.tolist(), field=self.field.add(t, tcp, resid, check=end is not None))
                text += f"; ink {resid[0] * 1e3:+.2f} {resid[1] * 1e3:+.2f} mm from the TCP"
                text += {"added": "", "held": " (held: far from the field, a second must agree)",
                         "moved": " (a second agreed: the sheet moved, the field starts again)"}[row["field"]]
            else:
                text += " (not accepted)"
        self.ex.event(Event.KIND_PAGE, text, op_id)
        with open(self.ex.run_dir / "track.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        return row.get("field")

    def close(self) -> None:
        try:
            self.drain(wait=True)
        finally:
            self.pool.shutdown(wait=True)
            self.cam.close()
