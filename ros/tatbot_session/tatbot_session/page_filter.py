"""The print's offset on the skin as one Kalman filter of the wrist's fits and the overhead's marker moves (ros/README.md
4.4, stencil tracking): what track.Tracker corrects each stroke by. It replaced track.Field, a fit of the wrist's
readings alone, removed 2026-10-24.

A wrist reading at u, where the stack believes the TCP is on the page in use, measures
    r(u) = M^-1 (F u + g(F u)) - u
with F the follows the page took since its setup (ArmExecutor._followed), M the sheet's move since setup, and g the
arm's error where the pen is, which stays put while the sheet moves. The state is M's slide and turn about the
drawing's centre c, g's affine part a + B (p - c), and the overhead's own slow error b. The sheet and the arm drift as
random walks; b decays to its own scatter in bias_s. Each wrist fit is taken with its own variance (floor^2 + (k
rms)^2 / markers; a gauge hold's hold_m more) and each overhead reading of the print's markers (geometry.marks_moved
against the setup's reference) as M plus b, with its frame's noise by the markers read, weighed down past its gate
rather than dropped, until move_n in a row read one place: the sheet moved. A wrist reading past its gate is held; a second that agrees as one rigid move, apart_m away,
opens the sheet's covariance and both are taken. A further wrist camera (`cameras`) reads the same residual plus an
offset of its own, a state learned where both read, and both share the stop's own error: on the forearm18 holder
(2026-10-23) the two D405s' readings at one stop differed by 0.12-0.15 mm about their offset, where each sat ~0.4 mm
from the next stop's, so a stop's second reading is mostly the same reading again (view_sd_m its own part).

Replayed on the 23 tracked runs of 2026-10-20..23 (internal/evidence/page_replay.py): the overhead's large drifts are
the sheet's (the forearm heart of 2026-10-23 slid 4.4 mm in 6 min, and the wrist's residual less it stayed within ~0.8
mm), but between them it errs 0.5-1 mm for tens of seconds, which a white-noise overhead cannot say: hence b. The
gauge holds, read at the skin, showed none of the -0.25 mm per mm gradient across that forearm that every hover and
lift over it read: hence hold_m.
"""
from __future__ import annotations

import copy

import numpy as np

SHEET, ARM, BIAS, CAMERAS = slice(0, 3), slice(3, 9), slice(9, 12), 12   # with further cameras: the stop's error,
# then each further camera's offset
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _yaw(m) -> float:
    return float(np.arctan2(m[1, 0], m[0, 0]))


def _apply(m, xy) -> np.ndarray:
    return np.atleast_2d(np.asarray(xy, float)) @ np.asarray(m, float)[:2, :2].T + np.asarray(m, float)[:2, 2]


class PageFilter:
    """The tracker's estimator (track.Tracker.field): add, predict, snapshot, corrected, samples, held."""

    # Tuned by replay (internal/evidence/page_replay.py) on the 23 tracked runs of 2026-10-20..23, held out by halves
    # first, then on all, and rounded: the sheet drifts drift_m_min and turns turn_rad_min per sqrt(minute), the arm's
    # error arm_m_min and its gradient arm_grad_min; a wrist fit is floor_m^2 + (k rms)^2 / markers (its repeat
    # scatter is 0.3-0.5 mm with 3+ markers and 0.7-1.1 with 1-2, the rest is what a view and a breath move); an
    # overhead frame overhead_m (4 / read)^overhead_p and overhead_rad, over its own error bias_m and bias_rad held
    # bias_s; the setup puts the sheet within setup_m and setup_rad; a gauge hold reads hold_m off a stroke's readings.
    # A reading while its stroke draws (page.track.stroke_s) weighs moving_m more: on the still holder of 2026-10-24
    # its 503 scattered ~0.35 mm an axis about the stops' filter, twice a stop's, with no lag along the motion, and
    # taken as stops put the next stop's reading 0.33 mm off where stops alone put it 0.19; at 3 mm 0.22, while the
    # bicep's stroke ends kept 2.66 mm of their 2.58 (stops alone 3.52). A pull is judged and taken without moving_m,
    # or that bicep's 6.5 mm move is never taken.
    def __init__(self, centre=(0.0, 0.0), drift_m_min=0.002, turn_rad_min=0.02, arm_m_min=0.0002, arm_grad_min=0.004,
                 floor_m=0.0006, k=2.8, overhead_m=0.0006, overhead_rad=0.005, overhead_p=1.25, overhead_gate=20.0,
                 bias_m=0.001, bias_rad=0.005, bias_s=60.0, setup_m=0.0001, setup_rad=0.015, arm_prior_m=0.010,
                 grad_prior=0.2, hold_m=0.003, gate=7.0, confirm_m=0.0015, apart_m=0.003, open_m=0.003,
                 open_rad=0.05, cap_m=0.060, cameras=1, view_m=0.003, view_m_min=0.0001, view_sd_m=0.00015,
                 stop_s=1.0, move_n=3, move_m=0.0015, move_rad=0.03, wrist_n=4, pull_gate=13.8, moving_m=0.003):
        extra = 2 * int(cameras) if cameras > 1 else 0   # the stop's error (2), each further camera's offset (2)
        self.c, self.n, self.views = np.asarray(centre, float), CAMERAS + extra, CAMERAS + 2
        self.q = np.r_[np.array([drift_m_min, drift_m_min, turn_rad_min, arm_m_min, arm_m_min, *[arm_grad_min] * 4]) ** 2,
                       np.zeros(3), np.zeros(min(extra, 2)), [view_m_min ** 2] * max(extra - 2, 0)] / 60
        self.view_sd, self.stop_s, self.stop = view_sd_m, stop_s, None   # the stop: (its time, its first reading's var)
        self.bias, self.bias_s = np.array([bias_m, bias_m, bias_rad]) ** 2, bias_s
        self.floor, self.k, self.hold, self.moving = floor_m, k, hold_m, moving_m
        self.o_m, self.o_rad, self.o_p, self.o_gate = overhead_m, overhead_rad, overhead_p, overhead_gate
        self.move, self.far = (move_n, move_m, move_rad), []   # the overhead's readings past its gate, in a row
        self.gate, self.confirm, self.apart, self.open, self.cap = gate, confirm_m, apart_m, (open_m, open_rad), cap_m
        self.wrist_n, self.pull_gate, self.recent = int(wrist_n), pull_gate, []   # the last checked readings' jumps
        self.setup = np.array([setup_m, setup_m, setup_rad]) ** 2
        self.x = np.zeros(self.n)
        self.P = np.diag(np.r_[self.setup, np.array([arm_prior_m, arm_prior_m, *[grad_prior] * 4]) ** 2, self.bias,
                               np.zeros(min(extra, 2)), [view_m ** 2] * max(extra - 2, 0)])
        self.t = None
        self.held = None
        self.samples: list = []   # the wrist readings taken: (t, u, r)
        self.follow = np.eye(3)   # a snapshot's: the page's follows since setup when it was taken

    # --- the model ---------------------------------------------------------------------------------------------------
    def _parts(self, x, u, follow) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(p = F u, w = p + g(p) - c - d, R(-theta)) at page-in-use points u (K, 2): r = R(-theta) w + c - u."""
        p = _apply(follow, u)
        return p, p + x[3:5] + (p - self.c) @ x[5:9].reshape(2, 2).T - self.c - x[:2], _rot(-x[2])

    def _residual(self, x, u, follow) -> np.ndarray:
        _, w, back = self._parts(x, u, follow)
        return w @ back.T + self.c - np.atleast_2d(u)

    def _jac(self, u, follow) -> np.ndarray:
        """d r / d x at each point (K, 2, n); the overhead's error b is not the wrist's, nor a camera's offset."""
        p, w, back = self._parts(self.x, u, follow)
        out = np.zeros((len(p), 2, self.n))
        out[:, :, 0:2], out[:, :, 3:5] = -back, back
        out[:, :, 2] = -w @ (back @ J).T                                   # d R(-theta) / d theta = -R(-theta) J
        for i, j in np.ndindex(2, 2):
            out[:, :, 5 + 2 * i + j] = back[:, i] * (p - self.c)[:, j:j + 1]   # g's gradient B_ij
        return out

    def to(self, t: float) -> None:
        """The state carried to t (the executor's clock): the sheet and the arm drift, b decays to its own scatter."""
        if self.t is not None and t > self.t:
            keep = np.exp(-(t - self.t) / self.bias_s)
            phi = np.r_[np.ones(9), [keep] * 3, np.ones(self.n - CAMERAS)]
            self.x = self.x * phi
            self.P = self.P * np.outer(phi, phi) + np.diag(self.q * (t - self.t))
            self.P[BIAS, BIAS] += np.diag(self.bias * (1 - keep ** 2))
        self.t = t if self.t is None else max(self.t, t)

    def _update(self, y, jac, cov_r) -> None:
        gain = self.P @ jac.T @ np.linalg.inv(jac @ self.P @ jac.T + cov_r)
        self.x = self.x + gain @ y
        joseph = np.eye(self.n) - gain @ jac
        self.P = joseph @ self.P @ joseph.T + gain @ cov_r @ gain.T

    # --- the readings ------------------------------------------------------------------------------------------------
    def add(self, t: float, u, r, check: bool = True, *, rms: float, markers: int, follow=None, camera: int = 0,
            moving: bool = False) -> str:
        """A wrist reading (residual r at u, page metres, `follow` the page's follows then) by `camera` (0: the session's,
        then the further ones): 'added', 'held' (past the gate: a second must agree) or 'moved' (a second agreed as one
        rigid move: both taken). Unchecked, a gauge hold's, read at the skin rather than over it, so hold_m less like
        the strokes' own. One taken while its stroke drew (`moving`) weighs moving_m more and is no stop's; a move it
        shows (a pull, a second view) is judged and taken on a stop's noise."""
        self.to(t)
        u, r, follow = np.asarray(u, float), np.asarray(r, float), np.eye(3) if follow is None else np.asarray(follow)
        stop = self.floor ** 2 + (self.k * rms) ** 2 / max(markers, 1) + (0.0 if check else self.hold ** 2)
        var = stop + (self.moving ** 2 if moving else 0.0)
        jac, cov_r = self._view_jac(u, follow, camera), np.eye(2) * var
        if self.n > CAMERAS and not moving:   # further cameras: this stop's error is a state its readings share
            cov_r = np.eye(2) * self._at_stop(t, var)
            jac[:, CAMERAS:self.views] = np.eye(2)
        y, s = r - self._read(u, follow, jac), jac @ self.P @ jac.T + cov_r
        still = np.eye(2) * (var - stop)
        if check and (pull := self._pulls(y, s - still)) is not None:   # the sheet moved: opened to it, taken whole
            self.P[SHEET, SHEET] += np.diag(np.r_[np.maximum(self.open[0] ** 2, pull ** 2), self.open[1] ** 2])
            self._update(y, jac, cov_r - still)
            self.samples.append((t, u, r))
            self.held = None
            return "moved"
        if check and float(y @ np.linalg.solve(s, y)) > self.gate:
            held = self.held
            if held is not None and np.linalg.norm(u - held[0]) < self.apart:
                return "held"   # the same place again: no second view
            if held is None or not self._one_move(u, y, held):
                self.held = (u, r, np.eye(2) * stop, follow, y, t, camera)
                return "held"
            self.P[SHEET, SHEET] += np.diag([self.open[0] ** 2] * 2 + [self.open[1] ** 2])
            for hu, hr, hcov, hf, _, ht, hc in (held, (u, r, np.eye(2) * stop, follow, y, t, camera)):
                hj = self._view_jac(hu, hf, hc)
                self._update(hr - self._read(hu, hf, hj), hj, hcov)
                self.samples.append((ht, hu, hr))
            self.held = None
            return "moved"
        self._update(y, jac, cov_r)
        self.samples.append((t, u, r))
        self.held = None
        return "added"

    # On a bicep (2026-10-24) a 6.5 mm move read at once while its stroke drew, on one marker a reading, was taken a
    # little at a time over some 17 s: each such reading weighs little, and the sheet drifts drift_m_min a minute.
    def _pulls(self, y, s) -> np.ndarray | None:
        """The mean jump of the last wrist_n checked readings when it lies past pull_gate (chi-square 99.9%, 2 dof) of
        its own spread: they keep pulling one way, as a moved sheet does and a misread or a breath does not."""
        self.recent = [*self.recent, (y, s)][-self.wrist_n:]
        if len(self.recent) < self.wrist_n:
            return None
        mean, spread = (np.mean(v, axis=0) for v in zip(*self.recent, strict=True))
        if float(mean @ np.linalg.solve(spread / self.wrist_n, mean)) <= self.pull_gate:
            return None
        self.recent = []
        return mean

    def _view_jac(self, u, follow, camera: int) -> np.ndarray:
        """d r / d x at one point (2, n) as `camera` reads it: the residual's, and its own offset (none for camera 0)."""
        jac = self._jac(u, follow)[0]
        if camera:
            jac[:, self.views + 2 * (camera - 1):self.views + 2 * camera] = np.eye(2)
        return jac

    def _at_stop(self, t: float, var: float) -> float:
        """A reading's own variance at its stop: a reading past stop_s of the stop's first starts a new one, its error
        (all of var but view_sd_m's) a fresh state; one at the stop adds view_sd_m and what it reads worse than the
        first."""
        if self.stop is None or t - self.stop[0] > self.stop_s:
            self.stop = (t, var)
            self.x[CAMERAS:self.views], self.P[CAMERAS:self.views, :], self.P[:, CAMERAS:self.views] = 0.0, 0.0, 0.0
            self.P[CAMERAS:self.views, CAMERAS:self.views] = np.eye(2) * max(var - self.view_sd ** 2, 0.0)
        return self.view_sd ** 2 + max(var - self.stop[1], 0.0)

    def _read(self, u, follow, jac) -> np.ndarray:
        """What a reading at u is predicted to read: the residual, plus its camera's offset (jac's own columns)."""
        return self._residual(self.x, u, follow)[0] + jac[:, CAMERAS:] @ self.x[CAMERAS:]

    def _one_move(self, u, y, held) -> bool:
        """This jump and the held one are one rigid move of the sheet: it keeps their points' distance."""
        du = u - held[0]
        return abs(np.linalg.norm(du + y - held[4]) - np.linalg.norm(du)) < self.confirm

    def overhead(self, t: float, move, read: int) -> None:
        """An overhead reading of the sheet's move since setup (3 x 3, setup page metres) by `read` common markers."""
        if read < 2:
            return
        self.to(t)
        s = (4.0 / read) ** self.o_p
        cov_r = np.diag([(self.o_m * s) ** 2] * 2 + [(self.o_rad * s) ** 2])
        jac = np.zeros((3, self.n))
        jac[:, SHEET] = jac[:, BIAS] = np.eye(3)
        z = np.r_[_apply(move, self.c)[0] - self.c, _yaw(move)]
        y = z - jac @ self.x
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        nis = float(y @ np.linalg.solve(jac @ self.P @ jac.T + cov_r, y))
        # Past the gate, move_n readings with none within it between that read one place (within move_m and move_rad
        # of the newest) are the sheet's move, not a misread: the sheet's covariance opens to it and it is taken whole.
        # The 2026-10-23 move test: a 22.8 mm slide the overhead held from 3-4 markers a second for 15 s before the
        # wrist's next reading, of which the weighing down alone had taken 74%.
        n, m, rad = self.move
        self.far = [] if nis <= self.o_gate else [
            f for f in self.far if np.linalg.norm(f[:2] - z[:2]) < m and abs(f[2] - z[2]) < rad] + [z]
        if len(self.far) >= n:
            self.P[SHEET, SHEET] += np.diag(y ** 2)
            self.far, nis = [], 0.0
        self._update(y, jac, cov_r * max(1.0, nis / self.o_gate))   # a misread weighs as little as its distance says

    def rebased(self, follow) -> None:
        """The overhead's watch starts again from the sheet as it reads it now (ArmExecutor._rebase), the page in use
        standing for it and its follows cleared: what the state predicts on the page in use goes into the arm's part,
        and the sheet and the overhead's error start again from the setup's."""
        corners = self.c + np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]])
        r = self.predict(corners, follow)[0]
        grad = np.c_[r[1] - r[0], r[2] - r[0]] / 0.01
        self.x = np.r_[np.zeros(3), r[0], grad.ravel(), np.zeros(3), self.x[CAMERAS:]]   # the cameras' offsets stay
        self.stop, self.far, self.recent = None, [], []
        keep, views = self.P[ARM, ARM], self.P[CAMERAS:, CAMERAS:]
        self.P = np.diag(np.r_[self.setup, np.zeros(6), self.bias, np.zeros(self.n - CAMERAS)])
        self.P[ARM, ARM], self.P[CAMERAS:, CAMERAS:] = keep, views
        self.held = None

    # --- the tracker's -----------------------------------------------------------------------------------------------
    def predict(self, points, follow=None) -> tuple[np.ndarray, np.ndarray]:
        """(r at each point (N, 2) on the page in use, its sigma (N,)), uncapped; with no follow, the snapshot's."""
        follow = self.follow if follow is None else np.asarray(follow)
        p, jac = np.atleast_2d(np.asarray(points, float)), self._jac(points, follow)
        return self._residual(self.x, p, follow), np.sqrt(np.einsum("kin,nm,kim->k", jac, self.P, jac) / 2)

    def snapshot(self, follow=None) -> PageFilter:
        snap = copy.deepcopy(self)
        snap.follow = np.eye(3) if follow is None else np.asarray(follow, float)
        return snap

    def corrected(self, op: dict) -> tuple[dict, np.ndarray]:
        """A copy of the op with its points moved by -r at them (capped at cap_m), and the corrections (N, 2)."""
        pts = np.asarray(op["points_m"], float)
        corr = self._residual(self.x, pts, self.follow)
        corr *= np.minimum(1.0, self.cap / np.maximum(np.linalg.norm(corr, axis=1), 1e-12))[:, None]
        return {**op, "points_m": (pts - corr).tolist()}, corr
