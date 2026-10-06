"""The tool's tip and the ball, fitted from opposed side contacts on the station probe (tatbot_calib.program).

Every contact is a row of joints latched at the probe's edge. Through forward kinematics to the tool mount, the
tip `a` and its axis `u` (both in the tool mount) land in the base, where the ball's centre `c` sits: the distance
from c to the axis line is the contact's side radius (the wall's radius + the ball's, less the arm's give before
the probe trips). The give differs with the push's direction: pink's pairs at v11 read half-spans of 1.6 mm across
its heading and 2.7 mm along it, and repeated (2026-09-30), so each attitude's pair axis (d, p) takes its own side
radius. A pair's midpoint, where both pushes give alike, carries the tip; its half-span, the give.

Side contacts say nothing along the axis, and with upright attitudes only, the axis's tilt and the joint offsets
trade off with the tip across it: the caller holds those with priors. Validation refits without each attitude group
and predicts that group's contacts. Pure numpy and scipy; the forward kinematics is a callable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

SIGMA_CONTACT_M = 0.0001        # one contact's scatter (the probe's repeatability plus the joints')


@dataclass
class Contact:
    kind: str                   # side
    q: np.ndarray               # joints at the probe's edge (7)
    group: str = ""             # the attitude it was taken at
    axis: str = ""              # its pair axis: d | p


@dataclass
class Fit:
    tip: np.ndarray             # in the tool mount, m
    axis: np.ndarray            # unit, tool mount, out of the nose
    ball: np.ndarray            # the ball's centre in the base, m
    side_radius: float          # the side radii's mean
    rms_m: dict                 # per kind
    sigma: dict                 # 1-sigma of tip, axis angle, ball, from the Jacobian
    residuals_m: np.ndarray
    side_radii: dict = field(default_factory=dict)   # {(group, axis) or None: radius}

    def as_dict(self) -> dict:
        return {"tip_m": self.tip.tolist(), "axis": self.axis.tolist(), "ball_m": self.ball.tolist(),
                "side_radius_m": self.side_radius, "rms_m": self.rms_m, "sigma": self.sigma,
                "side_radii_m": {("/".join(key) if key else "all"): r for key, r in self.side_radii.items()}}


def _basis(u0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(u0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u0, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u0, e1)


def side_keys(contacts, per_axis: bool) -> tuple:
    """The side radii's keys: one radius (None), or one per (attitude, pair axis)."""
    if not per_axis:
        return (None,)
    return tuple(sorted({(c.group, c.axis) for c in contacts}))


def _key(contact: Contact, keys) -> tuple | None:
    return None if keys == (None,) else (contact.group, contact.axis)


class Model:
    """Packs and unpacks the parameters: tip (3), axis tilt (2), ball (3), the side radii (one per key)."""

    def __init__(self, tip0, axis0, keys=(None,)):
        self.tip0, self.u0 = np.asarray(tip0, float), np.asarray(axis0, float) / np.linalg.norm(axis0)
        self.e1, self.e2 = _basis(self.u0)
        self.keys = tuple(keys)

    def unpack(self, x):
        u = self.u0 + x[3] * self.e1 + x[4] * self.e2
        return x[0:3], u / np.linalg.norm(u), x[5:8], dict(zip(self.keys, x[8:], strict=True))

    def x0(self, ball0, side0):
        sides = [side0.get(key, np.mean(list(side0.values()))) for key in self.keys] if isinstance(side0, dict) \
            else [side0] * len(self.keys)
        return np.concatenate([self.tip0, [0.0, 0.0], ball0, sides])


def _distances(tip, axis, ball, contacts, frame_at) -> np.ndarray:
    """Each contact's distance from the ball's centre to the axis line."""
    out = np.empty(len(contacts))
    for i, contact in enumerate(contacts):
        mount = frame_at(np.asarray(contact.q, float))
        a_w, u_w = mount[:3, :3] @ tip + mount[:3, 3], mount[:3, :3] @ axis
        out[i] = np.linalg.norm(np.cross(ball - a_w, u_w))
    return out


def contact_residuals(x, model: Model, contacts, frame_at) -> np.ndarray:
    """Each contact's miss in metres: its distance to the axis less its side radius."""
    tip, axis, ball, sides = model.unpack(x)
    distance = _distances(tip, axis, ball, contacts, frame_at)
    return np.array([distance[i] - sides[_key(c, model.keys)] for i, c in enumerate(contacts)])


def fit(contacts, frame_at, *, tip0, axis0, ball0, side0, sigma_tip=0.010, sigma_axis=0.1, sigma_ball=0.010,
        per_axis=False) -> Fit:
    """Least squares over every contact, with priors about their starting values on the tip (sigma_tip, a scalar or
    one per mount axis), the axis tilt (sigma_axis, rad) and the ball (sigma_ball): with the axis upright, no side
    contact sees the ball's height, which without one ran off 40 km and took the tip with it. `per_axis`: a side
    radius per attitude and pair axis."""
    model = Model(tip0, axis0, side_keys(contacts, per_axis))
    b0 = np.asarray(ball0, float)

    def residuals(x):
        tip, _, ball, _ = model.unpack(x)
        prior = [*((tip - model.tip0) / sigma_tip), *(x[3:5] / sigma_axis), *((ball - b0) / sigma_ball)]
        return np.concatenate([contact_residuals(x, model, contacts, frame_at) / SIGMA_CONTACT_M, prior])

    result = least_squares(residuals, model.x0(ball0, side0), x_scale="jac", method="trf")
    tip, axis, ball, sides = model.unpack(result.x)
    miss = contact_residuals(result.x, model, contacts, frame_at)
    sides = {key: float(r) for key, r in sides.items()}
    return Fit(tip, axis, ball, float(np.mean(list(sides.values()))), {"side": float(np.sqrt(np.mean(miss ** 2)))},
               _sigma(result, len(contacts), len(result.x)), miss, sides)


def _sigma(result, n_contacts: int, n_params: int) -> dict:
    """1-sigma of the tip (3), the axis tilt (rad) and the ball (3) from the Jacobian, scaled by the fit's
    residual scatter when it exceeds the assumed contact sigma."""
    j = result.jac
    dof = max(1, n_contacts - n_params)
    scale = max(1.0, float(np.sum(result.fun[:n_contacts] ** 2)) / dof)
    try:
        cov = np.linalg.pinv(j.T @ j) * scale
    except np.linalg.LinAlgError:
        return {}
    s = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {"tip_m": s[0:3].tolist(), "axis_rad": float(np.hypot(s[3], s[4])), "ball_m": s[5:8].tolist()}


def leave_one_group_out(contacts, frame_at, **kwargs) -> dict:
    """For each attitude group: refit without it and predict its contacts. {group: rms miss in metres}. With
    `per_axis` a held-out pair axis's own side radius is its contacts' mean distance to the predicted axis (the give
    at that attitude, which no other attitude measures), so what is predicted is where its pairs centre."""
    out = {}
    for group in sorted({c.group for c in contacts if c.group}):
        kept = [c for c in contacts if c.group != group]
        held = [c for c in contacts if c.group == group]
        if len(kept) < 8 or not held:
            continue
        f = fit(kept, frame_at, **kwargs)
        distance = _distances(f.tip, f.axis, f.ball, held, frame_at)
        radius = {}
        for i, c in enumerate(held):
            radius.setdefault(_key(c, tuple(f.side_radii)), []).append(distance[i])
        if kwargs.get("per_axis"):
            # a pair axis with one contact predicts nothing (its own radius takes it all): an attitude skipped
            # after one touch read 0.000 mm held out (blue, round 20)
            radius = {key: float(np.mean(v)) for key, v in radius.items() if len(v) >= 2}
        else:
            radius = dict.fromkeys(radius, f.side_radius)
        miss = np.array([distance[i] - radius[_key(c, tuple(f.side_radii))] for i, c in enumerate(held)
                         if _key(c, tuple(f.side_radii)) in radius])
        if len(miss):
            out[group] = float(np.sqrt(np.mean(miss ** 2)))
    return out
