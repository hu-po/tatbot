"""Palette dips: where a cap is, how the tool goes into it, and the motion there and back. Pure numpy;
tatbot_session.cap executes it (ros/README.md section 6).

The cap is placed from the station touched by the probe (station.touched_palette), its support frame from
urdf/palette.urdf and its bore from config/palette.yaml. The tool is the fitted datasheet's body of revolution,
ending at the loaded TCP, the end that takes ink (a needle cartridge's tube end).

A dip, as the tool's datasheet `dip:` block says (the program carries it per resource):
- approach: lift off the page to the standoff, rise until the tool clears the station's top, turn to the dip's
  attitude, cross to over the cap and come down its axis to `hover_m` over the rim;
- descend the axis until the tool's end stands `above_ink_m` over the cap's declared ink level;
- dwell `dwell_s` there (the machine running, its needles pulling ink into the tube);
- retract the axis to the hover, and return the way it came to the page standoff it left.

The tool keeps `wall_margin_m` from the bore wherever it is inside the cap: the margin is what the station
touch, the arm's tracking and the cap's seat may be off by together. Every control tick of every phase is checked
before knot resampling, and again as it runs (check_tick).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from tatbot_contracts.dip import validate_dip
from tatbot_description.transforms import rpy_matrix

from tatbot_motion import plan as pplan
from tatbot_motion import timelaw as tl

FLOOR_CLEAR_M = 0.0005     # the tool's end never comes nearer the cap's inner floor than this
TRACK_M = 0.0005           # a measured tick may stray this far from the planned axis line inside the cap
OVERSHOOT_M = 0.001        # going in, the tool's end may pass its dwell height by this much; the arm settles ~0.5 mm
                           # into its dwell (cap M2, 2026-10-05)
HOVER_LAG_M = 0.002        # a goal that rises to the hover may end this far under it while the arm catches up (a
                           # retract out of cap M2 ended 0.5 mm short, 2026-10-05); the hover is hover_m over the rim
STATION_CLEAR_M = 0.010    # the tool's lowest point over the station's top while it crosses


def _pose(value):
    value = np.array(value, float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError('cap/tool pose must be a finite rigid transform')
    r = value[:3, :3]
    if not (np.allclose(value[3], [0, 0, 0, 1], atol=1e-10, rtol=0) and
            np.allclose(r.T @ r, np.eye(3), atol=1e-9, rtol=0) and np.linalg.det(r) > 0):
        raise ValueError('cap/tool pose must be a proper rigid transform')
    return value


def _positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('cap dimensions must be finite positive metres')
    return float(value)


@dataclass(frozen=True)
class Cap:
    slot: str
    base_from_cap: np.ndarray  # cap support floor, +z out of the cap
    inner_floor_m: float       # the bore's floor over the support
    rim_m: float               # the rim over the support
    bore_radius_m: float

    def __post_init__(self):
        pose = _pose(self.base_from_cap)
        pose.flags.writeable = False
        object.__setattr__(self, 'base_from_cap', pose)
        for value in (self.inner_floor_m, self.rim_m, self.bore_radius_m):
            _positive(value)
        if self.inner_floor_m >= self.rim_m:
            raise ValueError('cap floor must be below its rim')

    @property
    def axis(self):
        return self.base_from_cap[:3, 2]


def read_cap(repo, slot, arm, base_from_palette) -> Cap:
    """A configured slot of this arm: its URDF support frame placed by base_from_palette, its bore from its outside
    dimensions less the wall (config/palette.yaml)."""
    import yaml

    palette = yaml.safe_load((repo / 'config/palette.yaml').read_bytes())
    item = palette['slots'].get(slot)
    if not isinstance(item, dict) or item.get('arm') != arm:
        raise ValueError('dip requires a canonical cap slot assigned to this arm')
    size = palette['sizes'][item['size']]
    diameter, height, wall = (_positive(size[key]) for key in ('diameter_m', 'height_m', 'wall_m'))
    if height <= wall or diameter <= 2 * wall:
        raise ValueError('cap wall/base thickness leaves no usable bore')
    joints = [joint for joint in ET.fromstring((repo / 'urdf/palette.urdf').read_bytes()).iter('joint')
              if joint.find('child') is not None and joint.find('child').get('link') == slot]
    return Cap(slot, _pose(base_from_palette) @ cap_support_frame(joints), wall, height, diameter / 2 - wall)


def cap_support_frame(joints):
    """One fixed support directly on palette_root, shared by bore and obstacle geometry."""
    if len(joints) != 1:
        raise ValueError('cap requires exactly one URDF support joint')
    joint = joints[0]
    parent, origin = joint.find('parent'), joint.find('origin')
    if joint.get('type') != 'fixed' or parent is None or parent.get('link') != 'palette_root' or origin is None:
        raise ValueError('cap support must be fixed directly to palette_root')
    return _pose(rpy_matrix([float(v) for v in origin.get('xyz', '0 0 0').split()],
                            [float(v) for v in origin.get('rpy', '0 0 0').split()]))


@dataclass(frozen=True)
class Tool:
    """The tool as a body of revolution about its axis: radius r at height h up the axis from the TCP (its end)."""
    heights: np.ndarray
    radii: np.ndarray

    @classmethod
    def from_profile(cls, profile) -> Tool:
        """A datasheet profile ([z, r], ... along the axis, the tool's end last) with its end at the TCP."""
        p = np.asarray(profile, float)
        if p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError('dip tool requires a finite axial profile')
        if np.any(np.diff(p[:, 0]) <= 0) or np.any(p[:, 1] <= 0):
            raise ValueError('dip profile requires increasing axial samples and positive radii')
        h = p[-1, 0] - p[::-1, 0]
        return cls(h, p[::-1, 1].copy())

    def radius_at(self, h):
        return np.interp(h, self.heights, self.radii)

    def widest(self, up_to_m: float) -> float:
        """The widest radius from the end up to up_to_m (the profile is linear between samples)."""
        inside = self.heights[self.heights < up_to_m]
        return float(max([self.radius_at(up_to_m), *self.radii[:len(inside)]]))

    def lowest(self, pose) -> float:
        """The base z of the tool's lowest point at a TCP pose (+z along the tool, toward its end)."""
        pose = np.asarray(pose, float)
        axis = pose[:3, 2]
        across = math.sqrt(max(0.0, 1.0 - float(axis[2]) ** 2))
        centres = pose[2, 3] - self.heights * axis[2]
        return float(np.min(centres - self.radii * across))


class Dip:
    """One cap entry: the hover over the rim and the dwell over the ink at one attitude, with the checks."""

    def __init__(self, cap: Cap, tool: Tool, settings, level_m: float, rotation, *, reach_m: float | None = None):
        """reach_m: how far the running needles reach past the tool's end (None: not recorded, so the machine
        stays off). Recorded, they must reach the ink from the dwell and stay off the cap's floor."""
        self.cap, self.tool, self.reach_m = cap, tool, reach_m
        validate_dip(settings)
        self.settings = MappingProxyType(dict(settings))
        depth = cap.rim_m - cap.inner_floor_m
        if not (math.isfinite(level_m) and 0 < level_m < depth):
            raise ValueError(f"{cap.slot}: the ink level must be over the inner floor and under the rim "
                             "(tatbot ros palette load SLOT=INK --level-mm SLOT=MM)")
        self.cap_from_base = np.linalg.inv(cap.base_from_cap)
        rotation = np.asarray(rotation, float)
        self.rotation = tl.align_rotation(rotation[:, 2], -cap.axis) @ rotation   # the tool down the cap's axis
        self.dwell_height_m = cap.inner_floor_m + level_m + settings['above_ink_m']
        if self.dwell_height_m < cap.inner_floor_m + FLOOR_CLEAR_M:
            raise ValueError(f"{cap.slot}: the dwell would put the tool's end on the cap's floor")
        if self.dwell_height_m >= cap.rim_m:
            raise ValueError(f"{cap.slot}: the declared ink is so high the tool's end stays over the rim")
        if reach_m is not None:
            if not reach_m > settings['above_ink_m']:
                raise ValueError(f"{cap.slot}: the needles reach {reach_m * 1000:.1f} mm past the tube's end, which the "
                                 f"dwell holds {settings['above_ink_m'] * 1000:.1f} mm over the ink: they never touch it")
            if self.dwell_height_m - reach_m < cap.inner_floor_m + FLOOR_CLEAR_M:
                raise ValueError(f"{cap.slot}: the running needles would reach the cap's floor: fill it deeper than "
                                 f"{(reach_m - settings['above_ink_m'] + FLOOR_CLEAR_M) * 1000:.1f} mm")
        immersed = cap.rim_m - self.dwell_height_m
        self.slack_m = cap.bore_radius_m - tool.widest(immersed) - settings['wall_margin_m']
        if self.slack_m < TRACK_M:
            raise ValueError(f"{cap.slot}: the tool is {tool.widest(immersed) * 2000:.1f} mm across {immersed * 1000:.1f} mm "
                             f"into the cap, which leaves {self.slack_m * 1000:+.2f} mm of the "
                             f"{cap.bore_radius_m * 2000:.1f} mm bore beyond its wall margin")
        self.dwell = self.at(self.dwell_height_m)
        self.hover = self.at(self.hover_height_m)

    @property
    def hover_height_m(self) -> float:
        return self.cap.rim_m + self.settings['hover_m']

    def at(self, height_m: float) -> np.ndarray:
        """The TCP pose on the cap's axis this high over its support, at the dip's attitude."""
        pose = np.eye(4)
        pose[:3, :3] = self.rotation
        pose[:3, 3] = (self.cap.base_from_cap @ [0, 0, height_m, 1])[:3]
        return pose

    def local(self, pose) -> tuple[float, float]:
        """(the TCP's height over the cap's support, its distance off the cap's axis)."""
        p = self.cap_from_base @ np.append(np.asarray(pose, float)[:3, 3], 1.0)
        return float(p[2]), float(np.hypot(p[0], p[1]))

    def in_cap(self, pose) -> bool:
        """The tool's end under the rim and over the bore: in the cap, or entering it."""
        height, off = self.local(pose)
        return height < self.cap.rim_m + self.settings['hover_m'] - 1e-4 and off < self.cap.bore_radius_m


def headings(rotation, step_deg: float = 45.0) -> list[np.ndarray]:
    """Upright tool rotations to try for a dip, the given one's heading first and then outward from it in
    step_deg turns about the vertical: the arm reaches a cap at some headings and not others (joint 3 at the
    crossing's height, joint 5's wrist turn), and the touches' heading is where the tcp's own error cancels."""
    r = np.asarray(rotation, float)
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    out = []
    for k in [0, *[s * n for n in range(1, int(180 / step_deg) + 1) for s in (-1, 1)]]:
        a = yaw + math.radians(k * step_deg)
        c, s = math.cos(a), math.sin(a)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        candidate = rz @ np.diag([1.0, -1.0, -1.0])
        if not any(np.allclose(candidate, seen, atol=1e-9) for seen in out):
            out.append(candidate)
    return out


def check_tick(dip: Dip, stage: str, pose, station_top_m: float) -> None:
    """ValueError when a planned or measured tick leaves its stage's envelope: over the station's top while it
    crosses; inside the cap over its floor, and going in on its axis and no deeper than the dwell and OVERSHOOT_M.
    Leaving (retract, cap_exit) starts wherever the tool is and only rises."""
    if stage in ('align', 'transit', 'over_page') and \
            dip.tool.lowest(pose) < station_top_m + STATION_CLEAR_M - 1e-6:
        raise ValueError(f"dip {stage}: the tool comes within {STATION_CLEAR_M * 1000:.0f} mm of the station's top")
    if stage in ('hover', 'descend', 'dwell', 'retract', 'cap_exit'):
        height, off = dip.local(pose)
        # going in keeps to the axis; leaving rises from wherever the tool is, since refusing it traps the tool in
        # the cap (an arm that sagged 9 mm and 6.7 mm aside in a restart, 2026-10-05)
        if stage in ('hover', 'descend', 'dwell') and height < dip.cap.rim_m + dip.settings['hover_m'] \
                and off > dip.slack_m:
            raise ValueError(f"dip {stage}: the tool's end is {off * 1000:.2f} mm off the cap's axis inside it, "
                             f"over the {dip.slack_m * 1000:.2f} mm the bore leaves it")
        if height < dip.cap.inner_floor_m + FLOOR_CLEAR_M:
            raise ValueError(f"dip {stage}: the tool's end comes within {FLOOR_CLEAR_M * 1000:.1f} mm of the cap's floor")
        if stage in ('hover', 'descend', 'dwell') and height < dip.dwell_height_m - OVERSHOOT_M:
            raise ValueError(f"dip {stage}: the tool's end goes {(dip.dwell_height_m - height) * 1000:.2f} mm under its "
                             "dwell height")


SLOWER = (1.0, 0.5, 0.25)   # a travel the CLIK cannot track at full speed is planned again slower


def _travel(dip, targets, q_seed, kin, motion, station_top_m, check, axial=False):
    """One goal through the target poses (name, pose, feedback phase), checked at every control tick. A plan whose
    tip trails its reference past the CLIK's bound is planned again at half and a quarter of the speed: the first
    real dip's return to the page missed by 0.01 mm over its 375 mm crossing (2026-10-05)."""
    from tatbot_motion.clik import PlanError

    for factor in SLOWER:
        try:
            return _travel_at(dip, targets, q_seed, kin, motion, station_top_m, check, axial, factor)
        except PlanError as error:
            if factor == SLOWER[-1] or 'its reference' not in str(error):
                raise
    raise AssertionError('unreachable')


def _travel_at(dip, targets, q_seed, kin, motion, station_top_m, check, axial, factor):
    seed = kin.fk(q_seed)
    dt, _ = pplan._cfg(motion)
    counts = []
    speed = factor * (min(dip.settings['speed_m_s'], motion['tip_speed']['pen_up_max_m_s']) if axial
                      else motion['tip_speed']['pen_up_max_m_s'])
    group = 'dip' if axial else 'approach'

    def build(scales):
        legs, previous = [], seed
        for stage, target, feedback in targets:
            if stage == 'dwell':
                p, r = tl.hold_rows(seed[:3, 3], seed[:3, :3], math.ceil(dip.settings['dwell_s'] / dt) * dt, dt)
            else:
                p, r = tl.line_segment(previous[:3, 3], target[:3, 3], previous[:3, :3], target[:3, :3],
                                       speed * scales.get(group, 1.), dt,
                                       omega_max=factor * motion['approach']['omega_max_rad_s'] * scales.get(group, 1.))
            legs.append(pplan._Leg(p, r, False, feedback, group))
            previous = target
        counts[:] = [len(leg.p) for leg in legs]
        return legs

    def checked(rows):
        start = 0
        for (stage, _, _), count in zip(targets, counts, strict=True):
            segment = rows[start:start + count + 1]
            check(stage, segment)
            for q in segment[1:] if start else segment:
                check_tick(dip, stage, kin.fk(q), station_top_m)
            start += count

    traj = pplan._solve(build, q_seed, kin, motion, check=checked)
    traj.info.update(stages_s=dict(zip([t[0] for t in targets], (np.cumsum(counts) * dt).tolist(), strict=True)),
                     slot=dip.cap.slot)
    return traj


def safe_height(dip: Dip, station_top_m: float, motion) -> float:
    """The TCP's height while it crosses: the tool's lowest point the clearance over the station's top."""
    if not math.isfinite(station_top_m):
        raise ValueError('a dip needs a finite station top')
    below = max(0.0, dip.hover[2, 3] - dip.tool.lowest(dip.hover))   # how far the body reaches under the TCP
    return station_top_m + STATION_CLEAR_M + below + float(motion['probe']['clearance_m'])


def plan_approach(dip: Dip, *, q_seed, kin, motion, base_from_page, station_top_m, check):
    """From the page to the cap's hover: lift to the standoff, rise, turn, cross, come down the cap's axis.
    The trajectory's info carries return_tcp, the page standoff the return comes back to."""
    page, seed = _pose(base_from_page), kin.fk(q_seed)
    if dip.cap.axis[2] <= 0:
        raise ValueError('a dip needs an upward cap axis')
    top = safe_height(dip, station_top_m, motion)
    lifted = seed.copy()
    height = float(page[:3, 2] @ (seed[:3, 3] - page[:3, 3]))
    lifted[:3, 3] += page[:3, 2] * max(0., motion['approach']['standoff_m'] - height)
    rise = lifted.copy()
    rise[2, 3] = max(lifted[2, 3], top)
    aligned = rise.copy()
    aligned[:3, :3] = dip.rotation
    over = dip.hover.copy()
    over[:3, 3] += dip.cap.axis * max(0., (rise[2, 3] - over[2, 3]) / dip.cap.axis[2])
    targets = [('page_lift', lifted, pplan.PHASE_LIFT), ('rise', rise, pplan.PHASE_LIFT),
               ('align', aligned, pplan.PHASE_TRAVEL), ('transit', over, pplan.PHASE_TRAVEL),
               ('hover', dip.hover, pplan.PHASE_TRAVEL)]
    traj = _travel(dip, targets, q_seed, kin, motion, station_top_m, check)
    traj.info.update(dip_phase='approach', return_tcp=lifted.tolist())
    return traj


def plan_axial(dip: Dip, phase: str, *, q_seed, kin, motion, check=lambda *a: None):
    """descend (hover to dwell), dwell (hold), retract (to the hover), each along the cap's axis from the measured
    pose, keeping its rotation."""
    seed = kin.fk(q_seed)
    target = seed.copy()
    height, _ = dip.local(seed)
    if phase == 'descend':
        target[:3, 3] += dip.cap.axis * (dip.dwell_height_m - height)
        feedback = pplan.PHASE_DESCEND
    elif phase == 'retract':
        target[:3, 3] += dip.cap.axis * max(0., dip.hover_height_m - height)
        feedback = pplan.PHASE_LIFT
    elif phase == 'dwell':
        if not -OVERSHOOT_M <= height - dip.dwell_height_m <= TRACK_M:
            raise ValueError(f"dwell: the tool's end is {(height - dip.dwell_height_m) * 1000:+.2f} mm from its dwell "
                             "height")
        feedback = pplan.PHASE_SETTLE
    else:
        raise ValueError('an axial phase is descend, dwell or retract')
    traj = _travel(dip, [(phase, target, feedback)], q_seed, kin, motion, -math.inf, check, axial=True)
    traj.info.update(dip_phase=phase)
    return traj


def plan_return(dip: Dip, *, q_seed, kin, motion, return_tcp, station_top_m, check):
    """From wherever the dip left the tool back to the page standoff: up the cap's axis when the tool is over the
    cap (else straight up), across at the safe height, turn to the page's attitude, down to return_tcp."""
    target, seed = _pose(return_tcp), kin.fk(q_seed)
    top = max(seed[2, 3], target[2, 3], safe_height(dip, station_top_m, motion))
    rise = seed.copy()
    _, off = dip.local(seed)
    if off < dip.cap.bore_radius_m + 0.002:
        stage = 'cap_exit'
        rise[:3, 3] += dip.cap.axis * ((top - seed[2, 3]) / dip.cap.axis[2])
    else:
        stage = 'escape'   # an interrupted crossing: straight up at the measured attitude
        rise[2, 3] = top
    over, aligned = rise.copy(), target.copy()
    over[:2, 3] = target[:2, 3]
    aligned[2, 3] = top
    traj = _travel(dip, [(stage, rise, pplan.PHASE_LIFT), ('transit', over, pplan.PHASE_TRAVEL),
                         ('over_page', aligned, pplan.PHASE_TRAVEL), ('page_return', target, pplan.PHASE_TRAVEL)],
                   q_seed, kin, motion, station_top_m, check)
    traj.info.update(dip_phase='return', return_tcp=target.tolist())
    return traj


def verify_return(pose, target, motion) -> None:
    """ValueError unless the measured TCP came back to the page standoff it left."""
    position = float(np.linalg.norm(np.asarray(pose)[:3, 3] - np.asarray(target)[:3, 3]))
    rotation = float(tl.rotation_angle(np.asarray(pose)[:3, :3].T @ np.asarray(target)[:3, :3]))
    if position > motion['dispatch_drift']['start_tolerance_m'] or rotation > motion['clik']['max_orientation_error_rad']:
        raise ValueError(f"the return ended {position * 1000:.2f} mm, {rotation:.4f} rad from the page standoff")
