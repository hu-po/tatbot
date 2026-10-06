"""The S1 search against a simulated station: the probe as cad/calibration-palette-v11 draws it (the ball on a
shaft and cone over a collar and the body) and a halo nose as a wall around a hole: the laser's, a 9.0 mm ring round
a 7.7 mm hole, or a ballpoint's, a metal tip under a body about 4 mm across, its bore narrower than the ball.
From a fix off across and in height it finds the ball, and on the way no touch lands on the stylus from above, meets
the body, the collar or an inkcap, or lets the ball under a hole it fits; a probe that never fires stops it early.

The worst case for the edge: a contact on the ball's upper cap with under 0.4 mm of it above the end plane does
not trip (the stylus is pushed down its axis, where the trigger force is highest), and the wall rides over it."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from tatbot_calib import halo, program
from tatbot_description import repo_root

BALL_R = 0.001
CAP_LATERAL, CAP_AXIAL = 0.0025, 0.0010
EDGE_TRIGGER = 0.0004    # a cap contact trips only with more of the ball than this over the end plane
CLEARANCE = 0.030


@dataclass(frozen=True)
class Nose:
    """The tool the simulated station meets: its end's outside radius, its hole, the wall's taper (radius gained per
    metre up) and how far up it reaches the stylus."""
    end: float
    hole: float
    taper: float = 0.0
    length: float = 0.025
    profile: tuple = ()     # ((h, r), ...) up the wall, where it is not one cone

    def radius(self, h: float) -> float:
        h = min(max(h, 0.0), self.length)
        if self.profile:
            heights, radii = zip((0.0, self.end), *self.profile, strict=True)
            return float(np.interp(h, heights, radii))
        return self.end + self.taper * h

    def corners(self) -> list:
        """The wall's cross-section from the hole out and up the nose."""
        return [(self.hole, 0.0), (self.end, 0.0), *((r, h) for h, r in self.profile if h < self.length),
                (self.radius(self.length), self.length), (self.hole, self.length)]


LASER = Nose(0.0090, 0.0077)
# the ballpoint as the palette camera saw it (2026-10-02): its metal tip, 0.45 mm across at the ball and 1.6 mm at
# the body's end 2.4 mm up, then the body
CARTRIDGE = Nose(0.000225, 0.0001, profile=((0.0024, 0.0008), (0.00241, 0.0019), (0.008, 0.00205), (0.025, 0.0024)))
GIVE = 0.0012        # the arm's travel past first contact before a side touch trips (program.SIDE_GIVE_M)


def datasheet_halo(tool_id: str) -> halo.Halo:
    """The halo the program plans with: the named tool's datasheet, as the tool gate reads it, whichever tool the
    checkout fits."""
    import tool_spec

    tool = tool_spec.load_tool(tool_id, repo_root(None))
    return halo.Halo.from_datasheet(tool.raw.get("calibration") or {}, tool.profile)


def stylus(ball):
    """(name, z_low, z_high, r_low, r_high, fixed) frusta under the ball: build.py's probe."""
    z = ball[2]
    return (("body", z - 0.2, z - 0.0256, 0.0175, 0.0175, True),
            ("collar", z - 0.0256, z - 0.0226, 0.005, 0.005, True),
            ("cone", z - 0.0226, z - 0.0106, 0.003, 0.0008, False),
            ("shaft", z - 0.0106, z, 0.0008, 0.0008, False))


def _to_segment(point, a, b) -> float:
    ab = b - a
    t = np.clip(((point - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * ab)))


def wall_gap(nose: Nose, d: float, z: float) -> float:
    """How far a point d from the tool's axis and z over its end plane lies from the wall (its cross-section from
    the hole to the outside, from the end up the nose), 0 inside it."""
    if 0.0 <= z <= nose.length and nose.hole <= d <= nose.radius(z):
        return 0.0
    corners = [np.array(c) for c in nose.corners()]
    point = np.array([d, z])
    return min(_to_segment(point, a, b) for a, b in zip(corners, corners[1:] + corners[:1], strict=True))


class Station:
    """What the nose, its end plane's centre at p (pointing down), meets: None, or (part, trips)."""

    def __init__(self, ball, inkcaps=(), ghost=False, nose: Nose = LASER):
        self.ball = np.asarray(ball, float)
        self.inkcaps = inkcaps          # (centre xy, radius, rim z) in the base
        self.ghost = ghost              # a stylus nothing meets: a probe that never fires, whatever pushes it
        self.nose = nose
        self.events = []

    def meets(self, p):
        if self.ghost:
            return None
        nose = self.nose
        d = float(np.hypot(*(p[:2] - self.ball[:2])))
        z = float(self.ball[2] - p[2])                         # the ball's centre over the end plane
        if wall_gap(nose, d, z) < BALL_R and z < nose.length:
            if d < nose.hole and z > 0.0 and nose.hole > BALL_R:
                return "hole", False
            over = BALL_R - max(0.0, -z)                       # how much of the ball stands over the end plane
            return "ball", over > EDGE_TRIGGER or d <= nose.end
        for name, lo, hi, r_lo, r_hi, fixed in stylus(self.ball):
            bottom, top = max(lo, p[2]), min(hi, p[2] + nose.length)
            if top < bottom:
                continue
            reach = max(nose.radius(h - p[2]) + r_lo + (r_hi - r_lo) * (h - lo) / (hi - lo) for h in (bottom, top))
            if d <= reach:
                return name, not fixed
        for centre, radius, rim in self.inkcaps:
            if np.hypot(*(p[:2] - centre)) <= nose.radius(rim - p[2]) + radius and p[2] < rim:
                return "inkcap", False
        return None


class FakeRig:
    def __init__(self, station, fires=True):
        self.station, self.fires = station, fires
        self.here = station.ball + [0.0, 0.0, 0.060]
        self.goals = []
        self.safety = SimpleNamespace(latched=False, estop_ok=True)

    def spin(self, seconds):
        pass

    def joints(self):
        return np.concatenate([self.here, np.zeros(4)])

    def _walk(self, a, b, travel_phase):
        """Step the end plane from a to b; the first thing met, or None. Ball under a wall that rides over it
        (an edge contact that did not trip) is recorded and walked on."""
        n = max(1, int(np.linalg.norm(b - a) / 0.0001))
        for t in np.linspace(0.0, 1.0, n + 1):
            p = a + t * (b - a)
            met = self.station.meets(p)
            if met is None:
                continue
            part, trips = met
            if part == "ball" and not trips:
                self.station.events.append(("rode over", p.copy()))
                continue
            if not travel_phase or part in ("hole", "body", "collar", "inkcap"):
                self.station.events.append((f"{part} while {'touching' if travel_phase else 'travelling'}", p.copy()))
            return p, trips and self.fires
        return None

    def goal(self, start, *, move=False, direction=(0.0, 0.0, -1.0), prior=0.0):
        u = np.asarray(direction, float)
        s = start[:3, 3]
        self.goals.append((s.copy(), u.copy(), prior))
        top = max(self.here[2], s[2] + CLEARANCE)
        for a, b in ((self.here, np.array([*self.here[:2], top])), (np.array([*self.here[:2], top]),
                                                                     np.array([*s[:2], top])),
                     (np.array([*s[:2], top]), s)):
            if self._walk(a, b, travel_phase=False) is not None:
                self.here = s
                return {"tripped": False, "joints": [], "run_dir": "", "message": "error: met on the way",
                        "contact": [s.tolist()]}
        cap = CAP_AXIAL if abs(u[2]) > 0.5 else CAP_LATERAL
        end = s + (prior + cap) * u
        met = self._walk(s, end, travel_phase=True)
        if met is not None and met[1]:
            at = met[0] + (GIVE if abs(u[2]) < 0.5 else 0.0) * u   # where the encoders put the trip
            self.here = at - 0.003 * u                      # the executor's back-off, the way it came
            return {"tripped": True, "joints": np.concatenate([at, np.zeros(4)]).tolist(), "run_dir": "",
                    "message": "touched", "contact": [at.tolist()]}
        self.here = end if met is None else met[0]
        return {"tripped": False, "joints": [], "run_dir": "", "message": "no trip", "contact": [self.here.tolist()]}


def calibration(tmp_path, rig, prior, base_from_palette=None, inkcaps=(), tool="picosecond-laser-pen"):
    from tatbot_description import repo_root
    (tmp_path / "urdf").mkdir(exist_ok=True)
    (tmp_path / "urdf" / "palette.urdf").write_text((repo_root(None) / "urdf" / "palette.urdf").read_text())
    kin = SimpleNamespace(fk=lambda q: np.block([[np.eye(3), np.asarray(q[:3]).reshape(3, 1)],
                                                 [np.zeros((1, 3)), 1]]))
    cal = program.Calibration(rig, SimpleNamespace(dir=tmp_path), tmp_path, "left", datasheet_halo(tool), 30.0, kin)
    cal.ball = np.asarray(prior, float)
    cal.floor_z = float(prior[2]) + BALL_R - program.FLOOR_UNDER_TOP_M + program.FIX_HEIGHT_M   # as _stages sets it
    if base_from_palette is not None:
        cal.station_pose, cal.ball_in_palette, cal.inkcaps = base_from_palette, np.array([0.0, 0.0, 0.0556]), inkcaps
    return cal


TRUTH = np.array([0.2310, -0.2660, 0.1140])


def under_the_floor(rig, truth=TRUTH) -> list:
    """The goals whose way (the start, and its travel to the cap) takes the tool's end more than
    FLOOR_UNDER_TOP_M under the true ball's top: the operator's floor (2026-09-29)."""
    floor = truth[2] + BALL_R - program.FLOOR_UNDER_TOP_M
    out = []
    for start, u, prior in rig.goals:
        end = start + (prior + (CAP_AXIAL if abs(u[2]) > 0.5 else CAP_LATERAL)) * u
        if min(start[2], end[2]) < floor - 1e-9:
            out.append((start, u, prior))
    return out


def palette_at(ball):
    out = np.eye(4)
    out[:3, 3] = ball - [0.0, 0.0, 0.0556]
    return out


@pytest.mark.parametrize("dz", [-0.002, -0.001, 0.0, 0.001, 0.002])
@pytest.mark.parametrize("across", [(0.0, 0.0), (-0.010, 0.003), (0.0085, 0.0085), (0.0, -0.012), (-0.006, -0.010)])
def test_s1_finds_the_ball_from_a_fix_off_across_and_in_height(tmp_path, dz, across):
    """A fix up to 12 mm off across and FIX_HEIGHT_M in height: the search finds the stylus, and no goal takes the
    tool more than 10 mm under the true ball's top."""
    station = Station(TRUTH)
    rig = FakeRig(station)
    prior = TRUTH + [*across, dz]
    cal = calibration(tmp_path, rig, prior, palette_at(prior))
    assert cal.s1(), station.events
    assert np.linalg.norm(cal.ball[:2] - TRUTH[:2]) < 2e-4
    assert abs(cal.ball[2] - TRUTH[2]) < 8e-4
    bad = [event for event in station.events if event[0] != "rode over"]
    assert not bad, bad                                   # nothing from above, no body, collar or hole
    assert not under_the_floor(rig)
    # the floor from S1's own top: never deeper than the operator's 10 mm under the true top (round 8, 2026-09-29: the
    # laser's passes read the top short and the floor stood 11 mm under it), and within 1.5 mm of it
    true_floor = TRUTH[2] + BALL_R - program.FLOOR_UNDER_TOP_M
    assert true_floor - 1e-9 <= cal.floor_z <= true_floor + 0.0015, (cal.floor_z, true_floor)


def test_nothing_is_sent_under_the_floor(tmp_path):
    """A touch whose way would take the tool under the floor is refused before it is sent, whatever planned it."""
    station = Station(TRUTH)
    rig = FakeRig(station)
    cal = calibration(tmp_path, rig, TRUTH, palette_at(TRUTH))
    rotation = halo.tool_pose(np.zeros(3), cal.heading)[:3, :3]
    heading = cal.heading_vector()
    deep = halo.side_touches(TRUTH - [0.0, 0.0, 0.009], rotation, heading, cal.halo, cal.side_radius)[0]
    with pytest.raises(RuntimeError, match="under the floor"):
        cal.touch(deep, "S1", (0.0,))
    assert rig.goals == []


def test_further_yaws_start_from_s1s_measured_ball_and_stay_clear(tmp_path):
    station = Station(TRUTH)
    rig = FakeRig(station)
    prior = TRUTH + [-0.006, 0.004, 0.004]
    cal = calibration(tmp_path, rig, prior, palette_at(prior))
    assert cal.s1()
    measured = cal.ball.copy()
    for yaw in (-30.0, 30.0, 60.0):
        cal.ball = measured.copy()
        assert cal.attitude(yaw, f"S4 {yaw:+.0f}/+0"), station.events
        assert np.linalg.norm(cal.ball[:2] - TRUTH[:2]) < 2e-4
    assert {c.group for c in cal.contacts} == {"S1", "S4 -30/+0", "S4 +30/+0", "S4 +60/+0"}
    assert all(c.kind == "side" for c in cal.contacts)
    assert not [event for event in station.events if event[0] != "rode over"]


def test_a_ball_ridden_over_never_reaches_the_hole(tmp_path):
    # the fix 3.5 mm high: the search's end plane stands right at the ball's equator, the worst height
    station = Station(TRUTH)
    rig = FakeRig(station)
    prior = TRUTH + [0.004, -0.003, 0.0035]
    cal = calibration(tmp_path, rig, prior, palette_at(prior))
    cal.s1()
    assert not [event for event in station.events if event[0].startswith("hole")]


def test_a_probe_that_never_fires_stops_the_search_without_retries(tmp_path):
    station = Station(TRUTH, ghost=True)
    rig = FakeRig(station, fires=False)
    cal = calibration(tmp_path, rig, TRUTH, palette_at(TRUTH))
    assert not cal.s1()
    passes, backs = rig.goals[0::2], rig.goals[1::2]      # the laser's ring backs out of every pass that met nothing
    priors = [prior for _, _, prior in passes]
    assert priors and max(priors) == min(priors) == pytest.approx(cal.search_standoff())
    assert len(passes) == 4                               # each pair once: from one estimate a pair repeats itself
    assert len(backs) == 4 and all(np.allclose(b[1], -p[1]) for p, b in zip(passes, backs, strict=True))


def test_the_search_starts_clear_of_the_inkcaps_and_the_camera(tmp_path):
    from pathlib import Path

    from tatbot_calib import station as station_mod

    repo = Path(__file__).resolve().parents[3]
    inkcaps = station_mod.inkcap_rims(repo / "urdf" / "palette.urdf", repo / "config" / "palette.yaml")
    assert len(inkcaps) == 6 and all(abs(rim[2] - 0.032) < 1e-6 for _, rim, _ in inkcaps)
    base_from_palette = palette_at(TRUTH)
    caps = [((base_from_palette @ np.append(rim, 1.0))[:2], radius, (base_from_palette @ np.append(rim, 1.0))[2])
            for _, rim, radius in inkcaps]
    world = Station(TRUTH, caps)
    rig = FakeRig(world)
    prior = TRUTH + [-0.006, 0.004, 0.004]               # high: the search goes lowest over the caps
    cal = calibration(tmp_path, rig, prior, palette_at(prior), inkcaps)
    assert cal.s1(), world.events
    assert not [event for event in world.events if event[0] != "rode over"]


def run_s1_cartridge(tmp_path, across, dz, inkcaps=()):
    station = Station(TRUTH, inkcaps, nose=CARTRIDGE)
    rig = FakeRig(station)
    prior = TRUTH + [*across, dz]
    cal = calibration(tmp_path, rig, prior, palette_at(prior), tool="lutin-ballpoint-dot")
    return cal, station, cal.s1()


@pytest.mark.parametrize("dz", [-0.002, 0.0, 0.002])
@pytest.mark.parametrize("across", [(0.0, 0.0), (0.005, 0.0), (0.006, 0.006), (-0.008, 0.002), (0.0, -0.0115)])
def test_s1_finds_the_ball_with_a_cartridge_tube_as_the_halo(tmp_path, dz, across):
    """The ballpoint's tube, 2 mm across: its reach across a search line is a few millimetres, so lines offset
    across find a fix up to 12 mm off; its bore is narrower than the ball, which never enters it."""
    cal, station, found = run_s1_cartridge(tmp_path, across, dz)
    assert found, station.events
    # its metal tip makes the edge passes read the top up to 0.6 mm low; the side pairs stand 7 mm under it
    assert np.linalg.norm(cal.ball[:2] - TRUTH[:2]) < 1e-4 and abs(cal.ball[2] - TRUTH[2]) < 7e-4
    assert not [event for event in station.events if event[0] != "rode over"]
    assert not under_the_floor(cal.rig)
    assert 0.0002 < cal.side_radius < 0.0005          # the tip's 0.45 mm and the ball, less the give, measured


def test_a_cartridge_searches_offset_lines_only_when_its_own_line_meets_nothing(tmp_path):
    near, far = tmp_path / "near", tmp_path / "far"
    near.mkdir(), far.mkdir()
    assert run_s1_cartridge(near, (0.0, 0.0), 0.0)[2] and "across it" not in (near / "events.jsonl").read_text()
    assert run_s1_cartridge(far, (0.0, 0.0115), 0.0)[2] and "across it" in (far / "events.jsonl").read_text()
    laser = datasheet_halo("picosecond-laser-pen")      # the laser's reach covers the prior's 12 mm from one line
    assert program.SEARCH_LINE_SPACING * (laser.wall_radius_m + halo.BALL_RADIUS_M) > program.SEARCH_ACROSS_M


def _rotvec(r: np.ndarray) -> np.ndarray:
    """The rotation vector of r, through its quaternion: the upright tool is a half turn, where the axis-angle
    formula divides by zero."""
    w = math.sqrt(max(0.0, 1.0 + np.trace(r))) / 2.0
    xyz = np.sqrt(np.maximum(0.0, 1.0 + 2.0 * np.diag(r) - np.trace(r))) / 2.0
    xyz *= np.sign([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]] + (w < 1e-6) * np.array(
        [1.0, np.sign(r[0, 1] + r[1, 0]) or 1.0, np.sign(r[0, 2] + r[2, 0]) or 1.0]))
    angle = 2.0 * math.atan2(float(np.linalg.norm(xyz)), w)
    return np.zeros(3) if angle < 1e-12 else angle * xyz / np.linalg.norm(xyz)


def _from_rotvec(v) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * kx @ kx


class PoseKin:
    """A fake arm whose joints are its tool mount's pose (xyz, rotation vector, carriage); the tcp is tip0 in it."""

    def __init__(self, tip0):
        self.tip0 = np.asarray(tip0, float)

    def frame(self, q, name=""):
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = _from_rotvec(np.asarray(q, float)[3:6]), np.asarray(q, float)[:3]
        return out

    def fk(self, q):
        out = self.frame(q)
        out[:3, 3] = out[:3, 3] + out[:3, :3] @ self.tip0
        return out


class TipRig(FakeRig):
    """FakeRig on PoseKin's joints, whose tube's true end sits at true_tip in the mount where the program plans with
    tip0: every commanded pose puts the true end off by the attitude's rotation of their difference."""

    def __init__(self, station, tip0, true_tip):
        super().__init__(station)
        self.tip0, self.true_tip = np.asarray(tip0, float), np.asarray(true_tip, float)
        self.rotation = halo.tool_pose(np.zeros(3), 0.0)[:3, :3]

    def _mount(self, end: np.ndarray, rotation: np.ndarray) -> list:
        return [*(end - rotation @ self.true_tip), *_rotvec(rotation), 0.0]

    def joints(self):
        return np.array(self._mount(self.here, self.rotation))

    def goal(self, start, *, move=False, direction=(0.0, 0.0, -1.0), prior=0.0, guard=None):
        rotation, offset = start[:3, :3], start[:3, :3] @ (self.true_tip - self.tip0)
        shifted = start.copy()
        shifted[:3, 3] += offset
        self.rotation = rotation
        result = super().goal(shifted, move=move, direction=direction, prior=prior)
        if result["tripped"]:
            end = np.array(result["joints"][:3])
            result.update(joints=self._mount(end, rotation), contact=[(end - offset).tolist()])
        return result


def test_a_whole_cartridge_run_recovers_the_tube_end_across_its_axis(tmp_path, monkeypatch):
    """S1 (search, top, sides), two further yaws (the second placed by the fit of the first two), the fit, end to end
    on the simulated station with the ballpoint's tube 0.8/-1.2/0.6 mm off the tcp the stack plans with. The fit
    finds the tube's end across the axis to 0.1 mm; along it, the installed length stays."""
    import json

    from tatbot_calib import station as station_mod

    tip0 = np.array([0.00002, -0.002674, 0.074495])
    true_tip = tip0 + [0.0008, -0.0012, 0.0006]
    world = Station(TRUTH, nose=CARTRIDGE)
    rig = TipRig(world, tip0, true_tip)
    prior = TRUTH + [0.003, -0.002, 0.001]                 # within FIX_HEIGHT_M with the tip's 0.6 mm
    cal = program.Calibration(rig, SimpleNamespace(dir=tmp_path), repo_root(None), "right",
                              datasheet_halo("lutin-ballpoint-dot"), 0.0, PoseKin(tip0))
    cal.tip0, cal.mount_in_tcp = tip0, np.eye(3)
    monkeypatch.setattr(program, "_preflight", lambda cal: None)
    monkeypatch.setattr(program, "_latched", lambda cal: False)
    fix = station_mod.StationFix("right", palette_at(prior), prior, "2026-09-29T18:00:00Z", "overhead_tag")
    args = SimpleNamespace(attitudes=[(-30.0, 0.0), (30.0, 0.0)], tool="lutin-ballpoint-dot", station="fix.json")
    assert program._stages(cal, fix, args) == 0
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    tip = np.array(candidate["fit"]["tip_m"])
    assert np.linalg.norm(tip[:2] - true_tip[:2]) < 1e-4                     # across the axis: 0.1 mm
    assert tip[2] == pytest.approx(tip0[2], abs=1e-6)                          # along it: held
    assert not [event for event in world.events if event[0] != "rode over"]
    assert max(candidate["held_out_rms_m"].values()) < 1e-4
    assert max(candidate["tip_moved_m"].values()) < 2e-4
    assert "placed ball met nothing" not in (tmp_path / "events.jsonl").read_text()
    assert "S4 +30/+0 search" not in (tmp_path / "touches.jsonl").read_text()   # the fit placed the third yaw
    touch = station_mod.StationTouch.from_dict(json.loads((tmp_path / "station-touch.json").read_text()))
    assert np.linalg.norm(touch.ball - TRUTH) < 2e-3 and touch.tool_id == "lutin-ballpoint-dot"   # what dips aim by
    np.testing.assert_allclose(touch.tcp_m, tip0)


def test_a_fit_from_one_attitude_leaves_the_side_radius_s1_measured(tmp_path, monkeypatch):
    """2026-09-30, pink at v11: after S1 alone the fit put the tip 12.8 mm off, and its side radius so fattened the
    tool that a touch straight down over the ball read as meeting the palette camera. S1's half-span stands until
    two attitudes are in."""
    from tatbot_calib import solve

    cal = program.Calibration(FakeRig(Station(TRUTH)), SimpleNamespace(dir=tmp_path), repo_root(None), "right",
                              datasheet_halo("lutin-ballpoint-dot"), 0.0, PoseKin(np.zeros(3)))
    cal.s1_side_radius = cal.side_radius = 0.0023
    cal.tip0 = np.zeros(3)
    fitted = SimpleNamespace(side_radius=0.0412, rms_m={"side": 0.0009}, tip=np.zeros(3))
    monkeypatch.setattr(solve, "fit", lambda *a, **k: fitted)
    cal.contacts = [solve.Contact("side", np.zeros(7), "S1") for _ in range(4)]
    program._fit(cal)
    assert cal.side_radius == 0.0023
    cal.contacts.append(solve.Contact("side", np.zeros(7), "-30/+0"))
    program._fit(cal)
    assert cal.side_radius == 0.0412


def test_s4_goes_on_past_refused_attitudes_until_it_has_the_ones_it_wants(monkeypatch):
    """Round 7 (2026-09-29): of the three attitudes kept by reach from rest, two were refused on the way from S1
    (joint 5's limit), and two attitude groups identify no tip. S4 now tries its candidates in turn until
    ATTITUDES_WANTED have measured."""
    tried = []

    def attitude(yaw, group):
        tried.append(yaw)
        cal.contacts.append(group)        # a touch or two before it is skipped
        if len(tried) in (1, 3):
            raise RuntimeError("+p: no plan to it (joint right/joint_5 reaches its guarded limit); not sent")
        return len(tried) != 4            # the fourth meets nothing

    cal = SimpleNamespace(attitude=attitude, say=lambda text: None, ball=None, stuck=False, contacts=["S1"])
    monkeypatch.setattr(program, "_latched", lambda cal: False)
    monkeypatch.setattr(program, "_fit", lambda cal: None)
    candidates = program._attitudes(program.DEFAULT_ATTITUDES)
    assert program.s4(cal, candidates, TRUTH) == program.ATTITUDES_WANTED
    assert len(tried) == program.ATTITUDES_WANTED + 3 and tried[0] == -60.0
    # a skipped yaw's contacts are not fitted (071f, 2026-10-01: a partial attitude counted as a held-out group)
    assert cal.contacts == ["S1", *(f"S4 {yaw:+.0f}/+0" for n, yaw in enumerate(tried, 1) if n not in (1, 3, 4))]


def test_a_pair_that_met_nothing_is_not_run_again_from_the_same_estimate():
    """2026-09-30, round 13: S1's search ran its four passes three times over from one estimate, 3.6 minutes, before
    trying another line. A pair runs again only once a trip has moved the estimate."""
    from tatbot_calib import program

    calls = []

    def pair(rotation, group, heading, name, search=False):
        calls.append(name)
        if name == "d" and len(calls) == 2:   # one pass of d trips and moves the estimate; p then runs again
            fake.ball = fake.ball + [0.001, 0.0, 0.0]
        return None

    fake = SimpleNamespace(ball=TRUTH.copy(), pair=pair)
    assert program.Calibration.sides(fake, np.eye(3), "S1 search", np.array([1.0, 0.0, 0.0]), search=True) is None
    assert calls == ["p", "d", "p", "d"]      # both again from the moved estimate, then neither
    calls.clear()
    fake.pair = lambda *a, **k: calls.append(a[3])
    assert program.Calibration.sides(fake, np.eye(3), "S1 search", np.array([1.0, 0.0, 0.0])) is None
    assert calls == ["p", "d"]


def test_a_search_pass_refused_on_its_way_is_skipped_and_its_opposite_sent(tmp_path, monkeypatch):
    """2026-09-30, round 13: the -4.2 mm line's +d pass would have brought the wrist within 14 mm of the camera's
    post, and the refusal ended the run with its -d pass, from the post's far side, unsent. A search pass the checks
    refuse is skipped as a miss; outside the search a refusal still stops."""
    rig = FakeRig(Station(TRUTH))
    cal = calibration(tmp_path, rig, TRUTH, palette_at(TRUTH))
    said, sent = [], []
    cal.say = said.append

    def touch(plan, group, retries):
        if plan.label == "+d":
            raise RuntimeError("+d: the wrist would pass 14 mm from the overhead camera's post; not sent")
        sent.append(plan.label)
        return None

    cal.touch = touch
    monkeypatch.setattr(program, "_latched", lambda cal: False)
    rotation, heading = np.eye(3), np.array([1.0, 0.0, 0.0])
    assert cal.pair(rotation, "S1 search", heading, "d", search=True) is None
    assert sent == ["-d"] and any("+d: not sent" in text for text in said)
    with pytest.raises(RuntimeError, match="not sent"):
        cal.pair(rotation, "S1", heading, "d")


def test_an_s4_attitude_searches_before_its_sides_and_counts_only_its_own_trips(tmp_path):
    """2026-09-30, round 15: S4's sides from S1's ball met nothing at +30, +45 and +90 degrees of yaw. Before the fit
    can place the ball, an S4 yaw searches first, as S1 does, and a trip from before the search does not end it."""
    rig = FakeRig(Station(TRUTH))
    cal = calibration(tmp_path, rig, TRUTH, palette_at(TRUTH), tool="lutin-ballpoint-dot")
    cal.proven, cal.trips = True, 5
    calls = []

    def sides(rotation, group, heading, search=False):
        calls.append((group, search))
        return None

    def pair(rotation, group, heading, name, search=False):
        calls.append((group, "line"))
        if len([c for c in calls if c[1] == "line"]) == 2:
            cal.trips += 1                       # the second offset line trips
        return None

    cal.sides, cal.pair, cal.say = sides, pair, lambda text: None
    assert not cal.attitude(30.0, "S4 +30/+0")
    assert calls[0] == ("S4 +30/+0 search", True) and calls[1:3] == [("S4 +30/+0 search", "line")] * 2
    assert calls[3] == ("S4 +30/+0 search", True)        # the search's sides from the line that tripped
    assert calls[4:] == [("S4 +30/+0", False)]           # then the yaw's own, whatever the search found
    calls.clear()
    assert not cal.attitude(0.0, "S1")
    assert calls == [("S1", False)]                                     # S1 searched in s1()


def test_a_rings_pass_that_met_nothing_backs_out_along_its_line_before_anything_lifts(tmp_path):
    """2026-09-29, round 10: the laser's ring ended a side pass over the stylus, pressing it untripped, and the next
    way's lift dragged it. After a ring's pass meets nothing, the arm backs straight out along the pass first."""
    rig = FakeRig(Station(TRUTH + [0.030, 0.0, 0.0]))       # the stylus far off: every pass meets nothing
    cal = calibration(tmp_path, rig, TRUTH, palette_at(TRUTH))
    cal.floor_z = TRUTH[2] + BALL_R - program.FLOOR_UNDER_TOP_M   # S1's floor, under the measured top
    rotation = halo.tool_pose(np.zeros(3), 0.0)[:3, :3]
    heading = np.array([1.0, 0.0, 0.0])
    plan = cal.clear(halo.side_touches(cal.ball, rotation, heading, cal.halo, cal.side_radius))
    assert cal.touch(plan, "S1", (0.0,)) is None
    (start, u, _), (back_start, back_u, back_prior) = rig.goals
    assert np.allclose(back_u, -u) and back_prior > 0.0
    assert np.linalg.norm(back_start - (start + (plan.prior_m + cal.lateral_cap_m) * u)) < 1e-3   # from where it ended
    assert abs(rig.here[2] - start[2]) < 1e-9                  # no lift: back along the line, at the pass's height


def test_a_back_out_that_trips_leaves_the_arm_where_it_is(tmp_path):
    rig = FakeRig(Station(TRUTH + [0.030, 0.0, 0.0]))
    cal = calibration(tmp_path, rig, TRUTH, palette_at(TRUTH))
    cal.floor_z = TRUTH[2] + BALL_R - program.FLOOR_UNDER_TOP_M
    rotation = halo.tool_pose(np.zeros(3), 0.0)[:3, :3]
    plan = cal.clear(halo.side_touches(cal.ball, rotation, np.array([1.0, 0.0, 0.0]), cal.halo, cal.side_radius))
    passes = []

    def goal(start, *, move=False, direction=(0.0, 0.0, -1.0), prior=0.0, guard=None):
        passes.append(prior)
        tripped = len(passes) == 2                               # the back-out trips
        return {"tripped": tripped, "joints": [0.0] * 7, "run_dir": "", "message": "touched" if tripped else "no trip",
                "contact": [np.asarray(start)[:3, 3].tolist()]}

    rig.goal = goal
    with pytest.raises(RuntimeError, match="stays where it is"):
        cal.touch(plan, "S1", (0.0,))
    assert cal.stuck and len(passes) == 2


def test_the_palette_zone_takes_every_part_and_stands_on_the_tables_normal(tmp_path, monkeypatch):
    """The lease's zone (tatbot_session.lease): a cylinder about the palette's root along the arm base's z, wide and
    tall enough for every part, the e-stop enclosure 144 mm out and the roof tag's 116 mm top among them."""
    import json as json_

    from tatbot_calib import cli, station

    registration = tmp_path / "reg.json"
    world_from_base = np.eye(4)
    world_from_base[:3, 3] = [0.1, -0.2, 0.7]
    registration.write_text(json_.dumps({"world_from_arm_base": world_from_base.tolist()}))
    monkeypatch.setattr(cli, "_stack", lambda repo: {"registration": {"left": str(registration)}})
    base_from_palette = np.eye(4)
    base_from_palette[:3, 3] = [0.30, -0.27, 0.02]
    fix = station.StationFix("left", base_from_palette, np.zeros(3), "2026-09-30T13:00:00Z", "overhead_tag")
    zone = program.palette_zone(repo_root(None), "left", fix, "run-1")
    assert zone["arm"] == "left" and zone["run"] == "run-1"
    assert np.allclose(np.asarray(zone["world_from_zone"])[:3, 3], [0.40, -0.47, 0.72])
    assert 0.144 + program.ZONE_MARGIN_M - 1e-9 <= zone["radius_m"] < 0.25
    assert 0.0776 + program.ZONE_MARGIN_M - 1e-6 <= zone["height_m"] < 0.2


def test_before_s1_places_the_ball_the_station_stands_where_the_fix_puts_it(tmp_path):
    """2026-09-30, v11: S1 plans its way in before it has a ball; anchored to none, every part stood at NaN, every
    body read -inf mm from it, and no way in or staging pose was planned."""
    from tatbot_motion.collision import Gap

    rig = FakeRig(Station(TRUTH))
    cal = calibration(tmp_path, rig, TRUTH, palette_at(TRUTH))
    cal.world_from_base = np.eye(4)
    cal.ball = None
    seen = []

    def parts_clearance(arm, rows, parts, pads=None, max_step_rad=0.01):
        seen.append(parts)
        return Gap(0.05, "right/link_6", parts[0][0], 0, 0.05)

    cal.guard = SimpleNamespace(scene=SimpleNamespace(parts_clearance=parts_clearance), step_rad=0.01)
    assert cal.station_clear(np.zeros(7))
    assert all(np.all(np.isfinite(part[2])) for part in seen[-1])
    ball = (palette_at(TRUTH) @ np.append(cal.ball_in_palette, 1.0))[:3]
    probe = next(part for part in seen[-1] if part[0] == "probe_body")
    assert np.allclose(probe[2][:2], ball[:2], atol=1e-9)


def test_a_way_that_brings_a_body_near_the_station_is_refused_and_a_back_out_may_stand_where_it_is(tmp_path):
    """2026-09-30: the blue gripper's pad hung over the roof tag's box, which only the tool was checked against;
    and round 17's back-out, retracing its own pass, was refused 0.1 mm under the floor for the arm's sag."""
    from tatbot_motion.collision import Gap

    rig = FakeRig(Station(TRUTH))
    cal = calibration(tmp_path, rig, TRUTH, palette_at(TRUTH))
    cal.world_from_base = np.eye(4)
    seen = []

    def parts_clearance(arm, rows, parts, pads=None, max_step_rad=0.01):
        seen.append((arm, [p[0] for p in parts], pads))
        return Gap(0.004, "left/carriage_right_0", "palette_tag", 0, 0.034)

    cal.guard = SimpleNamespace(scene=SimpleNamespace(parts_clearance=parts_clearance), step_rad=0.01)
    with pytest.raises(RuntimeError, match="carriage_right_0 would pass 4 mm from the station's palette_tag"):
        cal.palette_check("+p", np.zeros((3, 7)))
    arm, names, pads = seen[-1]
    assert {"palette_tag", "palette_estop", "probe_body"} <= set(names) and pads == program.PALETTE_PADS
    assert not cal.station_clear(np.zeros(7))
    cal.floor_z = 0.100
    with pytest.raises(RuntimeError, match="under the floor"):
        cal.floor_check("x", [None, np.array([0.0, 0.0, 0.0999])], np.array([0.0, 0.0, -1.0]))
    cal.floor_check("x back", [None, np.array([0.0, 0.0, 0.0999])], np.array([0.0, 0.0, -1.0]), stands_at=0.0999)


def test_the_way_in_takes_a_staging_pose_only_when_the_direct_way_is_refused(tmp_path):
    """2026-09-30, v11: the joint move from rest swept pink's wrist under the guard's margin from the landed blue
    wrist, while a stop between them cleared both legs. With neither planned, nothing moves."""
    sent = []
    rig = SimpleNamespace(goal=lambda pose, move=False: sent.append((pose, move)) or {"message": "holding"},
                          spin=lambda s: None, joints=lambda: np.zeros(7))
    here = np.eye(4)
    here[:3, 3] = [0.30, 0.0, 0.17]
    kin = SimpleNamespace(fk=lambda q: here, frame=lambda q, name: np.eye(4))
    cal = program.Calibration(rig, SimpleNamespace(dir=tmp_path), repo_root(None), "right",
                              datasheet_halo("lutin-ballpoint-dot"), 0.0, kin)
    prior = np.array([0.16, 0.37, 0.083])
    cal.way_check = lambda label, poses, q=None: np.zeros(7)
    cal.approach(prior)
    assert sent == []                                           # the direct way is planned: nothing extra
    checked = []

    def way_check(label, poses, q=None):
        checked.append(label)
        if "over the ball" in label and len(checked) < 6:
            raise RuntimeError(f"{label}: the right arm's way comes 116 mm from the other arm")
        return np.zeros(7)

    cal.way_check = way_check
    cal.approach(prior)
    assert len(sent) == 1 and sent[0][1] is True
    assert np.allclose(sent[0][0][:3, 3], [0.23, 0.185, 0.17]) and np.allclose(sent[0][0][:3, 2], [0.0, 0.0, -1.0])
    cal.way_check = lambda label, poses, q=None: (_ for _ in ()).throw(RuntimeError("refused"))
    with pytest.raises(RuntimeError, match="no way over the ball"):
        cal.approach(prior)
    assert len(sent) == 1
