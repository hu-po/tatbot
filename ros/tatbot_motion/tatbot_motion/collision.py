"""How close the two arms come: both arms and their tools as convex bodies in one frame, and the clearance of a
planned motion against the other arm.

On 2026-09-29, at a view the pink arm planned next to the landed blue arm, the pink wrist's cables (the pen wand's
RCA power lead, the wrist D405's USB) caught on the blue arm. Nothing in the stack
modelled the other arm. Here both arms hang in the overhead camera's world by their registrations (tatbot_description
robot_description), each link is the convex hull of its URDF collision mesh, and each arm's fitted tool is a chain of
capsules along its datasheet profile (config/tools; the laser prop's cradle carries visuals only). Distances come from
pinocchio's geometry model on coal, the library the stack's kinematics already load: about 0.1 ms for every pair of
bodies on different arms. A hull encloses its mesh and a capsule its profile, so a clearance is never overstated.

The cables cannot be modelled: tied along each forearm, they run to the end effector in loops that let the wrist
turn, and the loops swing. So each arm's wrist and end effector are bubbled out by the loops' reach (motion.yaml
`collision.wrist_m`). A pair of bodies keeps the larger of its two bubbles, plus the margin: a loop must not reach a
rigid part of the other arm, while two loops that brush each other catch nothing, so two wrists keep one reach apart,
not two. Pure numpy on pinocchio and coal; no ROS import.
"""
from __future__ import annotations

import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Gap:
    """The tightest pair of bodies on different arms: the distance left between them once the larger of their two
    bubbles is taken off (m; negative inside), which two, where along a path (its row) if a path was checked, and the
    bodies' own distance."""
    distance_m: float
    body_a: str
    body_b: str
    row: int | None = None
    raw_m: float = float("nan")


def _arm(name: str) -> str:
    return name.split("/", 1)[0]


def _capsules(profile, tip_z_m: float | None = None) -> list[tuple[float, float, float]]:
    """(centre z, half length, radius) along the tool axis for a datasheet profile ((z, r), ..., the tip last; z from
    the tool mount's bore face toward the tip): one capsule per segment, as wide as its wider end."""
    points = sorted((float(z), float(r)) for z, r in profile)
    if tip_z_m is not None:
        points = [(z, r) for z, r in points if z <= tip_z_m + 1e-9]
    return [((z0 + z1) / 2.0, (z1 - z0) / 2.0, max(r0, r1)) for (z0, r0), (z1, r1) in zip(points, points[1:], strict=False)
            if z1 - z0 > 1e-4]


def _hull(points: np.ndarray):
    """coal's convex polyhedron over `points`, by scipy (the ROS build of coal has no qhull to build it itself)."""
    import coal
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    index = {int(k): i for i, k in enumerate(hull.vertices)}
    vertices, triangles = coal.StdVec_Vec3s(), coal.StdVec_Triangle()
    for k in hull.vertices:
        vertices.append(points[k])
    for a, b, c in hull.simplices:
        triangles.append(coal.Triangle(index[int(a)], index[int(b)], index[int(c)]))
    return coal.Convex(vertices, triangles)


# Shared station body policy: clearance and the existing conservative carriage pad envelope.
PALETTE_BODY_MARGIN_M = 0.010
PALETTE_PADS = {"carriage": 0.030}


class Scene:
    """Both arms' bodies in one model, from URDF text with both arms placed. `tools` maps an arm to its fitted tool's
    profile ((z, r), ..., the tip last, in <arm>/tool_mount along its z). Joints are the seven controlled ones in
    controller order (tatbot_description.names.joint_names)."""

    def __init__(self, urdf_xml: str, tools: dict | None = None, padding: dict | None = None):
        """`padding` maps a body's name after its arm (a prefix: "link_6", "tool", "realsense") to its bubble, m."""
        import pinocchio as pin
        from tatbot_description import names

        self._pin = pin
        xml = re.sub(r'filename="file:(//)?', 'filename="', urdf_xml)   # coal opens plain paths
        self.model = pin.buildModelFromXML(xml)
        self.data = self.model.createData()
        self.geom = pin.buildGeomFromUrdfString(self.model, xml, pin.GeometryType.COLLISION)
        for body in self.geom.geometryObjects:   # a mesh's convex hull: 350x faster, and it encloses the mesh
            if hasattr(body.geometry, "vertices"):
                body.geometry = _hull(np.array(body.geometry.vertices(), float))
        self.arms = sorted({_arm(self.model.names[j]) for j in range(1, self.model.njoints)})
        for arm, profile in (tools or {}).items():
            self._add_tool(arm, profile, names.tool_mount_frame(arm))
        for arm in self.arms:
            self._add_cube(arm)
        bodies = [body.name for body in self.geom.geometryObjects]
        for i, a in enumerate(bodies):
            for j in range(i + 1, len(bodies)):
                # the wrist cube is the arm's own concern: across the arms the wrist's bubble, measured on the rig
                # without it (2026-09-29), already takes its place
                if (_arm(a) != _arm(bodies[j]) and _arm(a) in self.arms and _arm(bodies[j]) in self.arms
                        and not (a.endswith("wrist_cube") or bodies[j].endswith("wrist_cube"))):
                    self.geom.addCollisionPair(pin.CollisionPair(i, j))
        self.self_pairs = {arm: self._self_pairs(arm) for arm in self.arms}
        self.gdata = pin.GeometryData(self.geom)
        self.pad = np.array([next((float(m) for prefix, m in (padding or {}).items()
                                   if body.name.split("/", 1)[-1].startswith(prefix)), 0.0)
                             for body in self.geom.geometryObjects])
        self._iq = {arm: np.array([self.model.idx_qs[self.model.getJointId(n)] for n in names.joint_names(arm)])
                    for arm in self.arms}
        self._q = pin.neutral(self.model)

    def _add_cube(self, arm: str) -> None:
        """The wrist tag cube as a box, which the URDF gives no collision (its tags are visuals): solved from the
        tags' calibrated frames (the generated wrist fiducials), each face's tag at the cube's centre plus its half
        size along the tag's normal. 2026-09-30: the pink arm folded the cube into its own upper arm at an
        inspection view, and nothing modelled it."""
        import coal

        pin = self._pin
        tags = [frame for frame in self.model.frames if re.fullmatch(rf"{arm}/wrist_tag\d+", frame.name)]
        if len(tags) != 3 or len({frame.parentJoint for frame in tags}) != 1:
            return
        normals = np.array([np.asarray(frame.placement.rotation)[:, 2] for frame in tags])
        points = np.array([np.asarray(frame.placement.translation) for frame in tags])
        a = np.vstack([np.hstack([np.eye(3), n[:, None]]) for n in normals])
        solution = np.linalg.lstsq(a, points.ravel(), rcond=None)[0]
        centre, half = solution[:3], float(solution[3])
        u, _, vt = np.linalg.svd(normals.T)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1.0
        placement = pin.SE3(rotation, centre)
        self.geom.addGeometryObject(pin.GeometryObject(f"{arm}/wrist_cube", tags[0].parentJoint, tags[0].parentFrame,
                                                       placement, coal.Box(2 * half, 2 * half, 2 * half)))

    def _self_pairs(self, arm: str) -> list[tuple[int, int]]:
        """The pairs of one arm's bodies checked against each other: every two whose joints stand more than two
        apart along the chain (links within two joints stay a few millimetres apart in every pose, their hulls
        overlapping about the compact wrist: link_3 and link_5 at 4-6 mm over 249 pink calibration poses); and
        the wrist cube against every link from link_5 up, which it can fold onto (2026-09-30: it met link_2), but
        not the parts it rides with."""
        bodies, parents = self.geom.geometryObjects, self.model.parents

        def chain(j):
            out = [j]
            while j > 0:
                j = parents[j]
                out.append(j)
            return out

        def apart(a, b):
            up_a, up_b = chain(a), chain(b)
            common = next(j for j in up_a if j in up_b)
            return up_a.index(common) + up_b.index(common)

        mine = [i for i, body in enumerate(bodies) if _arm(body.name) == arm]
        cubes = [i for i in mine if bodies[i].name.endswith("wrist_cube")]
        above = set(chain(parents[parents[bodies[cubes[0]].parentJoint]])) if cubes else set()
        pairs = []
        for k, i in enumerate(mine):
            for j in mine[k + 1:]:
                if i in cubes or j in cubes:
                    other = bodies[j if i in cubes else i].parentJoint
                    keep = other in above
                else:
                    keep = apart(bodies[i].parentJoint, bodies[j].parentJoint) > 2
                if keep:
                    pairs.append((i, j))
        return pairs

    def self_gaps(self, arm: str, q) -> np.ndarray:
        """The distance of each of `arm`'s self_pairs at joints q (m, negative inside), no bubbles."""
        import coal

        pin = self._pin
        self._q[self._iq[arm]] = np.asarray(q, float)[: len(self._iq[arm])]
        pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata, self._q)
        bodies, out = self.geom.geometryObjects, []
        for i, j in self.self_pairs[arm]:
            mi, mj = self.gdata.oMg[i], self.gdata.oMg[j]
            out.append(coal.distance(bodies[i].geometry, coal.Transform3s(mi.rotation, mi.translation),
                                     bodies[j].geometry, coal.Transform3s(mj.rotation, mj.translation),
                                     coal.DistanceRequest(), coal.DistanceResult()))
        return np.array(out, float)

    def self_clearance(self, arm: str, q, floors=None) -> Gap:
        """The tightest two of `arm`'s own bodies at joints q: each pair's distance less its floor (none: 0),
        the pair's own distance as raw_m."""
        gaps = self.self_gaps(arm, q)
        if len(gaps) == 0:
            return Gap(float("inf"), "", "")
        left = gaps - (0.0 if floors is None else floors)
        k = int(np.argmin(left))
        i, j = self.self_pairs[arm][k]
        return Gap(float(left[k]), self.geom.geometryObjects[i].name, self.geom.geometryObjects[j].name, None,
                   float(gaps[k]))

    def path_self_clearance(self, arm: str, rows, max_step_rad: float = 0.01, floors=None) -> Gap:
        """The closest `arm` comes to itself through `rows`, checked as path_clearance checks."""
        rows = np.asarray(rows, float)
        worst, last = None, None
        for k, row in enumerate(rows):
            if last is not None and k < len(rows) - 1 and np.max(np.abs(row[:6] - last[:6])) < max_step_rad:
                continue
            last = row
            gap = self.self_clearance(arm, row, floors)
            if worst is None or gap.distance_m < worst.distance_m:
                worst = Gap(gap.distance_m, gap.body_a, gap.body_b, k, gap.raw_m)
        return worst if worst is not None else Gap(float("inf"), "", "")

    def parts_clearance(self, arm: str, rows, parts, pads: dict | None = None, skip=("tool",),
                        max_step_rad: float = 0.01) -> Gap:
        """The closest approach of `arm`'s bodies moving through `rows` to fixed parts of the table: `parts` as
        (name, "sphere" or "post", centre (3, world), radius, and for a post its height below the centre along
        `up` (3, world): it stands from there up to the centre). A body named with a prefix of `pads` (after
        "<arm>/") is taken that much larger; bodies with a prefix in `skip` are left out (the tool, whose own
        approach to the station is checked by its halo). Rows are checked as in path_clearance."""
        import coal

        pin = self._pin
        shapes = []
        for name, kind, centre, radius, *post in parts:
            centre = np.asarray(centre, float)
            if kind == "post":
                height, up = float(post[0]), np.asarray(post[1], float)
                z = up / np.linalg.norm(up)
                x = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
                x /= np.linalg.norm(x)
                rotation = np.column_stack([x, np.cross(z, x), z])
                shapes.append((name, coal.Cylinder(float(radius), height),
                               coal.Transform3s(rotation, centre - z * height / 2.0)))
            else:
                shapes.append((name, coal.Sphere(float(radius)), coal.Transform3s(np.eye(3), centre)))
        pads = pads or {}
        bodies = []
        for i, body in enumerate(self.geom.geometryObjects):
            short = body.name.split("/", 1)[-1]
            if _arm(body.name) != arm or short.startswith(tuple(skip)):
                continue
            bodies.append((i, next((float(m) for prefix, m in pads.items() if short.startswith(prefix)), 0.0)))
        rows = np.asarray(rows, float)
        worst, last = None, None
        for k, row in enumerate(rows):
            if last is not None and k < len(rows) - 1 and np.max(np.abs(row[:6] - last[:6])) < max_step_rad:
                continue
            last = row
            self._q[self._iq[arm]] = row[: len(self._iq[arm])]
            pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata, self._q)
            for i, pad in bodies:
                placed = self.gdata.oMg[i]
                at = coal.Transform3s(placed.rotation, placed.translation)
                for name, shape, where in shapes:
                    raw = float(coal.distance(self.geom.geometryObjects[i].geometry, at, shape, where,
                                              coal.DistanceRequest(), coal.DistanceResult()))
                    if worst is None or raw - pad < worst.distance_m:
                        worst = Gap(raw - pad, self.geom.geometryObjects[i].name, name, k, raw)
        return worst if worst is not None else Gap(float("inf"), "", "")

    def _add_tool(self, arm: str, profile, frame_name: str) -> None:
        import coal

        pin = self._pin
        fid = self.model.getFrameId(frame_name)
        if fid >= self.model.nframes:
            raise ValueError(f"the model has no {frame_name} for the {arm} arm's tool")
        frame = self.model.frames[fid]
        for k, (centre, half, radius) in enumerate(_capsules(profile)):
            placement = frame.placement * pin.SE3(np.eye(3), np.array([0.0, 0.0, centre]))
            self.geom.addGeometryObject(pin.GeometryObject(f"{arm}/tool_{k}", frame.parentJoint, fid, placement,
                                                           coal.Capsule(radius, half)))

    @classmethod
    def from_repo(cls, repo=None, arms=("right", "left"), registrations: dict | None = None,
                  padding: dict | None = None) -> Scene:
        """The checkout's two arms placed by `registrations` ({arm: registration path or world_from_arm_base}), each
        with its fitted tool (config/workspace.yaml `tool_id`, its datasheet's profile), and `padding` as in
        Scene."""
        from tatbot_description import repo_root, robot_description

        root = repo_root(repo)
        lib = str(root / "scripts" / "lib")
        if lib not in sys.path:
            sys.path.insert(0, lib)
        import tool_spec

        tools = {}
        for arm in arms:
            tool_id = tool_spec.active_tool_id(root, arm)
            if tool_id:
                tools[arm] = tool_spec.load_tool(tool_id, root).profile
        return cls(robot_description(root, arms=tuple(arms), registrations=registrations), tools, padding)

    def base_pose(self, arm):
        """Fixed arm registration in this loaded scene, independent of later file edits."""
        from tatbot_description.names import base_frame

        self._pin.framesForwardKinematics(self.model, self.data, self._q)
        return self.data.oMf[self.model.getFrameId(base_frame(arm))].homogeneous.copy()

    def clearance(self, joints: dict) -> Gap:
        """The tightest two bodies on different arms, the larger of their bubbles taken off, with each arm at
        `joints[arm]` (seven values; an arm left out keeps its last)."""
        pin = self._pin
        for arm, q in joints.items():
            self._q[self._iq[arm]] = np.asarray(q, float)[: len(self._iq[arm])]
        pin.computeDistances(self.model, self.data, self.geom, self.gdata, self._q)
        pairs = self.geom.collisionPairs
        raw = np.array([result.min_distance for result in self.gdata.distanceResults])
        left = raw - np.array([max(self.pad[pair.first], self.pad[pair.second]) for pair in pairs])
        k = int(np.argmin(left))
        return Gap(float(left[k]), self.geom.geometryObjects[pairs[k].first].name,
                   self.geom.geometryObjects[pairs[k].second].name, None, float(raw[k]))

    def path_clearance(self, arm: str, rows, others: dict, max_step_rad: float = 0.01) -> Gap:
        """The closest approach of `arm` moving through `rows` (joints, row by row) to the other arms held at
        `others`. Rows are checked where some joint has moved max_step_rad since the last checked row (10 mrad is at
        most 7 mm at the arm's reach), and the last row always."""
        rows = np.asarray(rows, float)
        worst, last = None, None
        for i, row in enumerate(rows):
            if last is not None and i < len(rows) - 1 and np.max(np.abs(row[:6] - last[:6])) < max_step_rad:
                continue
            gap = self.clearance({**others, arm: row})
            last = row
            if worst is None or gap.distance_m < worst.distance_m:
                worst = Gap(gap.distance_m, gap.body_a, gap.body_b, i, gap.raw_m)
        return worst if worst is not None else Gap(float("inf"), "", "")


    def zone_clearance(self, arm: str, rows, world_from_zone, radius_m: float, height_m: float,
                       max_step_rad: float = 0.01) -> Gap:
        """The closest approach of `arm`'s bodies, each less its bubble, moving through `rows`, to a cylinder of
        radius_m standing height_m along the z of `world_from_zone` from its origin (a zone of the table: the
        palette's, tatbot_session.lease). Rows are checked as in path_clearance."""
        import coal

        pin = self._pin
        pose = np.asarray(world_from_zone, float)
        centre = pose[:3, :3] @ np.array([0.0, 0.0, height_m / 2.0]) + pose[:3, 3]
        zone, zone_at = coal.Cylinder(float(radius_m), float(height_m)), coal.Transform3s(pose[:3, :3], centre)
        bodies = [i for i, body in enumerate(self.geom.geometryObjects) if _arm(body.name) == arm]
        rows = np.asarray(rows, float)
        worst, last = None, None
        for k, row in enumerate(rows):
            if last is not None and k < len(rows) - 1 and np.max(np.abs(row[:6] - last[:6])) < max_step_rad:
                continue
            last = row
            self._q[self._iq[arm]] = row[: len(self._iq[arm])]
            pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata, self._q)
            for i in bodies:
                placed = self.gdata.oMg[i]
                result = coal.DistanceResult()
                raw = coal.distance(self.geom.geometryObjects[i].geometry,
                                    coal.Transform3s(placed.rotation, placed.translation), zone, zone_at,
                                    coal.DistanceRequest(), result)
                left = float(raw) - float(self.pad[i])
                if worst is None or left < worst.distance_m:
                    worst = Gap(left, self.geom.geometryObjects[i].name, "zone", k, float(raw))
        return worst if worst is not None else Gap(float("inf"), "", "")


SELF_SLACK_M = 0.001    # a way near itself may come this much nearer than it starts (the model's own jitter)
MAX_SWEEP_ROWS = 60     # a way in flight is checked against another at most this many rows each (0.36 s at 0.1 ms)
REACH_M_PER_RAD = 0.7   # a joint turning one radian moves no point of the arm farther (its reach, the tool's tip)


class Guard:
    """Whether a planned way keeps an arm clear of the other: every goal the stack sends passes it (tatbot_session
    ArmIO.execute). The other arm stands where the stack measures it (/joint_states, when this stack drives it too),
    else at its landed pose (config/trossen/tatbot.yaml follower.staged_positions), which a peer that drives it
    elsewhere (the LeRobot policy on the arm node) does not keep; the refusal names which. A way that only takes the
    arm farther from the other is never refused, so an arm already inside the margin can always back away."""

    def __init__(self, scene: Scene, margin_m: float, rest: dict, step_rad: float = 0.01, self_margin_m: float = 0.010):
        self.scene, self.margin_m, self.rest, self.step_rad = scene, float(margin_m), rest, float(step_rad)
        self.self_margin_m = float(self_margin_m)
        self._lock = threading.Lock()   # one scene, and each arm's executor asks from its own thread
        self._floors: dict[str, np.ndarray] = {}   # arm -> each self pair's floor (_self_floors)
        self._ways: dict[str, tuple[np.ndarray, float]] = {}   # arm -> its way in flight (_sweep)
        self.zone_source = None   # () -> the palette zone another arm's run holds, or None (tatbot_session.lease)

    @classmethod
    def from_stack(cls, repo, stack: dict, motion: dict) -> tuple[Guard | None, str]:
        """(the guard, a line for the log), or (None, why not): both arms need a registration to share a frame."""
        import yaml
        from tatbot_description import names, repo_root

        registrations = {arm: str(Path(path).expanduser()) for arm, path in (stack.get("registration") or {}).items()
                         if path and Path(path).expanduser().is_file()}
        missing = [arm for arm in names.ARMS if arm not in registrations]
        if missing:
            return None, (f"arm clearance not checked: no registration for the {' and '.join(missing)} arm, so the "
                          "two arms share no frame")
        cfg = motion["collision"]
        staged = yaml.safe_load((repo_root(repo) / "config" / "trossen" / "tatbot.yaml").read_text())["follower"]
        rest = {arm: np.asarray(staged["staged_positions"], float) for arm in names.ARMS}
        guard = cls.from_registrations(repo, registrations, motion, rest)
        return guard, (f"arm clearance: every goal keeps {guard.margin_m * 1000:.0f} mm between the arms, each wrist "
                       f"and end effector bubbled out {float(cfg['wrist_m']) * 1000:.0f} mm for its cable loops (a "
                       "peer that drives the other arm outside this stack is assumed landed)")

    @classmethod
    def from_registrations(cls, repo, registrations: dict, motion: dict, rest: dict) -> Guard:
        """Both arms placed by `registrations` ({arm: path or world_from_arm_base}), with motion.yaml `collision`'s
        margin and wrist bubbles, and `rest` ({arm: joints}) for an arm the stack does not measure."""
        cfg = motion["collision"]
        padding = dict.fromkeys(cfg["wrist_bodies"], float(cfg["wrist_m"]))
        return cls(Scene.from_repo(repo, tuple(registrations), registrations, padding), cfg["arms_m"], rest,
                   cfg["step_rad"], cfg.get("self_m", 0.010))

    def others(self, arm: str, measured: dict) -> tuple[dict, list[str]]:
        """The other arms' joints ({arm: q}) and where each came from: `measured` ({arm: q}, current joints this
        stack reads), else its landed pose."""
        joints, sources = {}, []
        for other in self.scene.arms:
            if other == arm:
                continue
            q = measured.get(other)
            joints[other] = self.rest[other] if q is None else np.asarray(q, float)
            sources.append(f"the {other} arm {'as measured' if q is not None else 'assumed landed'}")
        return joints, sources

    def refusal(self, arm: str, rows, measured: dict) -> str | None:
        """Why the way `rows` (joints, row by row) of `arm` must not be sent, or None: onto itself (the self
        pairs, the wrist cube among them), then near the other arm."""
        with self._lock:
            return self._refusal(arm, rows, measured)

    def _refusal(self, arm: str, rows, measured: dict) -> str | None:
        rows = np.asarray(rows, float)
        if len(rows) == 0:
            return None
        others, sources = self.others(arm, measured)
        why = self._self_refusal(arm, rows)
        if why is not None:
            return why
        gap = self.scene.path_clearance(arm, rows, others, self.step_rad)
        if gap.distance_m >= self.margin_m:
            return None
        start = self.scene.clearance({**others, arm: rows[0]})
        if gap.distance_m >= start.distance_m - 1e-4:   # never nearer than it starts: backing away is allowed
            return None
        need = gap.raw_m - gap.distance_m + self.margin_m
        return (f"the {arm} arm's way comes {gap.raw_m * 1000:.0f} mm from the other arm ({gap.body_a} to "
                f"{gap.body_b}), where those two keep {need * 1000:.0f} mm (a wrist's cable-loop bubble and the "
                f"{self.margin_m * 1000:.0f} mm margin), with {', '.join(sources)}; not sent")

    def self_gap(self, arm: str, q) -> float:
        """How far the arm at joints q stands from coming onto itself: its tightest self pair's distance less that
        pair's floor (_self_floors), m; negative inside. For a single pose (a view to choose), no way needed."""
        with self._lock:
            return self.scene.self_clearance(arm, q, self._self_floors(arm)).distance_m

    def _self_refusal(self, arm: str, rows) -> str | None:
        """Why the way brings `arm` onto itself: two of its own bodies (Scene.self_pairs) under their floor
        (_self_floors) and nearer than where the way starts by more than SELF_SLACK_M, or None."""
        floors = self._self_floors(arm)
        gap = self.scene.path_self_clearance(arm, rows, self.step_rad, floors)
        if gap.distance_m >= 0.0:
            return None
        start = self.scene.self_clearance(arm, rows[0], floors)
        if gap.distance_m >= start.distance_m - SELF_SLACK_M:
            return None
        return (f"the {arm} arm's way brings its {gap.body_a.split('/')[-1]} {gap.raw_m * 1000:.0f} mm from its own "
                f"{gap.body_b.split('/')[-1]}, under the {(gap.raw_m - gap.distance_m) * 1000:.0f} mm those keep; "
                "not sent")

    def _self_floors(self, arm: str) -> np.ndarray:
        """How near each self pair may come: self_margin_m, or as near as the pair stands at rest (the staged and
        the sleep pose, which the arm holds every day) less SELF_SLACK_M, when that is nearer. The landed arm folds
        onto itself, its hulls overlapping (link_2 and link_5 by 1.5 mm), and ways to and from rest go."""
        if arm not in self._floors:
            staged = np.asarray(self.rest[arm], float)
            sleep = np.zeros(7)
            sleep[5], sleep[6] = staged[5], staged[6]
            at_rest = np.minimum(self.scene.self_gaps(arm, staged), self.scene.self_gaps(arm, sleep))
            self._floors[arm] = np.minimum(self.self_margin_m, at_rest - SELF_SLACK_M)
        return self._floors[arm]

    # --- ways in flight ---------------------------------------------------------------------------------------
    # Two arms in one stack move at once (2026-09-30: the pink arm drawing while the blue arm calibrates at the
    # palette). Where the other arm stands at dispatch says nothing of where its goal takes it next, so each goal's
    # way is reserved while it runs, and a way is checked against the other arms' reserved ways row by row, every
    # row of one against every row of the other: when each will be where is left out, which only ever refuses more.

    def reserve(self, arm: str, rows, measured: dict) -> tuple[str | None, bool]:
        """At once: why `arm`'s way must not be sent (refusal, then into a zone another arm's run holds, then
        against every way in flight), and whether only something that ends blocks it (a zone or a way in flight:
        waiting may clear it). When None, the way stays reserved for `arm` until release(arm)."""
        with self._lock:
            why = self._refusal(arm, rows, measured)
            if why is not None:
                return why, False
            why = self._into_zone(arm, rows)
            if why is not None:
                return why, True
            sweep = self._sweep(rows)
            why = self._against_ways(arm, sweep)
            if why is not None:
                return why, True
            self._ways[arm] = sweep
            return None, False

    def hold_way(self, arm: str, rows) -> None:
        """Reserve `arm`'s way unchecked: a landing, which goes whatever stands in its way, so that the other arm's
        next goals keep clear of it."""
        with self._lock:
            self._ways[arm] = self._sweep(rows)

    def release(self, arm: str) -> None:
        with self._lock:
            self._ways.pop(arm, None)

    def _into_zone(self, arm: str, rows) -> str | None:
        """Why the way enters the zone another arm's run holds (zone_source: tatbot_session.lease.held_zone), or
        None. One that never gets nearer than it starts can leave it."""
        zone = self.zone_source() if self.zone_source is not None else None
        if not zone or zone.get("arm") == arm:
            return None
        rows = np.asarray(rows, float)
        shape = (zone["world_from_zone"], zone["radius_m"], zone["height_m"])
        gap = self.scene.zone_clearance(arm, rows, *shape, self.step_rad)
        if gap.distance_m >= self.margin_m:
            return None
        if gap.distance_m >= self.scene.zone_clearance(arm, rows[:1], *shape).distance_m - 1e-4:
            return None
        return (f"the {arm} arm's way comes {gap.raw_m * 1000:.0f} mm from the palette ({gap.body_a}), which the "
                f"{zone['arm']} arm's run {zone.get('run', '')} holds; waiting for it")

    def _sweep(self, rows) -> tuple[np.ndarray, float]:
        """The way's rows where some joint has moved step_rad since the last kept one (the last always), at most
        MAX_SWEEP_ROWS of them, evenly; and the slack that coarser sampling leaves: half the largest step between
        kept rows past step_rad, at REACH_M_PER_RAD."""
        rows = np.asarray(rows, float)
        kept, last = [], None
        for i, row in enumerate(rows):
            if last is not None and i < len(rows) - 1 and np.max(np.abs(row[:6] - last[:6])) < self.step_rad:
                continue
            kept.append(row)
            last = row
        kept = np.array(kept) if kept else rows[:1]
        if len(kept) > MAX_SWEEP_ROWS:
            kept = kept[np.unique(np.linspace(0, len(kept) - 1, MAX_SWEEP_ROWS).round().astype(int))]
        jump = float(np.max(np.abs(np.diff(kept[:, :6], axis=0)))) if len(kept) > 1 else 0.0
        return kept, max(0.0, jump - self.step_rad) * REACH_M_PER_RAD / 2.0

    def _against_ways(self, arm: str, sweep) -> str | None:
        rows, slack = sweep
        for other, (theirs, their_slack) in self._ways.items():
            if other == arm:
                continue
            worst, start = None, float("inf")
            for i, row in enumerate(rows):
                for q in theirs:
                    gap = self.scene.clearance({other: q, arm: row})
                    left = gap.distance_m - slack - their_slack
                    if i == 0:
                        start = min(start, left)
                    if worst is None or left < worst[0]:
                        worst = (left, gap)
            if worst is None or worst[0] >= self.margin_m or worst[0] >= start - 1e-4:
                continue
            left, gap = worst
            return (f"the {arm} arm's way comes {(gap.raw_m - (gap.distance_m - left)) * 1000:.0f} mm from where the "
                    f"{other} arm's goal in flight takes it ({gap.body_a} to {gap.body_b}), under the "
                    f"{self.margin_m * 1000:.0f} mm margin past the cable-loop bubbles; waiting for it")
        return None
