"""How close the pink arm comes to itself and to the station's parts: the arm and its tool as convex bodies in the
overhead camera's world, placed by its registration (tatbot_description robot_description). Each link is the convex
hull of its URDF collision mesh (the wrist D405s' among them), the fitted tool a chain of capsules along its datasheet
profile (config/tools), and the wrist tag cube a box solved from its tags. Distances come from pinocchio's geometry
model on coal, the library the stack's kinematics already load. A hull encloses its mesh and a capsule its profile, so
a clearance is never overstated. The blue arm is the teleop leader alone and is not modelled. Pure numpy on pinocchio
and coal; no ROS import.
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
    """The tightest pair of bodies: the distance left between them once a floor or pad is taken off (m; negative
    inside), which two, where along a path (its row) if a path was checked, and the bodies' own distance."""
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
    """The arm's bodies in one model, from URDF text with the arm placed. `tools` maps an arm to its fitted tool's
    profile ((z, r), ..., the tip last, in <arm>/tool_mount along its z). Joints are the seven controlled ones in
    controller order (tatbot_description.names.joint_names)."""

    def __init__(self, urdf_xml: str, tools: dict | None = None):
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
        self.self_pairs = {arm: self._self_pairs(arm) for arm in self.arms}
        self.gdata = pin.GeometryData(self.geom)
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
        """The closest `arm` comes to itself through `rows` (joints, row by row). Rows are checked where some joint
        has moved max_step_rad since the last checked row (10 mrad is at most 7 mm at the arm's reach), and the last
        row always."""
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
        approach to the station is checked by its halo). Rows are checked as in path_self_clearance."""
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
    def from_repo(cls, repo=None, arms=("right",), registrations: dict | None = None) -> Scene:
        """The checkout's arm placed by `registrations` ({arm: registration path or world_from_arm_base}), with its
        fitted tool (config/workspace.yaml `tool_id`, its datasheet's profile)."""
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
        return cls(robot_description(root, arms=tuple(arms), registrations=registrations), tools)

    def base_pose(self, arm):
        """Fixed arm registration in this loaded scene, independent of later file edits."""
        from tatbot_description.names import base_frame

        self._pin.framesForwardKinematics(self.model, self.data, self._q)
        return self.data.oMf[self.model.getFrameId(base_frame(arm))].homogeneous.copy()


SELF_SLACK_M = 0.001    # a way near itself may come this much nearer than it starts (the model's own jitter)


class Guard:
    """Whether a planned way keeps the arm off itself: every goal the stack sends passes it (tatbot_session
    ArmIO.execute). A way that never brings two of its bodies nearer than it starts is never refused, so an arm already
    folded can always unfold."""

    def __init__(self, scene: Scene, rest: dict, step_rad: float = 0.01, self_margin_m: float = 0.010):
        self.scene, self.rest, self.step_rad = scene, rest, float(step_rad)
        self.self_margin_m = float(self_margin_m)
        self._lock = threading.Lock()   # one scene, asked from the executor's and the node's threads
        self._floors: dict[str, np.ndarray] = {}   # arm -> each self pair's floor (_self_floors)

    @classmethod
    def from_stack(cls, repo, stack: dict, motion: dict) -> tuple[Guard | None, str]:
        """(the guard, a line for the log), or (None, why not): the arm the stack registers (stack.yaml
        `registration`), placed by its registration, its rest the landed pose (config/trossen/tatbot.yaml
        follower.staged_positions)."""
        import yaml
        from tatbot_description import repo_root

        registrations = {arm: str(Path(path).expanduser()) for arm, path in (stack.get("registration") or {}).items()
                         if path and Path(path).expanduser().is_file()}
        if not registrations:
            return None, "arm clearance not checked: no arm registration"
        staged = yaml.safe_load((repo_root(repo) / "config" / "trossen" / "tatbot.yaml").read_text())["follower"]
        rest = {arm: np.asarray(staged["staged_positions"], float) for arm in registrations}
        guard = cls.from_registrations(repo, registrations, motion, rest)
        return guard, f"arm clearance: the {', '.join(registrations)} arm kept off itself"

    @classmethod
    def from_registrations(cls, repo, registrations: dict, motion: dict, rest: dict) -> Guard:
        """The arm placed by `registrations` ({arm: path or world_from_arm_base}), with motion.yaml `collision`'s
        step and self margin, and `rest` ({arm: joints}) its landed pose."""
        cfg = motion["collision"]
        return cls(Scene.from_repo(repo, tuple(registrations), registrations), rest, cfg["step_rad"],
                   cfg.get("self_m", 0.010))

    def refusal(self, arm: str, rows) -> str | None:
        """Why the way `rows` (joints, row by row) of `arm` must not be sent, or None: it brings the arm onto itself
        (the self pairs, the wrist cube among them)."""
        rows = np.asarray(rows, float)
        if len(rows) == 0:
            return None
        with self._lock:
            return self._self_refusal(arm, rows)

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
