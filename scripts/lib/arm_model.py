"""The arm a stack drives, as data.

config/arm-models/<model>.yaml describes one kind of arm: its joints in controller order, each one's kind
(revolute or prismatic) and role (base, shoulder, elbow, wrist, carriage), and the poses and ranges the stack
moves it by. config/arms.json names the model of each physical arm. Every joint index the drawing stack and the
scripts use comes from an ArmModel: which joints are the arm's revolute chain, which one is the WXAI's tool
carriage, if any, which are the wrist, and which zero offsets kinematic calibration may fit. No other code
names a joint by its number.

Stdlib only, as tool_spec is for the tool datasheets, and in the same YAML subset: the ROS stack
(through tatbot_description) and the bare-interpreter scripts read one file the same way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tatbot_paths import repo_root
from tool_spec import parse_simple_yaml

FORMAT = "tatbot.arm-model"
VERSION = 1
KINDS = ("revolute", "prismatic")
ROLES = ("base", "shoulder", "elbow", "wrist", "carriage")
CONTROLLERS = ("trossen", "xarm")   # the make of its controller: which SDK tatbot_hardware drives it with


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str   # KINDS: radians or metres
    role: str   # ROLES


@dataclass(frozen=True)
class Posture:
    """How a chain longer than the task's six joints spends its self-motion: CLIK pulls each joint toward `target` at
    `gain_s` times its weight, through the task's null space, so the tip never feels it, its fastest joint at
    `max_rad_s` at most however far from the target the arm starts."""
    gain_s: float
    target: tuple[float, ...]
    weights: tuple[float, ...]
    max_rad_s: float


@dataclass(frozen=True)
class FakePlant:
    """How tatbot_hardware's fake arm answers its commands: a pure delay, then a first-order lag."""
    delay_s: float
    lag_s: float


@dataclass(frozen=True)
class ArmModel:
    id: str
    urdf: str                                       # its description, relative to the repo (each arm's prefix in it)
    controller: str                                 # one of CONTROLLERS
    joints: tuple[Joint, ...]
    rest: tuple[float, ...]                         # the landing's sleep pose, controller order
    view_weights: tuple[float, ...]                 # joint-space cost of an inspect view's move
    wrist_range_rad: tuple[tuple[float, float], ...]  # each wrist joint's range for inspect's own moves
    calibrated_offsets: tuple[int, ...]             # joints whose zero offsets kinematic calibration fits
    command_delay_s: float                          # the controller's pure command-to-measurement delay
    wrist_spheres: tuple[tuple[str, float], ...]    # (link, radius m): the wrist's links as spheres about their frames
    posture: Posture | None = None                  # the self-motion's preference: a chain longer than six needs one
    fake_plant: FakePlant | None = None             # the fake arm's response; None: the fake's own (no delay)
    collision_sensitivity: int | None = None        # an xArm's collision detection, 0 (off) to 5; None: the controller's

    @property
    def n(self) -> int:
        """How many joints the controller drives."""
        return len(self.joints)

    @property
    def arm(self) -> tuple[int, ...]:
        """The revolute chain that places the tool: every joint but a carriage."""
        return tuple(i for i, joint in enumerate(self.joints) if joint.role != "carriage")

    @property
    def arm_index(self) -> slice | list[int]:
        """The revolute chain as a numpy index: a slice when its joints are contiguous, which numpy takes as a view
        and so computes exactly as a literal `[:6]` does, else a list."""
        chain = self.arm
        if chain == tuple(range(chain[0], chain[-1] + 1)):
            return slice(chain[0], chain[-1] + 1)
        return list(chain)

    @property
    def carriage(self) -> int | None:
        """The WXAI's tool carriage (prismatic, metres), or None for an arm without one."""
        return next((i for i, joint in enumerate(self.joints) if joint.role == "carriage"), None)

    @property
    def wrist(self) -> tuple[int, ...]:
        return tuple(i for i, joint in enumerate(self.joints) if joint.role == "wrist")

    def of_roles(self, *roles: str) -> tuple[int, ...]:
        """The joints with any of these roles, in controller order."""
        return tuple(i for i, joint in enumerate(self.joints) if joint.role in roles)

    def joint_names(self, prefix: str | None) -> tuple[str, ...]:
        """The URDF's joint names under a physical arm's prefix, in controller order; prefix None names them
        unprefixed, as a single-arm URDF (the simulator's) does."""
        return tuple(f"{prefix}/{joint.name}" if prefix else joint.name for joint in self.joints)


def parse(document: dict, source: str = "arm model") -> ArmModel:
    """An ArmModel from a loaded tatbot.arm-model/1 document; raises ValueError naming what is wrong."""
    if document.get("format") != FORMAT or document.get("version") != VERSION:
        raise ValueError(f"{source}: not {FORMAT}/{VERSION}")
    joints = tuple(Joint(str(j["name"]), str(j["kind"]), str(j["role"])) for j in document["joints"])
    if len({j.name for j in joints}) != len(joints):
        raise ValueError(f"{source}: joint names repeat")
    for joint in joints:
        if joint.kind not in KINDS or joint.role not in ROLES:
            raise ValueError(f"{source}: joint {joint.name}: kind {joint.kind!r} or role {joint.role!r} unknown")
        if (joint.role == "carriage") != (joint.kind == "prismatic"):
            raise ValueError(f"{source}: joint {joint.name}: only the carriage is prismatic")
    if sum(j.role == "carriage" for j in joints) > 1:
        raise ValueError(f"{source}: more than one carriage")
    model = ArmModel(
        id=str(document["id"]),
        urdf=str(document["urdf"]),
        controller=str(document["controller"]),
        joints=joints,
        rest=tuple(float(v) for v in document["rest"]),
        view_weights=tuple(float(v) for v in document["view_weights"]),
        wrist_range_rad=tuple((float(lo), float(hi)) for lo, hi in document["wrist_range_rad"]),
        calibrated_offsets=tuple(int(i) for i in document["calibrated_offsets"]),
        command_delay_s=float(document["command_delay_s"]),
        wrist_spheres=tuple((str(link), float(radius)) for link, radius in document["wrist_spheres"]),
        posture=_posture(document.get("posture")),
        fake_plant=None if document.get("fake_plant") is None else FakePlant(
            float(document["fake_plant"]["delay_s"]), float(document["fake_plant"]["lag_s"])),
        collision_sensitivity=None if document.get("collision_sensitivity") is None else
        int(document["collision_sensitivity"]),
    )
    _check_sizes(model, source)
    _check_values(model, source)
    _check_controller(model, source)
    return model


def _check_controller(model: ArmModel, source: str) -> None:
    if model.controller not in CONTROLLERS:
        raise ValueError(f"{source}: controller {model.controller!r} is none of {', '.join(CONTROLLERS)}")
    if model.collision_sensitivity is not None and not 0 <= model.collision_sensitivity <= 5:
        raise ValueError(f"{source}: collision_sensitivity runs 0 (off) to 5")


def _posture(section: dict | None) -> Posture | None:
    if section is None:
        return None
    return Posture(float(section["gain_s"]), tuple(float(v) for v in section["target"]),
                   tuple(float(v) for v in section["weights"]), float(section["max_rad_s"]))


def _check_sizes(model: ArmModel, source: str) -> None:
    """Each per-joint list has one entry per joint, and each per-wrist list one per wrist joint."""
    if len(model.rest) != model.n or len(model.view_weights) != model.n:
        raise ValueError(f"{source}: rest and view_weights need one value per joint ({model.n})")
    if len(model.wrist_range_rad) != len(model.wrist):
        raise ValueError(f"{source}: wrist_range_rad needs one range per wrist joint ({len(model.wrist)})")
    if any(not lo < hi for lo, hi in model.wrist_range_rad):
        raise ValueError(f"{source}: a wrist range is empty")
    if any(i not in model.arm for i in model.calibrated_offsets):
        raise ValueError(f"{source}: calibrated_offsets must name revolute joints")
    if model.posture is not None and (len(model.posture.target) != model.n or len(model.posture.weights) != model.n
                                      or not model.posture.gain_s > 0 or not model.posture.max_rad_s > 0
                                      or min(model.posture.weights) < 0):
        raise ValueError(f"{source}: posture needs a positive gain_s and max_rad_s, and a target and non-negative "
                         "weights per joint")


def _check_values(model: ArmModel, source: str) -> None:
    """Each delay, size and distance in its range, and a posture wherever the chain has a self-motion."""
    if not model.command_delay_s >= 0:
        raise ValueError(f"{source}: command_delay_s must be a delay, >= 0")
    if not model.wrist_spheres or any(not link or not radius > 0 for link, radius in model.wrist_spheres):
        raise ValueError(f"{source}: wrist_spheres needs each wrist link's name and a radius > 0")
    if model.fake_plant is not None and not (model.fake_plant.delay_s >= 0 and model.fake_plant.lag_s > 0):
        raise ValueError(f"{source}: fake_plant needs a delay_s >= 0 and a lag_s > 0")
    if model.posture is None and len(model.arm) > 6:
        raise ValueError(f"{source}: a chain of {len(model.arm)} joints needs a posture for its self-motion")


def _root(repo) -> Path:
    return Path(repo).expanduser().resolve() if repo else repo_root()


@lru_cache(maxsize=None)
def _load(root: Path, model_id: str) -> ArmModel:
    path = root / "config" / "arm-models" / f"{model_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no arm model {model_id!r}: {path}")
    model = parse(parse_simple_yaml(path.read_text()), str(path))
    if model.id != model_id:
        raise ValueError(f"{path}: id {model.id!r} is not its file name")
    return model


def load(model_id: str, repo=None) -> ArmModel:
    """The model config/arm-models/<model_id>.yaml describes, in `repo` (default: TATBOT_REPO, else this checkout)."""
    return _load(_root(repo), model_id)


@lru_cache(maxsize=None)
def _model_of(root: Path, arm: str) -> str:
    arms = json.loads((root / "config" / "arms.json").read_text())["arms"]
    if arm not in arms:
        raise KeyError(f"config/arms.json has no physical arm {arm!r}")
    return str(arms[arm]["model"])


def for_arm(arm: str, repo=None) -> ArmModel:
    """The model of the physical arm `arm` (config/arms.json)."""
    root = _root(repo)
    return _load(root, _model_of(root, arm))
