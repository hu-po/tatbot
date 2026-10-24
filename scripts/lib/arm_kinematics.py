"""One configured arm's URDF names: its six revolute joints, the driven carriage, link 6, the tool mount.

FK is `urdf_kinematics.UrdfChain` over the nominal urdf/tatbot.urdf. The drawing stack's model of
the arm (tatbot_description, tatbot_motion, the simulator) also carries config/workspace.yaml
`joint_offsets_rad`, the encoder-zero correction; a caller that places the real arm from its
measured joints adds those first (`kinematic_calibration`), as the stencil observer and
`tatbot teleop poses` do, or it sits millimetres from where the planner put the tip.
"""

from __future__ import annotations

from pathlib import Path

from tatbot_cli import arms as arm_registry

REPO = Path(__file__).resolve().parents[2]
URDF_PATH = REPO / "urdf" / "tatbot.urdf"

_CHAIN = None


def urdf_chain(urdf_path: Path | str | None = None):
    """The shared UrdfChain (scripts/vision/urdf_kinematics.py), the checkout's loaded once."""
    global _CHAIN
    from urdf_kinematics import UrdfChain  # scripts/vision on sys.path
    if urdf_path is not None:
        return UrdfChain(urdf_path)
    if _CHAIN is None:
        _CHAIN = UrdfChain(URDF_PATH)
    return _CHAIN


ARM_IDS = tuple(arm_registry.load(REPO))


class ArmModel:
    """One WXAI arm's frames under its configured URDF prefix.

    Both arms carry the same joint chain; the driver's seventh axis is
    `<arm>/left_carriage_joint`, which the tool mount rides.
    """

    def __init__(self, arm: str = "right", repo: Path = REPO):
        physical = arm_registry.load(repo).get(arm)
        if physical is None:
            raise ValueError(f"unknown arm {arm!r}; expected a configured arm ID")
        if not physical.sdk_end_effector.startswith("wxai_v0_"):
            raise ValueError(f"{arm}: the WXAI kinematics adapter does not support {physical.sdk_end_effector}")
        self.arm = arm
        self.prefix = physical.urdf_prefix
        self.joint_names = tuple(f"{self.prefix}/joint_{i}" for i in range(6))
        self.carriage_joint = f"{self.prefix}/left_carriage_joint"
        self.link6 = f"{self.prefix}/link_6"
        self.tool_mount = f"{self.prefix}/tool_mount"
        self.frame = f"{self.prefix}/base_link"
        self.controller_role = arm_registry.CONTROLLER_ROLES.get(
            physical.controller_config, Path(physical.controller_config).stem)

    def joint_map(self, joints, carriage_m: float = 0.0) -> dict[str, float]:
        """{urdf joint name: value} for the six arm joints plus the carriage, as given (no offsets)."""
        values = {name: float(v) for name, v in zip(self.joint_names, joints, strict=True)}
        values[self.carriage_joint] = float(carriage_m)
        return values
