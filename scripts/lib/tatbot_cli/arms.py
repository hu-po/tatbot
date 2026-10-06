"""Physical arm identities and teleop roles; stdlib, no hardware access.

Legacy driver profile keys and SDK EE names identify hardware here, even when
their spelling includes a teleop role. Assigning a role never exchanges them.
This registry is not a calibration or permission to command a controller.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

# The current teleop adapter has these two hardware roles.
ARM_IDS = ("left", "right")
REVERSE_BLOCKER = (
    "Reverse teleop execution is limited to --wrist-calibration: right-leading-left "
    "with joints 0, 4 and 5 mirrored from the starting poses for free-space capture. Other modes still require the "
    "physical right arm as receiver. Controller addresses stay physically bound."
)


@dataclass(frozen=True)
class TeleopRoles:
    leader: str
    follower: str


@dataclass(frozen=True)
class PhysicalArm:
    id: str
    control_role: str
    profile_ip_field: str
    controller_config: str
    sdk_end_effector: str
    workspace_section: str
    urdf_prefix: str


def select_roles(leader: str | None = None, follower: str | None = None) -> TeleopRoles:
    """One specified role assigns the other arm; neither preserves the default."""
    for value in (leader, follower):
        if value is not None and value not in ARM_IDS:
            raise ValueError(f"unknown physical arm {value!r}; expected left or right")
    if leader is None:
        leader = "right" if follower == "left" else "left"
    if follower is None:
        follower = "left" if leader == "right" else "right"
    if leader == follower:
        raise ValueError("teleop leader and follower must be different physical arms")
    return TeleopRoles(leader, follower)


def _physical_arm(arm_id: str, entry: dict) -> PhysicalArm:
    fields = set(PhysicalArm.__dataclass_fields__) - {"id"}
    if not isinstance(arm_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", arm_id):
        raise ValueError(f"invalid physical arm ID: {arm_id!r}")
    if (not isinstance(entry, dict) or set(entry) != fields
            or any(not isinstance(v, str) or not v.strip() for v in entry.values())):
        raise ValueError(f"invalid physical arm entry: {arm_id}")
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", entry[field])
               for field in ("workspace_section", "urdf_prefix", "profile_ip_field", "control_role")):
        raise ValueError(f"{arm_id}: invalid arm binding")
    if entry["workspace_section"] != entry["urdf_prefix"]:
        raise ValueError(f"{arm_id}: workspace and URDF must name the same physical chain")
    config = PurePosixPath(entry["controller_config"])
    if (config.is_absolute() or ".." in config.parts or len(config.parts) < 2
            or config.parts[0] != "config" or config.suffix != ".yaml"):
        raise ValueError(f"{arm_id}: controller config must be a relative YAML path under config/")
    return PhysicalArm(id=arm_id, **entry)


def load(repo: Path) -> dict[str, PhysicalArm]:
    """Read identity references only; never open controller or calibration files."""
    path = repo / "config/arms.json"
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read arm registry: {exc}") from exc
    if (not isinstance(document, dict) or set(document) != {"schema", "arms"}
            or document["schema"] != "tatbot.arms/1"
            or not isinstance(document["arms"], dict) or not document["arms"]):
        raise ValueError("arm registry requires tatbot.arms/1 and at least one arm")
    result = {arm_id: _physical_arm(arm_id, entry) for arm_id, entry in document["arms"].items()}
    # Renaming an ID may keep the installed URDF and calibration sections, but
    # two IDs may never address one controller, kinematic chain or tool fit.
    for field in ("profile_ip_field", "controller_config", "workspace_section", "urdf_prefix"):
        values = [getattr(arm, field) for arm in result.values()]
        if len(values) != len(set(values)):
            raise ValueError(f"physical arms must have distinct {field} references")
    return result


CONTROLLER_ROLES = {"config/trossen/leader.yaml": "leader", "config/trossen/follower.yaml": "follower"}


def execution_blockers(arms: dict[str, PhysicalArm], roles: TeleopRoles,
                       wrist_calibration: bool = False) -> list[str]:
    """Describe the existing executor, not a user-editable qualification switch."""
    blockers = []
    if set(arms) != set(ARM_IDS):
        blockers.append("the current teleop executor requires exactly the configured left and right arms")
    if wrist_calibration:
        if roles != select_roles("right", "left"):
            blockers.append("wrist calibration requires right leader and left follower")
    elif roles != select_roles():
        blockers.append(REVERSE_BLOCKER)
    for arm_id, legacy_role in (("left", "leader"), ("right", "follower")):
        if arm_id not in arms:
            continue
        arm = arms[arm_id]
        if (arm.control_role != "arm" or arm.profile_ip_field != f"{legacy_role}_ip"
                or arm.controller_config != f"config/trossen/{legacy_role}.yaml"
                or arm.sdk_end_effector != f"wxai_v0_{legacy_role}"):
            blockers.append(f"{arm_id}: registry differs from the current executor's physical binding")
    return blockers


def require_current_executor(repo: Path, roles: TeleopRoles, *, wrist_calibration: bool = False) -> None:
    blockers = execution_blockers(load(repo), roles, wrist_calibration)
    if blockers:
        raise ValueError("; ".join(blockers))


def teleop_plan(repo: Path, roles: TeleopRoles) -> dict:
    arms = load(repo)
    blockers = execution_blockers(arms, roles)
    return {
        "schema": "tatbot.teleop-assignment/1",
        "mode": "teleop",
        "assignments": {role: asdict(arms[getattr(roles, role)])
                        for role in ("leader", "follower")},
        "required_arm_claims": sorted((roles.leader, roles.follower)),
        "ownership": "both arms together; existing single-owner driver lease remains in force",
        "tool_binding": {"workspace": "config/workspace.yaml", "section": roles.follower},
        "execution": {"implemented": not blockers, "blockers": blockers,
                      "motion_authorized": False,
                      "note": "Assignment only; launcher gates, physical fit and calibration still required."},
        "camera_binding": "Physical-arm camera ownership is resolved separately from the vision registry; "
                          "teleop roles never exchange camera assignments.",
    }
