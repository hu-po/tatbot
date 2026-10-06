"""External joint efforts from the simulator's own contact force.

Until now ``ext_eff`` -- the second half of every episode's 14-wide state -- was
zero-filled, so the effort channel carried no signal, co-training had to mask
the very channel the real follower populates, and a policy could not learn "I
am touching". This module fills it from a quantity the simulator actually
computes: the contact force between the tool tip and the substrate, mapped to
joint efforts through the manipulator Jacobian.

**Measured, not modelled.** SAPIEN reports the pairwise contact force directly,
so there is no penalty stiffness to invent and no coefficient to tune. Pressing
the tip into the pad reads 0 N at 0.2 mm clear, 0.075 N on the surface, and
0.99 N at 2 mm of commanded overshoot (measured 2026-09-08). ``tau = Jv^T F``
is then the exact static relationship between a force at the TCP and the joint
efforts resisting it, which is what "external effort" means: the gravity and
inertia terms the driver compensates away never enter.

**Only where contact is simulated.** A rigid tip and a collision substrate are
what make the force real, and the env already distinguishes that case
(``surface_has_contact_collision``, ``rigid-contact-v1``). On a kinematic
surface -- curved silicone, a posed body patch -- there are no contact pairs
and the force is identically zero, which is an absence of modelling rather than
an absence of contact. :func:`external_joint_efforts` reports which case it is
so a reader is never handed zeros that look like measurements.

**The bench's channel is a different quantity, and is reconciled explicitly.**
The follower's ``effort.carriage_n`` reads about +3.64 N with the tip clear and
+3.15 N in contact: a standing preload that contact *lowers* while pinning it,
where the simulated force rises from zero. Left alone the two would teach a
co-trained policy that one channel means two things.
:class:`EffortCalibration` maps between them from a measured session, so a
generated shard's carriage channel sits on the bench's scale. It reproduces the
two levels and not the within-state spread, and it is calibration for
co-training only -- scoring the contact model that same session fitted against
these numbers would be scoring a fit on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FORCE_MODEL = "sapien-pairwise-contact-v1"
"""Named so a dataset says where its effort came from."""

NOT_BENCH_CALIBRATED = (
    "Simulated contact force mapped through the Jacobian. NOT the follower's "
    "effort.carriage_n, which carries a ~3.4 N static preload that contact "
    "lowers while collapsing its variance; this channel has no preload and "
    "rises from zero. Do not co-train it against recorded effort without "
    "reconciling that offset."
)


@dataclass(frozen=True)
class ContactEffort:
    """One control step's external efforts, and whether they mean anything."""

    joint_efforts: torch.Tensor
    """(B, n) Nm for revolute joints and N for the prismatic carriage."""

    force_world: torch.Tensor
    """(B, 3) the contact force itself, world frame."""

    simulated: bool
    """False on a kinematic surface: the zeros are "not modelled", not "no force"."""

    calibration: "EffortCalibration | None" = None
    """Set when the carriage channel was put on the bench's scale."""

    def as_metadata(self) -> dict:
        return {
            "model": FORCE_MODEL,
            "simulated": self.simulated,
            "disclaimer": NOT_BENCH_CALIBRATED,
            "carriage_calibration": (
                self.calibration.as_metadata() if self.calibration is not None
                else {"applied": False}
            ),
        }


def contact_force_world(env) -> tuple[torch.Tensor, bool]:
    """(B, 3) tip-substrate contact force, and whether it was simulated at all."""
    batch = env.num_envs
    device = env.device
    if not getattr(env, "surface_has_contact_collision", False):
        return torch.zeros((batch, 3), device=device), False
    force = env.scene.get_pairwise_contact_forces(env.agent.tcp, env.pad)
    return force.to(device), True


def joint_efforts(jacobian: torch.Tensor, force_world: torch.Tensor) -> torch.Tensor:
    """``tau = Jv^T F``: the joint efforts a TCP force is resisted by.

    ``jacobian`` is (B, 6, n) as pytorch_kinematics returns it; only its linear
    rows act on a pure force. The prismatic carriage's column is a unit axis,
    so its entry comes out in newtons while the revolute joints' are newton
    metres -- the same mixed units the follower reports.
    """
    if jacobian.ndim != 3 or jacobian.shape[1] != 6:
        raise ValueError(f"jacobian must be (B, 6, n), got {tuple(jacobian.shape)}")
    if force_world.shape != (jacobian.shape[0], 3):
        raise ValueError(
            f"force must be (B, 3) matching the jacobian batch, got {tuple(force_world.shape)}"
        )
    linear = jacobian[:, :3, :]                       # (B, 3, n)
    return torch.einsum("bij,bi->bj", linear, force_world.to(linear.dtype))


def external_joint_efforts(
    env, ik, q: torch.Tensor, *,
    calibration: "EffortCalibration | None" = None, tool_id: str | None = None,
    in_contact: torch.Tensor | None = None,
) -> ContactEffort:
    """This step's external joint efforts for the whole batch.

    ``q`` is the achieved joint state in the IK chain's own order, so the
    Jacobian is evaluated where the arm actually is rather than where it was
    asked to be.

    With a ``calibration`` whose tool matches, the CARRIAGE column is replaced
    by the bench's measured level for that contact state. The arm joints keep
    their raw ``Jv^T F`` value: the bench measured the carriage, and nothing
    here has a corresponding measurement for the other six.

    ``in_contact`` is the contact state to level-map, and should be the same
    one the rest of the simulator uses -- the interaction band that gates ink.
    Deriving it from ``force > 0`` instead gives the dataset two disagreeing
    notions of contact: the band deposits pigment where a rigid tip registers
    no force, and on a measured shard the two agreed on only 39% of steps, so
    ink appeared while the effort channel read clear. Falling back to the force
    is for callers that have no mask to offer.
    """
    force, simulated = contact_force_world(env)
    tau = joint_efforts(ik.chain.jacobian(q), force)
    applied = None
    carriage = ik.carriage_index
    if (
        calibration is not None and simulated and carriage is not None
        and tool_id is not None and calibration.applies_to(tool_id)
    ):
        contact = (
            in_contact if in_contact is not None
            else torch.linalg.norm(force, dim=-1) > 0
        )
        tau = tau.clone()
        tau[:, carriage] = calibration.carriage_channel(contact.to(torch.bool)).to(tau.dtype)
        applied = calibration
    return ContactEffort(
        joint_efforts=tau, force_world=force, simulated=simulated, calibration=applied,
    )


# --- putting the simulated channel on the bench's scale ------------------------

CALIBRATION_PATH = "config/contact_effort.json"
CALIBRATION_SCHEMA = "tatbot.contact-effort-calibration/1"


@dataclass(frozen=True)
class EffortCalibration:
    """The follower's carriage effort clear of the work and in contact.

    Measured, and bimodal: the axis floats near 3.64 N clear and is pinned near
    3.15 N in contact, with light and firm the same level to four decimals. So
    the map from a simulated contact force is a LEVEL change, not a rescale --
    there is no pressure proportionality on the bench to scale to.
    """

    tool_id: str
    clear_n: float
    contact_n: float
    session: str
    clear_sd: float
    contact_sd: float

    @classmethod
    def load(cls) -> "EffortCalibration":
        import json

        from tatbot_sim.repo import repo_root

        return cls.from_dict(json.loads((repo_root() / CALIBRATION_PATH).read_text()))

    @classmethod
    def from_dict(cls, raw: dict) -> "EffortCalibration":
        """Build from the run's frozen calibration payload."""
        if raw.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError(f"{CALIBRATION_PATH}: schema {raw.get('schema')!r}")
        carriage = raw["carriage_n"]
        return cls(
            tool_id=raw["tool_id"],
            clear_n=float(carriage["clear_median"]),
            contact_n=float(carriage["contact_median"]),
            session=raw["source"]["session"],
            clear_sd=float(carriage["clear_sd"]),
            contact_sd=float(carriage["contact_sd"]),
        )

    def applies_to(self, tool_id: str) -> bool:
        """Another tool has its own seating, so its preload is its own."""
        return tool_id == self.tool_id

    def carriage_channel(self, contact: "torch.Tensor") -> "torch.Tensor":
        """(B,) bool in contact -> (B,) carriage effort on the bench's scale."""
        return torch.where(
            contact,
            torch.full_like(contact, self.contact_n, dtype=torch.float32),
            torch.full_like(contact, self.clear_n, dtype=torch.float32),
        )

    def as_metadata(self) -> dict:
        return {
            "applied": True,
            "tool_id": self.tool_id,
            "session": self.session,
            "clear_n": self.clear_n,
            "contact_n": self.contact_n,
            "reproduces": "the two measured levels",
            "does_not_reproduce": (
                f"within-state spread (bench sd {self.clear_sd:.4f} N clear, "
                f"{self.contact_sd:.4f} N in contact); the simulated channel is "
                "noise-free, so a domain classifier could separate it on that alone"
            ),
            "not_for_scoring": (
                "calibration for co-training only; scoring the contact model this "
                "session fitted against these numbers would score a fit on its own"
            ),
        }
