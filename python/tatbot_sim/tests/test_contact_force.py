"""External joint efforts from the simulator's contact force.

ext_eff was zero-filled, so the effort half of every state carried no signal
and the co-train had to mask the one channel the real follower populates. It
now comes from SAPIEN's own tip-substrate contact force through Jv^T -- a
measured quantity, not a penalty model with a tuned stiffness.

These tests hold the arithmetic and the boundary: a force at the TCP maps to
the joints that resist it, and a surface with no simulated contact reports that
its zeros are an absence of modelling rather than an absence of force.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tatbot_sim.contact_force import (
    FORCE_MODEL,
    NOT_BENCH_CALIBRATED,
    ContactEffort,
    external_joint_efforts,
    joint_efforts,
)


def test_a_force_along_a_joints_axis_is_resisted_by_that_joint() -> None:
    """A prismatic axis takes exactly the force projected onto it."""
    jac = torch.zeros(1, 6, 2)
    jac[0, :3, 0] = torch.tensor([1.0, 0.0, 0.0])   # slides along world x
    jac[0, :3, 1] = torch.tensor([0.0, 0.0, 1.0])   # slides along world z
    force = torch.tensor([[3.0, 0.0, 5.0]])

    tau = joint_efforts(jac, force)

    assert torch.allclose(tau, torch.tensor([[3.0, 5.0]]))


def test_a_force_perpendicular_to_an_axis_does_not_load_it() -> None:
    jac = torch.zeros(1, 6, 1)
    jac[0, :3, 0] = torch.tensor([0.0, 0.0, 1.0])
    tau = joint_efforts(jac, torch.tensor([[7.0, -2.0, 0.0]]))
    assert torch.allclose(tau, torch.zeros(1, 1))


def test_only_the_linear_rows_act_on_a_pure_force() -> None:
    """A wrench-free contact force must not be read off the angular rows."""
    jac = torch.zeros(1, 6, 1)
    jac[0, 3:, 0] = torch.tensor([9.0, 9.0, 9.0])   # pure rotation
    tau = joint_efforts(jac, torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.allclose(tau, torch.zeros(1, 1))


def test_no_contact_means_no_effort() -> None:
    jac = torch.randn(4, 6, 7)
    tau = joint_efforts(jac, torch.zeros(4, 3))
    assert torch.allclose(tau, torch.zeros(4, 7))


def test_efforts_scale_with_the_force() -> None:
    jac = torch.randn(2, 6, 7)
    force = torch.randn(2, 3)
    assert torch.allclose(
        joint_efforts(jac, force * 3.0), joint_efforts(jac, force) * 3.0, atol=1e-5
    )


@pytest.mark.parametrize("bad", [torch.zeros(1, 3, 7), torch.zeros(6, 7)])
def test_a_malformed_jacobian_refuses(bad) -> None:
    with pytest.raises(ValueError, match="jacobian"):
        joint_efforts(bad, torch.zeros(1, 3))


def test_a_force_that_does_not_match_the_batch_refuses() -> None:
    with pytest.raises(ValueError, match="force"):
        joint_efforts(torch.zeros(2, 6, 7), torch.zeros(3, 3))


class _KinematicEnv:
    """A surface with no collision geometry: no contact pairs exist."""

    num_envs = 2
    device = torch.device("cpu")
    surface_has_contact_collision = False


class _StubIK:
    carriage_index = None   # this stub's chain has no carriage

    class chain:  # noqa: N801 - mirrors the pytorch_kinematics attribute
        @staticmethod
        def jacobian(q):
            return torch.zeros(len(q), 6, q.shape[1])


def test_a_kinematic_surface_reports_that_its_zeros_are_not_measurements() -> None:
    """Curved silicone and posed body patches have no simulated contact.

    Reporting zero effort without saying so would read as "the tip touched
    nothing", when it means "contact was never modelled here".
    """
    effort = external_joint_efforts(_KinematicEnv(), _StubIK(), torch.zeros(2, 7))

    assert effort.simulated is False
    assert torch.allclose(effort.joint_efforts, torch.zeros(2, 7))
    assert torch.allclose(effort.force_world, torch.zeros(2, 3))
    assert effort.as_metadata()["simulated"] is False
    assert effort.as_metadata()["model"] == FORCE_MODEL


def test_the_metadata_refuses_to_be_read_as_the_benchs_channel() -> None:
    """It is the same axis carrying a different quantity.

    The follower's effort.carriage_n sits near +3.64 N clear and +3.15 N in
    contact, its spread collapsing from 0.70 to 0.0004 N. That is a static
    preload contact lowers; this channel has no preload and rises from zero.
    """
    effort = ContactEffort(torch.zeros(1, 7), torch.zeros(1, 3), simulated=True)
    note = effort.as_metadata()["disclaimer"]
    assert note == NOT_BENCH_CALIBRATED
    assert "effort.carriage_n" in note and "preload" in note


def test_the_state_half_stops_being_identically_zero_under_load() -> None:
    """The point of the change: the channel now carries contact."""
    jac = torch.zeros(1, 6, 7)
    jac[0, :3, 6] = torch.tensor([0.0, 0.0, 1.0])   # carriage along world z
    loaded = joint_efforts(jac, torch.tensor([[0.0, 0.0, -0.99]]))
    assert not np.allclose(loaded.numpy(), 0.0)
    assert loaded[0, 6].item() == pytest.approx(-0.99)


# --- putting the simulated channel on the bench's scale -----------------------


def _calibration():
    from tatbot_sim.contact_force import EffortCalibration
    from tatbot_sim.repo import repo_root

    if not (repo_root() / "config/contact_effort.json").is_file():
        pytest.skip("public profile has no measured carriage-effort calibration")
    return EffortCalibration.load()


class _CollidingEnv:
    """A flat qualified substrate, so contact pairs exist."""

    num_envs = 2
    device = torch.device("cpu")
    surface_has_contact_collision = True

    def __init__(self, force):
        self._force = force
        self.scene = self
        self.agent = self
        self.tcp = object()
        self.pad = object()

    def get_pairwise_contact_forces(self, a, b):
        return self._force


class _CarriageIK:
    carriage_index = 6

    class chain:  # noqa: N801 - mirrors pytorch_kinematics
        @staticmethod
        def jacobian(q):
            jac = torch.zeros(len(q), 6, 7)
            jac[:, :3, 6] = torch.tensor([0.0, 0.0, 1.0])
            return jac


def test_the_calibration_is_the_measured_session_not_a_guess() -> None:
    cal = _calibration()
    assert cal.session == "sweep-20260904_143433"
    assert cal.tool_id == "lutin-ballpoint-dot"
    # Contact LOWERS a standing preload; it does not add a growing force.
    assert cal.contact_n < cal.clear_n
    assert cal.clear_n == pytest.approx(3.6401, abs=1e-4)
    assert cal.contact_n == pytest.approx(3.1546, abs=1e-4)


def test_the_carriage_channel_takes_the_benchs_two_levels() -> None:
    cal = _calibration()
    out = cal.carriage_channel(torch.tensor([True, False]))
    assert out[0].item() == pytest.approx(cal.contact_n)
    assert out[1].item() == pytest.approx(cal.clear_n)


def test_a_generated_shard_lands_on_the_benchs_scale() -> None:
    """The point: sim's 0-to-0.13 N channel becomes the bench's 3.15/3.64 N."""
    cal = _calibration()
    touching = torch.tensor([[0.0, 0.0, -0.4], [0.0, 0.0, 0.0]])   # env 0 in contact
    effort = external_joint_efforts(
        _CollidingEnv(touching), _CarriageIK(), torch.zeros(2, 7),
        calibration=cal, tool_id="lutin-ballpoint-dot",
    )
    carriage = effort.joint_efforts[:, 6]
    assert carriage[0].item() == pytest.approx(cal.contact_n)
    assert carriage[1].item() == pytest.approx(cal.clear_n)
    assert effort.calibration is cal


def test_another_tool_does_not_borrow_the_ballpoints_preload() -> None:
    """Its seating is its own, so the channel stays raw and says so."""
    touching = torch.tensor([[0.0, 0.0, -0.4], [0.0, 0.0, 0.0]])
    effort = external_joint_efforts(
        _CollidingEnv(touching), _CarriageIK(), torch.zeros(2, 7),
        calibration=_calibration(), tool_id="picosecond-laser-pen",
    )
    assert effort.calibration is None
    raw = effort.joint_efforts[:, 6]
    assert raw[0].item() == pytest.approx(-0.4) and raw[1].item() == pytest.approx(0.0)
    assert effort.as_metadata()["carriage_calibration"] == {"applied": False}


def test_a_kinematic_surface_is_not_calibrated_either() -> None:
    """There is no contact to put on a scale."""
    effort = external_joint_efforts(
        _KinematicEnv(), _CarriageIK(), torch.zeros(2, 7),
        calibration=_calibration(), tool_id="lutin-ballpoint-dot",
    )
    assert effort.calibration is None and effort.simulated is False


def test_the_metadata_states_what_the_calibration_does_not_reproduce() -> None:
    meta = _calibration().as_metadata()
    assert meta["applied"] is True
    assert "spread" in meta["does_not_reproduce"]
    assert "co-training" in meta["not_for_scoring"]


def test_the_arm_joints_keep_their_raw_efforts() -> None:
    """The bench measured the carriage; nothing measured the other six."""
    cal = _calibration()
    jac = torch.zeros(1, 6, 7)
    jac[:, :3, 0] = torch.tensor([0.0, 0.0, 1.0])   # joint 0 also along z
    jac[:, :3, 6] = torch.tensor([0.0, 0.0, 1.0])

    class _IK:
        carriage_index = 6

        class chain:  # noqa: N801
            @staticmethod
            def jacobian(q):
                return jac

    effort = external_joint_efforts(
        _CollidingEnv(torch.tensor([[0.0, 0.0, -0.4]])), _IK(), torch.zeros(1, 7),
        calibration=cal, tool_id="lutin-ballpoint-dot",
    )
    assert effort.joint_efforts[0, 0].item() == pytest.approx(-0.4)   # untouched
    assert effort.joint_efforts[0, 6].item() == pytest.approx(cal.contact_n)


def test_the_effort_channel_uses_the_same_contact_gate_as_the_ink() -> None:
    """Otherwise a shard shows pigment while its effort channel reads clear.

    Measured 2026-09-08 before this: the interaction band and a physics
    force > 0 agreed on only 39% of steps, because the band deposits pigment
    where a rigid tip registers no force.
    """
    cal = _calibration()
    # No physics force at all, but the band says both envs are touching.
    effort = external_joint_efforts(
        _CollidingEnv(torch.zeros(2, 3)), _CarriageIK(), torch.zeros(2, 7),
        calibration=cal, tool_id="lutin-ballpoint-dot",
        in_contact=torch.tensor([True, False]),
    )
    carriage = effort.joint_efforts[:, 6]
    assert carriage[0].item() == pytest.approx(cal.contact_n), (
        "a banded contact must read the bench's contact level even with no force"
    )
    assert carriage[1].item() == pytest.approx(cal.clear_n)


def test_without_a_mask_it_falls_back_to_the_force() -> None:
    cal = _calibration()
    effort = external_joint_efforts(
        _CollidingEnv(torch.tensor([[0.0, 0.0, -0.4], [0.0, 0.0, 0.0]])),
        _CarriageIK(), torch.zeros(2, 7), calibration=cal, tool_id="lutin-ballpoint-dot",
    )
    carriage = effort.joint_efforts[:, 6]
    assert carriage[0].item() == pytest.approx(cal.contact_n)
    assert carriage[1].item() == pytest.approx(cal.clear_n)
