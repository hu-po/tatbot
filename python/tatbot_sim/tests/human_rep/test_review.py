from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from tatbot_sim.human_rep.contracts import ContractError, canonical_digest
from tatbot_sim.human_rep.review import (
    ACTIONS,
    APPROVALS,
    assert_action_enabled,
    make_pending_release_gates,
    validate_release_gates,
)

WHEN = "2026-09-04T16:45:00Z"


def _pending():
    return make_pending_release_gates(created_utc=WHEN, source_sha256="a" * 64)


def _redigest(value):
    value["content_sha256"] = canonical_digest(value)
    return value


def test_pending_packet_has_exactly_four_separate_fail_closed_approvals():
    gates = _pending()
    assert tuple(gates["approvals"]) == APPROVALS
    assert tuple(gates["actions"]) == ACTIONS
    assert all(
        item["status"] == "pending" and item["approved"] is False for item in gates["approvals"].values()
    )
    assert not any(gates["actions"].values())
    for action in ACTIONS:
        with pytest.raises(ContractError) as caught:
            assert_action_enabled(gates, action)
        assert caught.value.code == "execution_binding_mismatch"


def test_action_cannot_be_enabled_without_its_signed_approval():
    gates = deepcopy(_pending())
    gates["actions"]["powered_motion_enabled"] = True
    _redigest(gates)
    with pytest.raises(ContractError) as caught:
        validate_release_gates(gates)
    assert "powered_evaluation is not approved" in caught.value.detail


def test_signed_approval_can_enable_only_its_matching_action():
    gates = deepcopy(_pending())
    approval = gates["approvals"]["inkmap_deployment"]
    approval.update(
        {
            "status": "approved",
            "approved": True,
            "reviewer": "maintainer@example.invalid",
            "reviewed_utc": WHEN,
            "evidence_sha256": "b" * 64,
        }
    )
    gates["actions"]["deployment_enabled"] = True
    _redigest(gates)
    validate_release_gates(gates)
    assert_action_enabled(gates, "deployment_enabled")
    for action in ACTIONS[1:]:
        with pytest.raises(ContractError):
            assert_action_enabled(gates, action)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value["claims"].pop("anatomy_prior"), "execution_binding_mismatch"),
        (
            lambda value: value["approvals"]["public_release"].update({"reviewer": "unsigned"}),
            "execution_binding_mismatch",
        ),
        (
            lambda value: value["rollback"].update({"reintroduces_retired_body": True}),
            "body_model_unsupported",
        ),
        (lambda value: value["provenance"].update({"source_sha256": "bad"}), "wrong_hash"),
    ],
)
def test_gate_reader_rejects_malformed_or_unsafe_packets(mutation, code):
    gates = deepcopy(_pending())
    mutation(gates)
    _redigest(gates)
    with pytest.raises(ContractError) as caught:
        validate_release_gates(gates)
    assert caught.value.code == code


def test_gate_reader_rejects_digest_tampering():
    gates = deepcopy(_pending())
    gates["claims"]["core_nominal_body"]["enabled"] = True
    with pytest.raises(ContractError) as caught:
        validate_release_gates(gates)
    assert caught.value.code == "wrong_hash"


def test_checked_in_release_gates_are_valid_and_physical_actions_stay_disabled():
    """The private file records real approvals (ae3b9d18 approved the inkmap
    deployment and public release; the public export ships a pending, disabled
    template). What must hold whatever has been approved: it validates, an
    action is enabled only by its own approval, an approval is either pending
    with no evidence or approved with reviewer, time and evidence, and nothing
    that touches a body or moves a powered arm is enabled from a config file."""
    from tatbot_sim.human_rep.review import APPROVAL_FOR_ACTION, load_release_gates

    root = Path(__file__).resolve().parents[4]
    gates = load_release_gates(root / "config" / "human-representation" / "release-gates.json")
    for name, item in gates["approvals"].items():
        if item["status"] == "pending":
            assert not item["approved"] and item["evidence_sha256"] is None, name
        else:
            assert item["status"] == "approved" and item["approved"], name
            assert item["evidence_sha256"] and item["reviewer"] and item["reviewed_utc"], name
    for action, enabled in gates["actions"].items():
        assert enabled == gates["approvals"][APPROVAL_FOR_ACTION[action]]["approved"], action
    assert not gates["actions"]["human_contact_enabled"]
    assert not gates["actions"]["powered_motion_enabled"]
