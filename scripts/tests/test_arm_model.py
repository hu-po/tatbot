"""The arm model as data: scripts/lib/arm_model.py and config/arm-models/."""
import json
from dataclasses import replace
from pathlib import Path

import arm_model
import kinematic_calibration
import pytest
from tool_spec import parse_simple_yaml

REPO = Path(__file__).resolve().parents[2]


def _wxai() -> dict:
    return parse_simple_yaml((REPO / "config" / "arm-models" / "trossen-wxai.yaml").read_text())


def test_every_physical_arm_names_a_model_that_loads():
    arms = json.loads((REPO / "config" / "arms.json").read_text())
    assert arms["schema"] == "tatbot.arms/2"
    for arm, entry in arms["arms"].items():
        model = arm_model.for_arm(arm, REPO)
        assert model.id == entry["model"] and (REPO / model.urdf).is_file()


def test_the_wxai_keeps_the_joint_order_the_controllers_use():
    model = arm_model.load("trossen-wxai", REPO)
    assert model.joint_names("right") == (*(f"right/joint_{i}" for i in range(6)), "right/left_carriage_joint")
    assert model.joint_names(None)[0] == "joint_0"
    assert model.n == 7 and model.arm == (0, 1, 2, 3, 4, 5) and model.carriage == 6 and model.wrist == (3, 4, 5)
    assert model.arm_index == slice(0, 6) and model.joints[model.carriage].kind == "prismatic"
    assert model.rest[5] == pytest.approx(1.5707963267948966) and len(model.wrist_range_rad) == 3
    assert model.command_delay_s == 0.0   # the guards' thresholds were set on the WXAI's lag as it is


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(version=2), "not tatbot.arm-model/1"),
    (lambda d: d["joints"].append(dict(d["joints"][0])), "repeat"),
    (lambda d: d["joints"][0].update(kind="spherical"), "unknown"),
    (lambda d: d["joints"][0].update(kind="prismatic"), "only the carriage is prismatic"),
    (lambda d: d["joints"][6].update(kind="revolute"), "only the carriage is prismatic"),
    (lambda d: d["rest"].pop(), "one value per joint"),
    (lambda d: d["wrist_range_rad"].pop(), "one range per wrist joint"),
    (lambda d: d["wrist_range_rad"].__setitem__(0, [0.3, -0.3]), "empty"),
    (lambda d: d.update(calibrated_offsets=[6]), "revolute joints"),
    (lambda d: d.update(command_delay_s=-0.01), "a delay"),
    (lambda d: d.update(controller="kuka"), "controller 'kuka' is none of trossen, xarm"),
    (lambda d: d.update(collision_sensitivity=6), "0 \\(off\\) to 5"),
])
def test_a_malformed_model_says_what_is_wrong(change, message):
    document = _wxai()
    change(document)
    with pytest.raises(ValueError, match=message):
        arm_model.parse(document)


def _seven_revolute() -> arm_model.ArmModel:
    document = _wxai()
    document["joints"] = [{"name": f"joint{i}", "kind": "revolute", "role": role} for i, role in
                          enumerate(("base", "shoulder", "base", "elbow", "wrist", "wrist", "wrist"), start=1)]
    document["view_weights"] = [1.0] * 7
    document["posture"] = {"gain_s": 1.0, "target": [0.0] * 7, "weights": [1.0] * 7, "max_rad_s": 0.25}
    return arm_model.parse(document)


def test_a_chain_longer_than_the_task_needs_a_posture_for_its_self_motion():
    document = _wxai()
    document["joints"] = [{"name": f"joint{i}", "kind": "revolute", "role": role} for i, role in
                          enumerate(("base", "shoulder", "shoulder", "elbow", "wrist", "wrist", "wrist"), start=1)]
    document["view_weights"] = [1.0] * 7
    with pytest.raises(ValueError, match="needs a posture"):
        arm_model.parse(document)
    document["posture"] = {"gain_s": 1.0, "target": [0.0] * 6, "weights": [1.0] * 7, "max_rad_s": 0.25}
    with pytest.raises(ValueError, match="a target and non-negative weights per joint"):
        arm_model.parse(document)


def test_an_arm_without_a_carriage_has_every_joint_in_its_chain():
    model = _seven_revolute()
    assert model.carriage is None and model.arm == tuple(range(7)) and model.wrist == (4, 5, 6)
    assert model.arm_index == slice(0, 7) and model.joint_names("xarm")[0] == "xarm/joint1"


def test_offsets_are_one_per_joint_and_nonzero_only_where_the_model_calibrates():
    wxai = arm_model.load("trossen-wxai", REPO)
    assert kinematic_calibration.validate_offsets(None, wxai) == [0.0] * 7
    assert kinematic_calibration.validate_offsets([0, 0.1, -0.2, 0, 0.3, 0, 0], wxai)[4] == 0.3
    with pytest.raises(ValueError, match="nonzero only on joints 1, 2, 3, 4"):
        kinematic_calibration.validate_offsets([0, 0, 0, 0, 0, 0.1, 0], wxai)
    with pytest.raises(ValueError, match="7 finite values"):
        kinematic_calibration.validate_offsets([0.0] * 6, wxai)
    # the last joint is a carriage on the WXAI, a revolute one here: its offset is the model's to allow
    calibrates_last = replace(_seven_revolute(), calibrated_offsets=(6,))
    assert kinematic_calibration.validate_offsets([0, 0, 0, 0, 0, 0, 0.1], calibrates_last)[6] == 0.1


def test_the_xarm_is_seven_revolute_joints_with_its_wrist_last_and_no_carriage():
    model = arm_model.load("ufactory-xarm7", REPO)
    assert model.joint_names("xarm") == tuple(f"xarm/joint{i}" for i in range(1, 8))
    assert model.carriage is None and model.arm_index == slice(0, 7) and model.wrist == (4, 5, 6)
    assert model.calibrated_offsets == () and model.command_delay_s == pytest.approx(0.095)
    assert model.posture.target == model.rest and model.posture.gain_s > 0
    assert kinematic_calibration.validate_offsets(None, model) == [0.0] * 7
    with pytest.raises(ValueError, match="nonzero only on joints"):
        kinematic_calibration.validate_offsets([0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], model)
