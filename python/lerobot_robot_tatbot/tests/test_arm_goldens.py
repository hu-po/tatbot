"""The arm goldens (config/trossen/) against the pinned Python SDK.

C++ teleop, arm recovery and the ROS 2 stack's hardware plugin load these
files through the trossen_arm driver; this venv carries its Python binding,
so the schema and gain checks run here. No hardware, no lerobot import.
"""

import json
from pathlib import Path

import pytest
import trossen_arm
import yaml

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "config" / "trossen"


def test_follower_yaml_has_follower_end_effector():
    lead = yaml.safe_load((CONFIG_DIR / "leader.yaml").read_text())
    foll = yaml.safe_load((CONFIG_DIR / "follower.yaml").read_text())
    assert foll["end_effector"]["palm"]["mass"] != lead["end_effector"]["palm"]["mass"]
    # The goldens and the hardware profile must agree about which arm is which:
    # a mismatch here means one of them was edited alone.
    profile = REPO / "config/profiles/tatbot.json"
    if profile.is_file():
        driver = json.loads(profile.read_text()).get("driver", {})
        assert foll["manual_ip"] == driver.get("follower_ip")
        assert lead["manual_ip"] == driver.get("leader_ip")


def test_goldens_match_pinned_sdk_schema():
    """The arm goldens are also loaded by the C++ teleop via
    load_configs_from_file (SDK v1.11.0), whose YAML schema is strict: an
    unknown key fails the whole load. Keep the golden keys to exactly what
    the pinned driver's JointCharacteristic knows — position_offset, absent
    before 1.8.8, broke a 1.8.5 load once already.
    """
    allowed = {f for f in dir(trossen_arm.JointCharacteristic)
               if not f.startswith("_")}
    for name in ("leader.yaml", "follower.yaml"):
        doc = yaml.safe_load((CONFIG_DIR / name).read_text())
        for i, jc in enumerate(doc["joint_characteristics"]):
            unknown = set(jc) - allowed
            assert not unknown, f"{name} joint {i}: {unknown} not in {allowed}"


def test_goldens_carry_the_named_gains_the_tools_set():
    """Firmware 1.10+ boots with Trossen's 2026-03-17 gains (position kp 16, no
    velocity integral), which sagged both tools 30-38 mm on 2026-10-07. Every
    golden load restores 2025-05-09's; a tool that loads no golden sets
    tatbot.yaml's named set, so the two must be the same numbers."""
    named = yaml.safe_load((CONFIG_DIR / "tatbot.yaml").read_text())["follower"]["motor_parameters"]
    assert named == "wxai_v0_20250509"
    standard = getattr(trossen_arm.StandardMotorParameters, named)
    for name in ("leader.yaml", "follower.yaml"):
        doc = yaml.safe_load((CONFIG_DIR / name).read_text())["motor_parameters"]
        assert len(doc) == len(standard) == 7
        for joint, (golden, sdk) in enumerate(zip(doc, standard, strict=True)):
            for mode, gains in sdk.items():
                for loop in ("position", "velocity"):
                    want, got = getattr(gains, loop), golden[mode.name][loop]
                    for term in ("kp", "ki", "kd", "imax"):
                        assert got[term] == pytest.approx(getattr(want, term), rel=1e-6), (name, joint, mode, loop, term)
