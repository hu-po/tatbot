import tatbot_motion


def test_motion_yaml_loads():
    motion = tatbot_motion.load_motion()
    assert motion["control_rate_hz"] == 400
