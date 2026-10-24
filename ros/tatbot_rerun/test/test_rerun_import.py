"""The Rerun bridge imports and names Draw's phases and the driver's latch reasons. Its scene models are
tatbot_contracts.ros_scene, tested without ROS in scripts/tests/test_ros_scene.py."""


def test_import_and_names():
    from tatbot_rerun import node

    assert node.PHASES[5] == "DRAW" and node.PHASES[9] == "LATCHED"
    assert node.LATCHES[1] == "estop" and node.LATCHES[11] == "guard_snag"


def test_a_display_handler_takes_keywords_and_swallows_its_error():
    """_on_event calls _follow(run_id, clear_tips=...): a wrapper without **kwargs turned every run start into a
    swallowed TypeError, and the bridge never followed a new draw."""
    from tatbot_rerun import node

    class Bridge:
        failed: set = set()

        def get_logger(self):
            return type("Logger", (), {"warning": lambda self, text: None})()

    @node.shown
    def follow(self, run_id, clear_tips=False):
        if run_id is None:
            raise ValueError("no run")
        return run_id, clear_tips

    bridge = Bridge()
    assert follow(bridge, "a-run", clear_tips=True) == ("a-run", True)
    assert follow(bridge, None) is None and bridge.failed == {"follow"}
