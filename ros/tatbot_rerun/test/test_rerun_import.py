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


def test_the_design_map_and_strokes_are_blank_between_draws():
    """2026-11-09: a bridge started a day after a hover heart's run still drew it on the station's design map. A run
    that has ended is cleared once (the hub keeps what an earlier bridge logged); one that draws is logged again."""
    from types import SimpleNamespace

    from tatbot_rerun import node

    class Run:
        def __init__(self, drawing):
            self.drawing = lambda now: drawing

    class Bridge:
        _blank_between_draws = node.RerunBridge._blank_between_draws

        def __init__(self, run):
            logged = []
            self.rr = SimpleNamespace(Clear=lambda recursive: ("clear", recursive), logged=logged,
                                      log=lambda entity, value: logged.append((entity, value)))
            self.run, self.dirty, self.drawing = run, set(), None
            self.frame, self.tip_shown = [[0.0, 0.0]], True

    ended = Bridge(Run(False))
    ended._blank_between_draws(0.0)
    assert ended.rr.logged == [("draw/ros/strokes", ("clear", True)), ("draw/ros/design", ("clear", True))]
    assert ended.frame is None and not ended.tip_shown and ended.drawing is False
    ended._blank_between_draws(1.0)
    assert len(ended.rr.logged) == 2                           # once, not every tick
    ended.run = Run(True)                                      # the next draw starts
    ended._blank_between_draws(2.0)
    assert ended.drawing and ended.dirty == set(node.RUN_LAYERS) and len(ended.rr.logged) == 2
    idle = Bridge(None)                                        # no run at all
    idle._blank_between_draws(0.0)
    assert idle.drawing is False and len(idle.rr.logged) == 2
