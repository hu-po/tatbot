from pathlib import Path


def test_estop_release_automatically_resumes_without_floating() -> None:
    source = (
        Path(__file__).parents[2] / "cpp" / "teleop" / "wxai_teleop.cpp"
    ).read_text()
    flow = source.split("auto run_estop_flow", 1)[1].split("StopChoice choice;", 1)[0]

    assert "return StopChoice::resume;" in flow
    assert "automatically resuming from held poses" in flow
    assert "Mode::external_effort" not in flow
    assert "set_arm_external_efforts" not in flow
    assert "POLLIN" not in flow


def test_estop_release_never_moves_the_follower_unannounced() -> None:
    """Releasing the e-stop re-aligns the follower onto the leader, which is
    motion the old zero-step resume never produced. It must stay bounded and
    confirmed: the arms are already held, so waiting costs nothing."""
    source = (
        Path(__file__).parents[2] / "cpp" / "teleop" / "wxai_teleop.cpp"
    ).read_text()
    resume = source.split("if (choice == StopChoice::resume) {", 1)[1].split(
        "stop_baseline = g_stop_signals.load();", 1
    )[0]

    # The e-stop path lowers the confirmation bar to 2 deg, regardless of the
    # operator's --align-confirm-deg.
    assert "released_from_estop ?" in resume
    assert "std::min(opt.align_confirm_deg, 2.0)" in resume
    # An interrupt at that prompt returns to the hold prompt; it must not fall
    # through to the exit path, which idles both arms.
    assert "continue;" in resume
