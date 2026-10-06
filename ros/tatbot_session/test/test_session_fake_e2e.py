"""End to end on TatbotArm with its fake SDK (every driver interlock live, no network, no lock): the e-stop over UDP
from an emulated relay, the no-step unlatch, the three tip-lag page touches against a fake paper plane, a SIGTERM
that cancels a touch mid-descent, a mid-stroke press -> continue refused while pressed -> release -> continue
resumes at the arc, press -> release -> land to idle, and the landed arm refusing every motion goal. Skips unless
the installed TatbotArm is the full driver (its e-stop reader header is installed) and the description renders
fake hardware."""
from __future__ import annotations

import itertools
import signal
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import session_graph as sg

pytestmark = pytest.mark.skipif(not sg.ZENOHD.exists(), reason="needs ROS 2 Jazzy with rmw_zenoh_cpp")


class Pico:
    """EST1 frames at 100 Hz from 127.0.0.1, like the relay forwarding a Pico: state 1 released, 0 pressed."""

    def __init__(self, port: int):
        self.port, self.state, self.running = port, 1, True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        seq = 0
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            while self.running:
                seq += 1
                sock.sendto(f"EST1 {seq} {self.state}\n".encode(), ("127.0.0.1", self.port))
                time.sleep(0.01)


def _full_driver() -> bool:
    """The Phase-0 stub TatbotArm has no e-stop source, latching or fake paper; the full driver installs
    tatbot_hardware/estop.hpp beside its plugin."""
    from ament_index_python.packages import PackageNotFoundError, get_package_prefix

    try:
        include = Path(get_package_prefix("tatbot_hardware")) / "include"
    except PackageNotFoundError:
        return False
    return any(include.rglob("estop.hpp"))


def _fake_urdf(stack: dict) -> str:
    from tatbot_description import robot_description

    if not _full_driver():
        pytest.skip("the installed tatbot_hardware is the stub driver (no e-stop reader, no latch)")

    # the fake paper lies where the fixed page says and gives like the arm and EE mount
    stack["fake"] = {"page_z": stack["page"]["fixed"]["right"]["xyz"][2], "page_stiffness_n_m": 1500.0}
    try:
        return robot_description(sg.repo(), arms=("right",), hardware="fake", ros2_control=stack)
    except NotImplementedError:
        pytest.skip("tatbot_description renders no fake-hardware <ros2_control> yet")


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fake_e2e")
    port = sg.free_port(socket.SOCK_DGRAM)
    stack = sg.base_stack(hardware="fake", flight_path=str(tmp))
    stack["estop"].update(source="udp", udp_port=port, relay_addr="127.0.0.1")
    urdf = _fake_urdf(stack)
    pico = Pico(port)
    g = sg.Graph(tmp)
    g.pico, g.page_z, g.page_xyz = pico, stack["fake"]["page_z"], stack["page"]["fixed"]["right"]["xyz"]
    try:
        yield g.up(urdf, stack)
    finally:
        g.stop()
        pico.running = False


def _status(graph) -> dict:
    return sg.last_json(graph.client("status", "--json").stdout)["safety"]["right"]


def _decide(graph, decision: str) -> dict:
    return sg.last_json(graph.client("decide", "right", decision).stdout)


def test_startup_latch_clears_with_continue_without_a_step(graph):
    # Whether the driver comes up latched (e-stop stale before the first frame, or a first command inside
    # the feedback warm-up) depends on start-up timing; either way `continue` leaves it unlatched.
    latched = _status(graph)["latched"]
    decision = _decide(graph, "continue")
    assert decision["accepted"] is latched, decision
    right = _status(graph)
    assert right["latched"] is False and right["estop_ok"] is True and right["estop_source"] == 2


def _touch_timeout_s() -> float:
    """The page touches on the configured motion (motion.yaml touch). Every trip on the fake page meets the others,
    so three descents confirm the first point and two each of the other two: seven. Each plans its whole guarded leg,
    touch.max_travel_m at touch.slow_m_s, before it moves; at 0.2 mm/s a descent and its lift took 45-85 s of that
    125 s leg on the ros node, 387 s in all (2026-10-04). Each gets the leg's whole duration."""
    import tatbot_motion

    touch = tatbot_motion.load_motion()["touch"]
    return 7 * float(touch["max_travel_m"]) / float(touch["slow_m_s"])


def test_three_touches_find_the_fake_page(graph):
    proc = graph.client("touch", "--arm", "right", timeout=_touch_timeout_s())
    result = sg.last_json(proc.stdout)
    assert proc.returncode == 0 and result["tripped"], proc.stdout
    normal, d = np.array(result["plane"][:3]), result["plane"][3]
    assert normal[2] > 0.9999 and abs(-d - graph.page_z) < 0.0003
    assert all(abs(c[2] - graph.page_z) < 0.0003 for c in result["contacts"])
    assert _status(graph)["latched"] is False


def test_sigterm_mid_touch_cancels_its_goal(graph):
    """A signal to a client in a page touch's guarded descent, where nothing calls the client back until the trip:
    one that waited for a callback ran its handler only then (2026-10-04). It cancels within a wake now, and the goal
    ends with the guard off."""
    from tatbot_description import names

    log = "touch-sigterm.log"
    proc = graph.client_bg("touch", "--arm", "right", log=log)
    try:
        deadline = time.monotonic() + 300
        while _status(graph)["guard_mode"] != names.GUARD_TIP_LAG:
            assert proc.poll() is None and time.monotonic() < deadline, (graph.tmp / log).read_text()
        proc.send_signal(signal.SIGTERM)
        assert graph.wait_for(log, "cancelling; the arm holds", 2), (graph.tmp / log).read_text()
        assert proc.wait(timeout=10) == 1
    finally:
        graph.end(proc)
    assert sg.last_json((graph.tmp / log).read_text())["message"] == "cancelled"
    right = _status(graph)
    assert right["guard_mode"] == names.GUARD_NONE and right["latched"] is False


def _press_mid_stroke(graph, log: str):
    program = sg.square_program(graph.tmp, name=log.split(".")[0], speed_m_s=0.005)
    proc = graph.client_bg("draw", str(program), log=log)
    assert graph.wait_for(log, " draw", 90)
    time.sleep(3.0)   # the draw phase first presses pen.press.lift_m into the page (~1.6 s at 3 mm/s^2)
    graph.pico.state = 0
    assert graph.wait_for(log, " latched ", 5)
    return proc


def test_estop_mid_stroke_then_continue_resumes_at_the_arc(graph):
    proc = _press_mid_stroke(graph, "estop-continue.log")
    try:
        refused = _decide(graph, "continue")
        assert refused["accepted"] is False and "e-stop" in refused["reason"]
        graph.pico.state = 1
        assert graph.wait_for("estop-continue.log", " released ", 5)
        assert _decide(graph, "continue")["accepted"] is True
        assert proc.wait(timeout=120) == 0
    finally:
        graph.pico.state = 1
        graph.end(proc)
    result = sg.last_json((graph.tmp / "estop-continue.log").read_text())
    rows = [r for r in sg.ledger(result["run_dir"]) if r["event"] in ("sent", "aborted", "done") and r["op"][0] == "s"]
    assert [r["event"] for r in rows] == ["sent", "aborted", "sent", "done"]
    assert rows[1]["arc_m"] > 0.001 and rows[1]["reason"] == "latched: estop"
    assert rows[2]["arc_m"] == rows[1]["arc_m"]


def _flight(graph) -> np.ndarray:
    """Every record of the driver's 400 Hz flight log so far (ros/README.md "Flight log")."""
    raw = (graph.tmp / "right-flight.bin").read_bytes()
    header, _, body = raw.partition(b"\n")
    dtype = np.dtype([("t", "<f8"), *((name, "<f4", 7) for name in ("q", "qd", "effort", "cmd_q", "sent_q", "sent_qd")),
                      ("estop_age_s", "<f4"), ("rt_period_max_ms", "<f4"), ("estop_ok", "u1"), ("latched", "u1"),
                      ("latch_reason", "u1"), ("phase", "u1")])
    assert int(header.split()[-1]) == dtype.itemsize
    return np.frombuffer(body[: len(body) // dtype.itemsize * dtype.itemsize], dtype=dtype)


def test_numpad_trims_mid_stroke_replace_the_goal_without_a_step(graph):
    """Presses on /tatbot/keys/right while a stroke draws: each a pen_trim ledger row, the running goal replaced
    ahead of now on its own time base, so the driver's commanded tip steps 0.1 mm per press along the page normal
    and nowhere jumps (no restart from the lagging measured pose, no time shift along the path); one goal."""
    import tatbot_motion
    from tatbot_description import names

    before = len(_flight(graph))
    program = sg.square_program(graph.tmp, name="trim", speed_m_s=0.003)
    proc = graph.client_bg("draw", str(program), log="trim.log")
    try:
        assert graph.wait_for("trim.log", " draw", 90)
        for key in ("down", "down", "up"):
            pub = graph.run(["ros2", "topic", "pub", "--once", "-w", "1", names.keys_topic("right"), "std_msgs/msg/String",
                             f"{{data: {key}}}"])
            assert pub.returncode == 0, pub.stdout + pub.stderr
        assert proc.wait(timeout=180) == 0
    finally:
        graph.end(proc)
    rows = sg.ledger(sg.last_json((graph.tmp / "trim.log").read_text())["run_dir"])
    assert [r["trim_m"] for r in rows if r["event"] == "pen_trim"] == pytest.approx([-0.0001, -0.0002, -0.0001])
    assert [r["event"] for r in rows if r["event"] in ("sent", "aborted", "done") and r["op"][0] == "s"] == ["sent", "done"]
    rec = _flight(graph)[before:]
    kin = tatbot_motion.Kinematics.from_repo(arm="right")
    tip = np.array([kin.fk(q)[:3, 3] for q in rec["cmd_q"][rec["phase"] == 1].astype(float)])
    ride = graph.page_z + next(r for r in rows if r["event"] == "page")["page"]["pen"]["height_m"]
    down = tip[np.abs(tip[:, 2] - ride) < 0.0005]
    level = [next((trim for trim in (0.0, -0.0001, -0.0002) if abs(z - ride - trim) < 1.5e-5), None) for z in down[:, 2]]
    runs = [(trim, len(list(run))) for trim, run in itertools.groupby(level)]
    assert [trim for trim, n in runs if trim is not None and n > 100] == [0.0, -0.0001, -0.0002, -0.0001], runs
    smooth = np.stack([np.convolve(down[:, k], np.ones(5) / 5, mode="valid") for k in range(3)], axis=1)
    jump = float(np.linalg.norm(down[2:-2] - smooth, axis=1).max())
    assert jump < 1.5e-5, jump


def test_estop_then_land_goes_idle(graph):
    proc = _press_mid_stroke(graph, "estop-land.log")
    try:
        graph.pico.state = 1
        assert graph.wait_for("estop-land.log", " released ", 5)
        assert _decide(graph, "land")["accepted"] is True
        assert proc.wait(timeout=120) == 1  # the draw ends landed, not complete
    finally:
        graph.end(proc)
    log = (graph.tmp / "estop-land.log").read_text()
    assert " landed " in log and "clear before landing" in log  # the pen was down: lifted, then landed
    right = _status(graph)
    assert right["landed"] is True and right["latched"] is False


def _runs(graph) -> set[str]:
    return {run.name for workflow in ("ros-touch", "ros-draw") for run in (graph.tmp / "logs" / workflow).glob("*")}


def test_a_landed_arm_refuses_motion_until_it_is_woken(graph):
    """The driver keeps a landed arm idle until it is woken, and a Touch move to a hold over the page was
    reported a success with the arm at rest (2026-09-29). A Touch and a Draw are now refused before any run opens,
    and a jog before its goal is sent, each saying how to wake it."""
    if not _status(graph)["landed"]:   # test_estop_then_land_goes_idle leaves it landed
        assert graph.client("land", "--arm", "right", timeout=150).returncode == 0
    runs = _runs(graph)
    # a registration hold of 2026-09-29, 120 mm over the page: from rest a joint move reaches it
    move = graph.client("touch", "--arm", "right", "--move", "--start", "0.2767", "0.0233", "0.123",
                        "--rpy", "-2.9391", "-0.1671", "-0.8897")
    assert move.returncode == 1 and "until it is woken" in move.stderr, move.stdout + move.stderr
    draw = graph.client("draw", str(sg.square_program(graph.tmp, name="landed")), "--arm", "right")
    assert draw.returncode == 1 and "until it is woken" in draw.stderr, draw.stdout + draw.stderr
    jog = graph.client("jog", "--arm", "right", "--joint", "0", "--delta", "0.05")
    result = sg.last_json(jog.stdout)
    assert jog.returncode == 1 and "until it is woken" in result["text"], jog.stdout + jog.stderr
    assert abs(result["moved"]) < 1e-9 and result["q_after"] == result["q_before"]
    assert _runs(graph) == runs and _status(graph)["landed"] is True
