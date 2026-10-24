"""The session client's exit status is its verb's: a middleware that fails to close after the verb finished does not
turn a done jog or land into a failure."""
from types import SimpleNamespace

import pytest
from tatbot_session import client


@pytest.mark.parametrize("rc", [0, 1])
def test_a_verb_s_exit_status_stands_when_the_middleware_fails_to_close(monkeypatch, capsys, rc):
    """2026-10-21 joint_3 bench: a jog reported ok, then rclpy.try_shutdown raised "Failed to close the session(Error
    code: -128 )" (rmw_zenoh's zenoh session close timing out) and the client exited 1, so the bench took the done
    jog for a failed one and aborted."""
    closed = []

    def fail():
        closed.append(True)
        raise RuntimeError("Failed to close the session(Error code: -128 )")

    monkeypatch.setattr(client.rclpy, "init", lambda **kw: None)
    monkeypatch.setattr(client.rclpy, "create_node", lambda name: SimpleNamespace(destroy_node=lambda: None))
    monkeypatch.setattr(client.rclpy, "try_shutdown", fail)
    monkeypatch.setattr(client.signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(client, "_status", lambda node, args: rc)
    assert client.main(["client", "status"]) == rc and closed
    assert "Failed to close the session" in capsys.readouterr().err
