"""Tests for portable, non-secret simulator source provenance."""

from tatbot_sim.repo import _repository_slug


def test_repository_slug_accepts_common_network_remotes():
    assert _repository_slug("git@example.com:robotics/tatbot.git") == "robotics/tatbot"
    assert _repository_slug("https://example.com/robotics/tatbot.git") == "robotics/tatbot"
    assert _repository_slug("ssh://git@example.com/robotics/tatbot") == "robotics/tatbot"


def test_repository_slug_discards_credentials_and_local_paths():
    assert _repository_slug("https://user:secret@example.com/robotics/tatbot.git") == "robotics/tatbot"
    assert _repository_slug("/workspace/tatbot") == "local-checkout"
    assert _repository_slug("file:///workspace/tatbot") == "local-checkout"
    assert _repository_slug("") == "local-checkout"
