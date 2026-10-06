"""Starting the generator on demand, and refusing to when the GPU is spoken for.

Nothing here opens a socket or starts a process: `health` and `start` are the
two seams, and every test replaces them.
"""
from __future__ import annotations

import pytest
from tatbot_sim.inkmap import inkgen_client as client
from tatbot_sim.inkmap.design_build import DesignBuildError

URL = "http://127.0.0.1:8600"
UP = {"ok": True, "model": "Tongyi-MAI/Z-Image-Turbo", "idle_stop_in_s": 720.0}


FLEET = {"gpu-node": {"roles": ["inkgen"], "ssh": "user@192.0.2.9"}}
NO_ROLE = {"arm-node": {"roles": ["arm"]}}


@pytest.fixture
def seam(monkeypatch):
    """Replace the health probe, the start path and the role map; record asks."""
    state = {"health": None, "starts": 0}

    def health(url, timeout_s=client.HEALTH_TIMEOUT_S):
        return state["health"]

    def start(*, timeout_s=client.START_TIMEOUT_S, check_vram=True):
        state["starts"] += 1
        state["health"] = UP

    monkeypatch.setattr(client, "health", health)
    monkeypatch.setattr(client, "start", start)
    monkeypatch.setattr(client, "fleet_endpoint", lambda node_config=None: URL)
    return state


def test_a_running_generator_is_used_and_not_restarted(seam):
    seam["health"] = UP
    url, document = client.ensure_running(URL)
    assert (url, document) == (URL, UP)
    assert seam["starts"] == 0


def test_a_stopped_fleet_generator_is_started_once(seam):
    url, document = client.ensure_running(URL)
    assert seam["starts"] == 1 and document == UP and url == URL


def test_no_autostart_refuses_instead_of_starting(seam):
    with pytest.raises(client.InkgenUnreachableError, match="no-autostart"):
        client.ensure_running(URL, autostart=False)
    assert seam["starts"] == 0


def test_the_hosted_space_is_never_ours_to_start(seam):
    with pytest.raises(client.InkgenUnreachableError, match="not ours to start"):
        client.ensure_running(client.SPACE_URL)
    assert seam["starts"] == 0


def test_an_address_we_do_not_manage_is_never_started(seam):
    """A URL a caller typed is somebody else's server, however dead it looks."""
    with pytest.raises(client.InkgenUnreachableError, match="not ours to start"):
        client.ensure_running("http://192.0.2.7:8600")
    assert seam["starts"] == 0


def test_a_start_that_leaves_nothing_answering_is_an_error(monkeypatch):
    monkeypatch.setattr(client, "health", lambda url, timeout_s=5.0: None)
    monkeypatch.setattr(client, "start", lambda **_: None)
    monkeypatch.setattr(client, "fleet_endpoint", lambda node_config=None: URL)
    with pytest.raises(client.InkgenUnreachableError, match="still does not answer"):
        client.ensure_running(URL)


def test_backend_kinds_are_named_before_anything_is_contacted():
    assert client.resolve_backend(node_config=FLEET).kind == "fleet"
    assert client.resolve_backend(space=True, node_config=FLEET).kind == "space"
    assert client.resolve_backend("http://192.0.2.7:8600", node_config=FLEET).kind == "endpoint"
    assert client.resolve_backend("http://192.0.2.9:8600/", node_config=FLEET).kind == "fleet"
    assert client.resolve_backend(client.SPACE_URL, node_config=FLEET).kind == "space"


def test_only_the_fleet_worker_is_ours_to_manage():
    assert client.resolve_backend(node_config=FLEET).managed is True
    assert client.resolve_backend("http://192.0.2.7:8600", node_config=FLEET).managed is False
    assert client.resolve_backend(space=True, node_config=FLEET).managed is False


def test_interactive_use_may_fall_back_to_the_public_space():
    assert client.resolve_backend(node_config=NO_ROLE).kind == "space"


def test_bulk_work_refuses_instead_of_reaching_for_the_public_space():
    with pytest.raises(client.InkgenBackendError, match="never falls back"):
        client.resolve_backend(node_config=NO_ROLE, require_configured=True)


def test_a_url_without_a_scheme_is_refused():
    with pytest.raises(client.InkgenBackendError, match="http://"):
        client.resolve_backend("gpu-node:8600", node_config=FLEET)


def test_space_and_an_explicit_address_are_not_both_choosable():
    with pytest.raises(client.InkgenBackendError, match="not both"):
        client.resolve_backend("http://192.0.2.7:8600", space=True, node_config=FLEET)


def test_a_busy_gpu_refuses_the_start_and_names_the_holder(monkeypatch):
    monkeypatch.setattr(client, "free_vram_mb", lambda: 2048)
    monkeypatch.setattr(client, "gpu_holders", lambda: ["4242, python, 21000 MiB"])
    with pytest.raises(client.InkgenBusyError) as excinfo:
        client.start()
    message = str(excinfo.value)
    assert "2048 MB" in message and str(client.VRAM_MIN_MB) in message and "4242" in message


def test_enough_free_memory_proceeds_to_the_start_path(monkeypatch):
    monkeypatch.setattr(client, "free_vram_mb", lambda: client.VRAM_MIN_MB + 1)
    calls = []

    class Result:
        returncode = 0

    monkeypatch.setattr(client.subprocess, "run", lambda argv, **kw: calls.append(argv) or Result())
    client.start()
    # One start path: the CLI's own idempotent verb, which hops and waits itself.
    assert calls and calls[0][-4:] == ["inkgen", "ctl", "--", "start"]


def test_a_card_we_cannot_query_does_not_block_a_start(monkeypatch):
    monkeypatch.setattr(client, "free_vram_mb", lambda: None)

    class Result:
        returncode = 0

    monkeypatch.setattr(client.subprocess, "run", lambda argv, **kw: Result())
    client.start()  # no nvidia-smi is not evidence of a busy GPU


def test_a_failed_start_says_where_to_look(monkeypatch):
    monkeypatch.setattr(client, "free_vram_mb", lambda: None)

    class Result:
        returncode = 5

    monkeypatch.setattr(client.subprocess, "run", lambda argv, **kw: Result())
    with pytest.raises(client.InkgenUnreachableError, match="logs"):
        client.start()


@pytest.mark.parametrize("subject", ["", "   ", "x" * 121])
def test_an_unusable_subject_never_reaches_the_generator(subject, seam):
    seam["health"] = UP
    with pytest.raises(DesignBuildError, match="1-120 characters"):
        client.generate(subject)


def test_holders_are_one_readable_line_each_largest_first(monkeypatch):
    """A browser's GPU process argv runs to kilobytes; a refusal must stay readable."""
    csv = ("10050, /opt/google/chrome/chrome --type=gpu-process --ozone-platform=wayland "
           + "--flag=" + "x" * 2000 + ", 110 MiB\n"
           "1248295, /usr/lib/claude-desktop/claude-desktop --type=gpu-process, 300 MiB\n")

    class Result:
        returncode = 0
        stdout = csv

    monkeypatch.setattr(client.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(client.subprocess, "run", lambda argv, **kw: Result())
    holders = client.gpu_holders()
    assert holders == ["claude-desktop (pid 1248295, 300 MiB)", "chrome (pid 10050, 110 MiB)"]
    assert all(len(item) < 80 for item in holders)


def test_a_missing_nvidia_smi_names_no_holders(monkeypatch):
    monkeypatch.setattr(client.shutil, "which", lambda name: None)
    assert client.gpu_holders() == [] and client.free_vram_mb() is None


def test_the_fleet_url_comes_from_the_inkgen_role():
    url = client.fleet_url()
    assert url.startswith("http://") or url == client.SPACE_URL


def test_seed_zero_reaches_the_generator_as_a_seed(monkeypatch, seam):
    """`if seed:` would have dropped it and randomized the image instead."""
    seam["health"] = UP
    sent = {}

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _n):
            import base64
            png = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
            return ('{"png_base64": "%s", "seed": 0, "prompt": "p", "model": "m"}' % png).encode()

    def urlopen(request, timeout=None):
        import json as _json
        sent.update(_json.loads(request.data))
        return Reply()

    monkeypatch.setattr(client.urllib.request, "urlopen", urlopen)
    reply = client.generate("a swallow", seed=0)
    assert sent["seed"] == 0
    assert reply["seed"] == 0 and reply["backend"] == "fleet"
