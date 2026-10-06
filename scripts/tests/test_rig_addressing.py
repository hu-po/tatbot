"""Every rig-LAN address agrees with config/nodes.json `__rig__`, and the
renumbering path names each file that still disagrees."""
from __future__ import annotations

import ipaddress
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
from tatbot_cli import nodes, rig_addressing  # noqa: E402

RIG_FILES = [
    "config/nodes.json",
    "config/profiles/tatbot.json",
    "config/trossen/leader.yaml",
    "config/trossen/follower.yaml",
    "rust/visiond/config/vision.toml",
]


@pytest.fixture
def rig_copy(tmp_path):
    for rel in RIG_FILES:
        if not (REPO / rel).is_file():
            pytest.skip(f"{rel} absent (public checkout)")
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    return tmp_path


def _set_rig(repo: Path, subnet: str, gateway: str) -> None:
    path = repo / "config/nodes.json"
    data = json.loads(path.read_text())
    data["__rig__"] = {"subnet": subnet, "gateway": gateway}
    path.write_text(json.dumps(data))


def test_the_live_rig_is_consistent():
    if not nodes.rig(REPO):
        pytest.skip("no __rig__ stanza (public checkout)")
    problems, skip = rig_addressing.check(REPO)
    assert skip is None
    assert problems == []


def test_renumbering_names_every_file_that_still_holds_the_old_subnet(rig_copy):
    # RFC 5737 documentation range stands in for the target subnet; the point is
    # that every file carrying the old one is named, whatever the new one is.
    _set_rig(rig_copy, "198.51.100.0/24", "198.51.100.1")
    problems, skip = rig_addressing.check(rig_copy)
    assert skip is None
    joined = "\n".join(problems)
    router = nodes.require_role(nodes.load(rig_copy), "bus-router")
    arm = nodes.require_role(nodes.load(rig_copy), "arm")
    for needle in ("__cameras__.camera1", f"{router}.lan", f"{arm}.ssh_lan", "vision.toml: ntp_server",
                   "tatbot.json: driver.leader_ip", "tatbot.json: driver.follower_ip",
                   "leader.yaml: gateway", "follower.yaml: dns"):
        assert needle in joined, f"{needle} missing from:\n{joined}"


def test_a_camera_that_drifts_between_vision_toml_and_nodes_json_is_named(rig_copy):
    path = rig_copy / "rust/visiond/config/vision.toml"
    cams = nodes.cameras(rig_copy)
    subnet = nodes.rig_subnet(rig_copy)
    stray = str(subnet.network_address + 200)  # still inside the subnet, so only the drift is reported
    text = path.read_text().replace(f'address = "{cams["camera2"]}"', f'address = "{stray}"', 1)
    path.write_text(text)
    problems, _ = rig_addressing.check(rig_copy)
    assert problems == [f"rust/visiond/config/vision.toml: camera2 is {stray}, __cameras__ says {cams['camera2']}"]


def test_controller_yaml_must_match_the_profile_and_the_gateway(rig_copy):
    path = rig_copy / "config/trossen/follower.yaml"
    stray = str(nodes.rig_subnet(rig_copy).network_address + 222)
    path.write_text(path.read_text().replace("manual_ip: ", f"manual_ip: {stray} #", 1))
    problems, _ = rig_addressing.check(rig_copy)
    assert any(f"follower.yaml: manual_ip {stray} but the profile drives the follower at" in p for p in problems)


def test_a_migration_in_progress_accepts_both_subnets_and_lists_what_remains(rig_copy):
    path = rig_copy / "config/nodes.json"
    data = json.loads(path.read_text())
    if data["__rig__"].get("migrating_from"):
        pytest.skip("the live rig is mid-migration; this test needs a single-subnet baseline")
    data["__rig__"] = {"subnet": "198.51.100.0/24", "gateway": "198.51.100.1", "migrating_from": data["__rig__"]["subnet"]}
    path.write_text(json.dumps(data))
    problems, skip = rig_addressing.check(rig_copy)
    assert skip is None and problems == []
    remaining = rig_addressing.remaining(rig_copy)
    assert remaining and all("198.51.100." not in line for line in remaining)
    assert any(line.startswith("__cameras__.") for line in remaining)


def test_no_rig_stanza_is_a_skip_not_a_failure(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/nodes.json").write_text(json.dumps({"host": {"ssh": "u@203.0.113.5", "roles": ["arm"]}}))
    problems, skip = rig_addressing.check(tmp_path)
    assert problems == [] and skip


def test_vision_toml_parser_reads_blocks_not_lines():
    text = ('[sync]\nntp_server = "192.0.2.5"\n\n[[cameras.poe]]\nname = "camera1"\naddress = "192.0.2.91"\n'
            '[cameras.poe.main]\nstream = "main"\n\n[[cameras.poe]]\nname = "camera2"\naddress = "192.0.2.92"\n')
    assert rig_addressing.vision_cameras(text) == {"ntp_server": "192.0.2.5", "camera1": "192.0.2.91", "camera2": "192.0.2.92"}


# `ip -4 -o addr` as a rig node prints it: loopback, a link-local fallback, the
# wired rig address and a Wi-Fi address on another LAN (RFC 5737 ranges, which
# the ipaddress module classes as private like RFC 1918).
IP_ADDR = ("1: lo    inet 127.0.0.1/8 scope host lo\n"
           "2: usb0    inet 169.254.7.7/16 scope link usb0\n"
           "3: eth0    inet 192.0.2.49/24 brd 192.0.2.255 scope global eth0\n"
           "4: wlan0    inet 203.0.113.61/24 brd 203.0.113.255 scope global wlan0\n")


def test_lan_ip_is_the_address_on_the_rig_subnet_never_a_baked_in_prefix():
    assert nodes.pick_lan_ip(IP_ADDR, ipaddress.IPv4Network("192.0.2.0/24")) == "192.0.2.49"
    assert nodes.pick_lan_ip(IP_ADDR, ipaddress.IPv4Network("203.0.113.0/24")) == "203.0.113.61"
    assert nodes.pick_lan_ip(IP_ADDR, ipaddress.IPv4Network("198.51.100.0/24")) is None
    # no subnet described: the first LAN-looking address, never loopback or link-local
    assert nodes.pick_lan_ip(IP_ADDR, None) == "192.0.2.49"
