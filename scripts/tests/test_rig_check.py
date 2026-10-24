"""`tatbot rig check`: what each demo host is expected to have, and how a probe is judged. No ssh."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/lib"))

from tatbot_cli import rig_check  # noqa: E402

NMAP = {
    "__rig__": {"subnet": "192.0.2.0/24", "gateway": "192.0.2.1"},
    "ros": {"ssh": "u@ros", "lan": "192.0.2.5", "off": True, "roles": ["ros", "overhead-depth"],
            "services": [{"unit": "tatbot-visiond-d555.service"}],
            "hardware": [{"name": "numpad", "dev": "/dev/tatbot-keypad"}]},
    "station": {"ssh": "u@st", "off": True, "roles": ["train"],
                "hardware": [{"name": "touchscreen", "usb": "28E9:028A", "part": "station"},
                             {"name": "model", "user_unit": "tatbot-qwen.service", "part": "station"}]},
    "desk": {"ssh": "u@desk", "roles": ["operator"]},
}


def test_only_demo_case_hosts_are_checked_and_parts_follow_roles():
    exps = rig_check.expectations(REPO, NMAP)
    assert set(exps) == {"ros", "station"}
    assert exps["ros"]["part"] == "pink drawing" and exps["ros"]["jumbo"] is True
    assert exps["station"]["part"] == "station"
    assert exps["station"]["usb"] == [{"name": "touchscreen", "part": "station", "id": "28e9:028a"}]
    assert [u["unit"] for u in exps["ros"]["units"]] == ["tatbot-visiond-d555.service"]


def test_every_usb_wrist_camera_is_expected_on_its_owner(tmp_path):
    vision = tmp_path / "rust/visiond/config/vision.toml"
    vision.parent.mkdir(parents=True)
    vision.write_text('''[[cameras.realsense]]
name = "test-left"
serial = "left-serial"
arm = "left"
owner_role = "train"

[[cameras.realsense]]
name = "test-right"
serial = "right-serial"
arm = "right"
owner_role = "ros"

[[cameras.realsense]]
name = "network-camera"
serial = "network-serial"
arm = "right"
owner_role = "ros"
transport = "dds"
''')
    cams = rig_check.realsense_cameras(vision.read_text())
    assert {c["name"] for c in cams} == {"test-left", "test-right"}
    exps = rig_check.expectations(tmp_path, NMAP)
    for owner, arm in (("station", "left"), ("ros", "right")):
        wrists = [u for u in exps[owner]["usb"] if u["id"] == rig_check.D405_USB_ID]
        assert len(wrists) == 1 and wrists[0]["name"] == f"{arm} wrist D405"
    deployed = REPO / "rust/visiond/config/vision.toml"
    if deployed.is_file():
        configured = rig_check.realsense_cameras(deployed.read_text())
        assert configured and all(c["owner_role"] and c["arm"] for c in configured)


def _facts(**over):
    base = {"uptime_s": 600, "ntp_synced": True, "unix": 1000.0, "disk": {"free": 50, "total": 100},
            "ifaces": {"eth0": {"ipv4": ["192.0.2.5"], "speed": 1000, "carrier": True, "mtu": 9000}},
            "nm_profiles": [], "usb": [], "devs": {"/dev/tatbot-keypad": True},
            "units": {"tatbot-visiond-d555.service": {"active": "active", "restarts": 1}},
            "user_units": {}, "pings": {}}
    return {**base, **over}


def _status(results, needle):
    return [r["status"] for r in results if needle in r["what"]]


def test_a_healthy_probe_passes_and_a_restart_is_noted_not_failed():
    exp = rig_check.expectations(REPO, NMAP)["ros"]
    pings = {p["address"]: True for p in exp["peers"]}
    usb = [{"id": u["id"], "mbps": 5000} for u in exp["usb"]]
    results = rig_check.judge("ros", exp, _facts(pings=pings, usb=usb), local_unix=1000.5)
    assert {r["status"] for r in results} == {"ok"}
    assert any("restarted 1x" in r["what"] for r in results)


def test_slow_link_unsynced_clock_and_dead_peer_link_warn():
    exp = rig_check.expectations(REPO, NMAP)["ros"]
    facts = _facts(ntp_synced=False, unix=1010.0,
                   ifaces={"eth0": {"ipv4": ["192.0.2.5"], "speed": 100, "carrier": True, "mtu": 9000},
                           "sfp0": {"ipv4": [], "speed": None, "carrier": False, "mtu": 1400}},
                   nm_profiles=[{"name": "direct", "iface": "sfp0"}])
    results = rig_check.judge("ros", exp, facts, local_unix=1000.0)
    assert _status(results, "100 Mb/s") == ["warn"]
    assert _status(results, "NTP") == ["warn"]
    assert _status(results, "+10.0 s") == ["warn"]
    assert _status(results, "direct: sfp0 has no link") == ["warn"]


def test_missing_devices_inactive_units_small_mtu_and_silent_peers_fail():
    exp = rig_check.expectations(REPO, NMAP)["ros"]
    facts = _facts(devs={"/dev/tatbot-keypad": False}, units={"tatbot-visiond-d555.service": {"active": "inactive"}},
                   ifaces={"eth0": {"ipv4": ["192.0.2.5"], "speed": 1000, "carrier": True, "mtu": 1500}})
    results = rig_check.judge("ros", exp, facts)
    assert _status(results, "/dev/tatbot-keypad") == ["fail"]
    assert _status(results, "is inactive") == ["fail"]
    assert _status(results, "MTU 1500") == ["fail"]
    assert all(_status(results, p["address"]) == ["fail"] for p in exp["peers"])
    assert rig_check.verdicts(results)["pink drawing"]["ready"] is False


def test_an_unreachable_host_fails_every_part_it_serves():
    exp = rig_check.expectations(REPO, NMAP)["station"]
    results = rig_check.judge("station", exp, None)
    assert [(r["part"], r["status"]) for r in results] == [("station", "fail")]
    assert "still booting" in results[0]["what"]


def test_a_usb_id_is_matched_once_per_expected_device():
    exp = {"part": "station", "usb": [{"name": "a", "id": "1:1", "part": "station"},
                                      {"name": "b", "id": "1:1", "part": "station"}],
           "devs": [], "units": [], "user_units": [], "peers": [], "lan": None}
    results = rig_check.judge("n", exp, _facts(usb=[{"id": "1:1"}]))
    assert _status(results, "a on n") == ["ok"] and _status(results, "b not on") == ["fail"]


def test_the_estop_line_is_read_steady_released_and_fails_pressed_or_flickering():
    exp = {"part": "pink drawing", "estop_gpio": {"line": "GPIO17", "part": "pink drawing"}}
    judge = rig_check._estop
    assert judge("pi", exp, {"estop_line": ["lo"] * 20})[0]["status"] == "ok"
    assert "pressed" in judge("pi", exp, {"estop_line": ["hi"] * 20})[0]["what"]
    flicker = judge("pi", exp, {"estop_line": ["lo", "hi"] * 10})[0]
    assert flicker["status"] == "fail" and "flickering" in flicker["what"]
    assert judge("pi", exp, {})[0]["status"] == "warn"
    assert judge("pi", {"part": "station"}, {"estop_line": ["hi"]}) == []


DISPLAY_NMAP = {
    "__rig__": NMAP["__rig__"],
    "ros": {**NMAP["ros"], "ssh": "robot@ros",
            "services": [{"unit": "tatbot-visiond-d555.service"},
                         {"unit": "tatbot-cockpit.service", "part": "station"}],
            "hardware": [{"name": "fleet viewer hub", "unit": "tatbot-viewer@.service", "part": "station"},
                         {"name": "ros node load monitor", "unit": "tatbot-sysmon@.service", "part": "station"}]},
}


def test_a_dead_display_fails_the_station_and_leaves_pink_drawing_ready():
    exp = rig_check.expectations(REPO, DISPLAY_NMAP)["ros"]
    assert exp["part"] == "pink drawing"
    # a template unit is the login user's instance, the way `viewer install` enables it
    units = {u["unit"]: u["part"] for u in exp["units"]}
    assert units == {"tatbot-visiond-d555.service": "pink drawing", "tatbot-cockpit.service": "station",
                     "tatbot-viewer@robot.service": "station", "tatbot-sysmon@robot.service": "station"}
    facts = _facts(pings={p["address"]: True for p in exp["peers"]},
                   usb=[{"id": u["id"], "mbps": 5000} for u in exp["usb"]],
                   units={"tatbot-visiond-d555.service": {"active": "active"},
                          "tatbot-cockpit.service": {"active": "failed"},
                          "tatbot-viewer@robot.service": {"active": "inactive"}},
                   failed_units=["tatbot-cockpit.service", "tatbot-other.service"])
    results = rig_check.judge("ros", exp, facts)
    verdicts = rig_check.verdicts(results)
    assert verdicts["pink drawing"] == {"ready": True, "fail": 0, "warn": 1, "ok": verdicts["pink drawing"]["ok"]}
    assert verdicts["station"]["ready"] is False and verdicts["station"]["fail"] == 3
    # an expected unit fails once, under its own part; an unexpected one warns under the node's
    assert [r["part"] for r in results if "cockpit" in r["what"]] == ["station"]
    assert _status(results, "unit failed: tatbot-other.service") == ["warn"]


def test_every_display_piece_in_the_fleet_map_is_part_station():
    from tatbot_cli import nodes

    nmap = nodes.load(REPO)
    exps = rig_check.expectations(REPO, nmap)
    display = [(n, u) for n, e in exps.items() for k in ("units", "user_units") for u in e[k]
               if any(word in u["unit"] for word in ("viewer", "sysmon", "cockpit"))]
    if not display:
        return  # a public clone describes no fleet
    assert all(u["part"] == "station" for _, u in display), display
    assert all("@.service" not in u["unit"] for _, u in display)
    names = {u["unit"].split("@")[0].removesuffix(".service") for _, u in display}
    assert names == {"tatbot-viewer", "tatbot-viewer-window", "tatbot-sysmon", "tatbot-cockpit"}
    station = nodes.nodes_with(nmap, "station")
    assert all(exps[n]["lan"] for n in station), "the station's rig address is checked by its link"
