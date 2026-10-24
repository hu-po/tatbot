"""Where a node dials the fleet viewer hub: its direct cable while that is live, else the rig address."""
from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

from tatbot_cli import viewer_path as vp

REPO = Path(__file__).resolve().parents[2]

# RFC 5737 ranges: the rig LAN on 192.0.2.0/24, the hub's cable to the station on 198.51.100.0/30.
NMAP = {
    "hub": {"lan": "192.0.2.30", "roles": ["ros", "rerun-server"], "links": {"station": "198.51.100.2"}},
    "station": {"lan": "192.0.2.204", "roles": ["train", "station"]},
    "arm-host": {"lan": "192.0.2.20", "roles": ["arm"]},
}
CABLE, RIG = "198.51.100.2", "192.0.2.30"


def _routes(table):
    return lambda address: table.get(address)


def _never(*_a):
    raise AssertionError("probed the hub although the route already ruled the cable out")


def test_a_live_cable_off_the_rig_interface_carries_the_station():
    address, path = vp.endpoint(NMAP, "station", route=_routes({CABLE: "eth1", RIG: "eth0"}),
                                probe=lambda a, p: (a, p) == (CABLE, 9876))
    assert (address, path) == (CABLE, "direct link to the station over eth1")


def test_a_cable_routed_out_the_rig_interface_falls_back_without_a_probe():
    # The 2026-10-18 failure: no route for the cable's subnet, so the kernel sent it to the rig gateway.
    address, path = vp.endpoint(NMAP, "station", route=_routes({CABLE: "eth0", RIG: "eth0"}), probe=_never)
    assert address == RIG
    assert path == f"rig LAN over eth0; the direct link {CABLE} routes over eth0, the rig address's interface"
    address, path = vp.endpoint(NMAP, "station", route=_routes({RIG: "eth0"}), probe=_never)
    assert address == RIG and "routes over nothing" in path


def test_a_cable_the_hub_does_not_answer_on_falls_back_and_says_so():
    address, path = vp.endpoint(NMAP, "station", route=_routes({CABLE: "eth1", RIG: "eth0"}),
                                probe=lambda a, p: False)
    assert address == RIG and path.endswith(f"the hub does not answer on the direct link {CABLE}:9876")


def test_nodes_without_a_cable_and_the_hub_itself_dial_the_rig_address():
    for node in ("arm-host", "hub", "unknown-node"):
        address, path = vp.endpoint(NMAP, node, route=_routes({RIG: "eth0"}), probe=_never)
        assert (address, path) == (RIG, "rig LAN over eth0")
    assert vp.link(NMAP, "station") == ("station", CABLE)
    assert vp.link(NMAP, "hub") is None
    assert vp.endpoint({"station": NMAP["station"]}, "station", route=_routes({}), probe=_never) is None


def test_describe_names_the_cable_the_rig_lan_or_just_the_interface():
    route = _routes({CABLE: "eth1", RIG: "eth0", "127.0.0.1": "lo"})
    assert vp.describe(NMAP, "station", CABLE, route=route) == "direct link to the station over eth1"
    assert vp.describe(NMAP, "station", RIG, route=route) == "rig LAN over eth0"
    assert vp.describe(NMAP, "station", "127.0.0.1", route=route) == "127.0.0.1 over lo"
    assert vp.describe(NMAP, "arm-host", CABLE, route=route) == f"{CABLE} over eth1"  # not its cable


# `ss -H -tn state established '( sport = :9876 )'` and `ip -4 -o addr` on the hub.
SS = ("0      0      192.0.2.30:9876   192.0.2.204:51446\n"
      "0      0      198.51.100.2:9876  198.51.100.1:40000\n"
      "0      0      198.51.100.2:9876  198.51.100.1:40002\n"
      "0      0       127.0.0.1:9876    127.0.0.1:54952\n"
      "0      0      192.0.2.30:9876    192.0.2.30:34960\n"
      "0      0      [::ffff:192.0.2.30]:9876 [::ffff:192.0.2.20]:4100\n")
ADDRS = ("1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever preferred_lft forever\n"
         "2: eth0    inet 192.0.2.30/24 brd 192.0.2.255 scope global eth0\\       valid_lft forever\n"
         "4: eth1    inet 198.51.100.2/30 brd 198.51.100.3 scope global noprefixroute eth1\n")


def test_hub_clients_say_which_path_each_peer_rides():
    assert vp.clients(SS, ADDRS, NMAP, "hub") == [
        "192.0.2.20 over the rig LAN (eth0) x1",
        "192.0.2.204 over the rig LAN (eth0) x1",
        "198.51.100.1 over the direct link to the station (eth1) x2",
        "2 from this node",
    ]
    assert vp.clients("", ADDRS, NMAP, "hub") == []


def test_the_probe_is_a_plain_tcp_connect():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert vp.answers("127.0.0.1", port)
    assert not vp.answers("127.0.0.1", port)  # closed again


def test_the_shell_helper_reports_a_path_for_any_node():
    # Whatever this machine's routes, the launcher gets an address and a path, or nothing without a hub.
    env = {**os.environ, "PYTHONPATH": str(REPO / "scripts/lib"), "TATBOT_NODE": "arm-host-that-is-not-listed"}
    r = subprocess.run(["python3", "-m", "tatbot_cli.viewer_path", "endpoint"], capture_output=True, text=True,
                       env=env, timeout=30, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "" or len(r.stdout.rstrip("\n").split("\t")) == 2
    bad = subprocess.run(["python3", "-m", "tatbot_cli.viewer_path", "nope"], capture_output=True, text=True,
                         env=env, timeout=30, check=False)
    assert bad.returncode == 2 and "usage:" in bad.stderr
