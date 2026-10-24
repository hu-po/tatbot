"""`tatbot rig lan-install`'s generated files: the ssh fallback and the rig clock. No ssh."""

from __future__ import annotations

import ipaddress
import shutil
import subprocess

import pytest
from tatbot_cli import rig_lan

SUBNET = ipaddress.IPv4Network("192.0.2.0/24")
NMAP = {
    "__rig__": {"subnet": str(SUBNET), "gateway": "192.0.2.1"},
    "clock": {"ssh": "u@198.51.100.4", "lan": "192.0.2.4", "hostname": "clock-host", "roles": ["station", "rig-clock"]},
    "ros": {"ssh": "u@198.51.100.5", "lan": "192.0.2.5", "roles": ["ros"]},
    "desk": {"ssh": "u@198.51.100.6", "roles": ["operator"]},
    "away": {"ssh": "u@198.51.100.7", "lan": "203.0.113.7", "roles": []},
}


def test_ssh_config_falls_back_to_each_rig_nodes_lan_with_the_tailnet_host_key():
    text = rig_lan.ssh_config(NMAP, SUBNET)
    assert 'Match host 198.51.100.4,clock,clock-host exec "ip -4 -o addr show to 192.0.2.0/24' in text
    assert "  HostName 192.0.2.5\n  HostKeyAlias 198.51.100.5\n" in text
    assert "! timeout 1 bash -c '</dev/tcp/198.51.100.5/22' 2>/dev/null && timeout 1 bash -c '</dev/tcp/192.0.2.5/22'" in text
    # no rig address, or one off the rig subnet: no block
    assert "198.51.100.6" not in text and "203.0.113.7" not in text


@pytest.mark.skipif(not shutil.which("ssh"), reason="no ssh client")
def test_ssh_parses_the_generated_config(tmp_path):
    cfg = tmp_path / "config"
    cfg.write_text(rig_lan.ssh_config(NMAP, SUBNET) + "\nHost ros\n  HostName 198.51.100.5\n")
    r = subprocess.run(["ssh", "-F", str(cfg), "-G", "u@198.51.100.5"], capture_output=True, text=True,
                       timeout=10, check=False)
    assert r.returncode == 0, r.stderr
    # this test host is not on the example subnet, so the exec fails and the tailnet target stands
    assert "hostname 198.51.100.5" in r.stdout.splitlines()


def test_clock_node_is_the_one_rig_clock_with_a_lan():
    assert rig_lan.clock_node(NMAP) == "clock"
    with pytest.raises(ValueError):
        rig_lan.clock_node({k: v for k, v in NMAP.items() if k != "clock"})
    with pytest.raises(ValueError):
        rig_lan.clock_node({**NMAP, "clock": {**NMAP["clock"], "lan": None}})


def test_clock_files():
    server = rig_lan.chrony_server(SUBNET)
    assert "allow 192.0.2.0/24\n" in server and f"local stratum {rig_lan.LOCAL_STRATUM}\n" in server
    assert rig_lan.chrony_client("192.0.2.4").endswith("server 192.0.2.4 iburst prefer\n")
    assert rig_lan.timesyncd_client("192.0.2.4").endswith(f"[Time]\nNTP=192.0.2.4 {rig_lan.TIMESYNCD_SECOND}\n")
