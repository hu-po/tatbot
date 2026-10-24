#!/usr/bin/env python3
"""`tatbot rig check` — is every piece of demo hardware up and where it belongs?

Probes each demo-case host once over ssh (the node running it through its own
shell), in parallel, and judges what it finds against tatbot_cli.rig_check.
Read-only: it pings, lists devices and reads unit state; it starts, stops and
opens nothing, so it never touches an arm or a camera stream. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
from rig_power import Remote  # noqa: E402
from tatbot_cli import EXIT_HW_UNREACHABLE, EXIT_OK, nodes, rig_check  # noqa: E402

# Runs on the probed host with the expectation as argv[1]; prints one JSON object.
PROBE = r"""
import glob, json, os, shutil, subprocess, sys, time
exp = json.loads(sys.argv[1])
def sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""
def read(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except Exception:  # some sysfs nodes (a Thor thermal zone) raise odd errors on read
        return None
facts = {"uptime_s": float(read("/proc/uptime").split()[0])}
ntp = sh("timedatectl", "show", "-p", "NTPSynchronized", "--value").strip()
facts["ntp_synced"] = {"yes": True, "no": False}.get(ntp)
du = shutil.disk_usage(os.path.expanduser("~"))
facts["disk"] = {"free": du.free, "total": du.total}
temps = [int(t) / 1000 for t in (read(p) for p in glob.glob("/sys/class/thermal/thermal_zone*/temp")) if t and t.lstrip("-").isdigit()]
facts["temp_c"] = max(temps) if temps else None
facts["failed_units"] = [l.split()[0] for l in sh("systemctl", "--failed", "--no-legend", "--plain").splitlines() if l.strip()]
ifaces = {}
for line in sh("ip", "-o", "-4", "addr").splitlines():
    w = line.split()
    ifaces.setdefault(w[1], {"ipv4": []})["ipv4"].append(w[3].split("/")[0])
for name in os.listdir("/sys/class/net"):
    f = ifaces.setdefault(name, {"ipv4": []})
    speed = read(f"/sys/class/net/{name}/speed")
    f["speed"] = int(speed) if speed and speed.lstrip("-").isdigit() and int(speed) > 0 else None
    f["carrier"] = read(f"/sys/class/net/{name}/carrier") == "1"
    mtu = read(f"/sys/class/net/{name}/mtu")
    f["mtu"] = int(mtu) if mtu and mtu.isdigit() else None
facts["ifaces"] = ifaces
profiles = []
for line in sh("nmcli", "-t", "-f", "NAME,AUTOCONNECT", "con", "show").splitlines():
    name, _, auto = line.rpartition(":")
    if auto != "yes":
        continue
    got = sh("nmcli", "-g", "connection.interface-name,ipv4.method", "con", "show", name.replace("\\:", ":")).splitlines()
    if len(got) >= 2 and got[0] and got[1] == "manual":
        profiles.append({"name": name, "iface": got[0]})
facts["nm_profiles"] = profiles
usb = []
for d in glob.glob("/sys/bus/usb/devices/*"):
    vid, pid = read(d + "/idVendor"), read(d + "/idProduct")
    if vid and pid:
        speed = read(d + "/speed")
        usb.append({"id": f"{vid}:{pid}", "serial": read(d + "/serial"), "product": read(d + "/product"),
                    "mbps": int(float(speed)) if speed else None})
facts["usb"] = usb
facts["devs"] = {p: os.path.exists(p) for p in exp["devs"]}
def units(user, names):
    if not names:
        return {}
    out = sh("systemctl", *(["--user"] if user else []), "show", "--no-pager", "-p", "Id,ActiveState,NRestarts", *names)
    states = {}
    for block in out.strip().split("\n\n"):
        f = dict(l.split("=", 1) for l in block.splitlines() if "=" in l)
        if f.get("Id"):
            states[f["Id"]] = {"active": f.get("ActiveState"), "restarts": int(f["NRestarts"]) if f.get("NRestarts", "").isdigit() else None}
    return states
facts["units"] = units(False, exp["units"])
facts["user_units"] = units(True, exp["user_units"])
pings = {}
for a in exp["peers"]:
    pings[a] = any(subprocess.run(["ping", "-c1", "-W1", a], capture_output=True).returncode == 0 for _ in range(2))
facts["pings"] = pings
if exp.get("estop_gpio"):   # the e-stop line, 20 reads over 2 s: "lo" released, "hi" pressed or open
    pin = exp["estop_gpio"].replace("GPIO", "")
    levels = []
    for _ in range(20):
        out = sh("pinctrl", "get", pin)
        if "| lo" in out or "| hi" in out:
            levels.append("lo" if "| lo" in out else "hi")
        time.sleep(0.1)
    facts["estop_line"] = levels
facts["unix"] = time.time()
print(json.dumps(facts))
"""


def probe(remote: Remote, exp: dict) -> tuple[dict | None, str | None, float]:
    """(facts, error, local time at reply) for one node."""
    want = {"devs": [d["path"] for d in exp["devs"]], "units": [u["unit"] for u in exp["units"]],
            "user_units": [u["unit"] for u in exp["user_units"]], "peers": [p["address"] for p in exp["peers"]],
            "estop_gpio": (exp.get("estop_gpio") or {}).get("line")}
    if not remote.reachable(timeout=6):
        return None, None, time.time()
    try:
        r = remote.run("python3 - " + shlex.quote(json.dumps(want)), timeout=60, stdin=PROBE)
        sent = time.time()
        if r.returncode:
            return None, r.stderr.strip()[-200:] or f"probe exited {r.returncode}", sent
        return json.loads(r.stdout.strip().splitlines()[-1]), None, sent
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return None, str(exc)[:200], time.time()


def render(report: dict, *, every: bool) -> str:
    lines = [f"tatbot rig check — {report['at']}"]
    for part, v in report["parts"].items():
        verdict = "READY" if v["ready"] else f"NOT READY ({v['fail']} failed)"
        lines.append(f"\n{part}: {verdict}" + (f", {v['warn']} warning(s)" if v["warn"] else ""))
        for r in report["results"]:
            if r["part"] == part and (every or r["status"] != "ok"):
                lines.append(f"  {r['status'].upper():4} {r['what']}")
        if not every and v["ok"]:
            lines.append(f"  ok   {v['ok']} other check(s)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", action="store_true")
    p.add_argument("--all", action="store_true", help="list passing checks too")
    args = p.parse_args(argv)
    nmap = nodes.load(REPO)
    me = nodes.this_node(nmap)
    exps = rig_check.expectations(REPO, nmap)
    remotes = {n: Remote(n, nodes.ssh_target(nmap, n), local=n == me) for n in exps}
    with ThreadPoolExecutor(max_workers=max(1, len(exps))) as pool:
        found = dict(zip(exps, pool.map(lambda n: probe(remotes[n], exps[n]), exps), strict=True))
    results = []
    for n, (facts, error, at) in found.items():
        results += rig_check.judge(n, exps[n], facts, error=error, local_unix=at)
    results.sort(key=lambda r: ({"fail": 0, "warn": 1, "ok": 2}[r["status"]],))
    parts = rig_check.verdicts(results)
    report = {"schema": rig_check.SCHEMA, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "from": me,
              "ready": all(v["ready"] for v in parts.values()), "parts": parts, "results": results}
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report, every=args.all))
    return EXIT_OK if report["ready"] else EXIT_HW_UNREACHABLE


if __name__ == "__main__":
    sys.exit(main())
