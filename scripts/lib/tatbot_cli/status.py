"""Local observations and explicit, bounded fleet collection; never readiness gates."""

from __future__ import annotations

import concurrent.futures
import ipaddress
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from tatbot_cli import gates, locks, nodes

SCHEMA = "tatbot.status/2"
STATES = {"ok", "failed", "unknown", "not_applicable"}


def _cpu_counters(proc=Path('/proc')):
    fields = (proc / 'stat').read_text().splitlines()[0].split()
    if fields[0] != 'cpu' or len(fields) < 9:
        raise ValueError('aggregate CPU counters unavailable')
    # Guest time is already included in user/nice; do not count it twice.
    values = [int(value) for value in fields[1:9]]
    return sum(values), values[3] + values[4]


def _resources(repo, timeout_s, *, proc=Path('/proc'), sleeper=time.sleep):
    if timeout_s < .2:
        raise TimeoutError('insufficient time for a CPU sample')
    before = _cpu_counters(proc)
    started = time.monotonic()
    sleeper(.2)
    after = _cpu_counters(proc)
    interval = time.monotonic() - started
    total, idle = after[0] - before[0], after[1] - before[1]
    if total <= 0 or idle < 0 or idle > total:
        raise ValueError('CPU counters reset or did not advance')
    memory = {}
    for line in (proc / 'meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemTotal', 'MemAvailable'):
            memory[name] = int(value.split()[0]) * 1024
    if not 0 <= memory['MemAvailable'] <= memory['MemTotal'] or memory['MemTotal'] == 0:
        raise ValueError('invalid memory counters')
    import tatbot_runlog
    disks = []
    for path in dict.fromkeys((Path(repo), tatbot_runlog.log_root())):
        # A not-yet-created log directory uses its nearest existing parent.
        existing = path.expanduser().resolve()
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        usage = shutil.disk_usage(existing)
        disks.append({'path': str(path), 'sampled_path': str(existing),
                      'total_bytes': usage.total, 'used_bytes': usage.used, 'free_bytes': usage.free})
    return {'cpu': {'busy_percent': round(100 * (total - idle) / total, 2),
                    'denominator': 'all host CPU capacity', 'sample_s': round(interval, 3)},
            'memory': {'total_bytes': memory['MemTotal'], 'available_bytes': memory['MemAvailable'],
                       'used_bytes': memory['MemTotal'] - memory['MemAvailable']}, 'disks': disks}


def _systemd_services(timeout_s, *, runner=subprocess.run):
    deadline = time.monotonic() + min(3.0, timeout_s)
    result = {}
    for scope in ('system', 'user'):
        try:
            def run(*args, scope=scope):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('service collection deadline expired')
                reply = runner(['systemctl', '--' + scope, *args], capture_output=True,
                               text=True, timeout=remaining)
                if reply.returncode:
                    raise ValueError(reply.stderr.strip()[:300] or f'systemctl exit {reply.returncode}')
                return reply.stdout
            listed = json.loads(run('list-units', '--all', '--no-pager', '--output=json', 'tatbot*.service'))
            units = sorted({row['unit'] for row in listed})
            if any(not re.fullmatch(r'tatbot[-\w@.]+\.service', unit) for unit in units):
                raise ValueError('unexpected service unit identity')
            values = []
            if units:
                properties = run('show', '--no-pager', '--property=Id,LoadState,ActiveState,SubState,NRestarts', *units)
                for block in properties.strip().split('\n\n'):
                    fields = dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
                    if fields.get('Id') not in units:
                        raise ValueError('service observation identity missing')
                    values.append({'unit': fields['Id'], 'load': fields.get('LoadState', 'unknown'),
                                   'active': fields.get('ActiveState', 'unknown'),
                                   'sub': fields.get('SubState', 'unknown'),
                                   'restarts': int(fields['NRestarts']) if fields.get('NRestarts') else None})
                if {row['unit'] for row in values} != set(units):
                    raise ValueError('service observations incomplete')
            result[scope] = {'status': 'ok', 'services': values,
                             'coverage': 'loaded tatbot services; absence does not establish expected service readiness'}
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            result[scope] = {'status': 'unknown', 'services': None, 'reason': str(exc)}
    return result


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exists(path: Path) -> bool:
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def _estop_relay(device: str, nmap: dict) -> str:
    """The relay a `udp://[HOST]:PORT?from=SOURCE` e-stop reads: SOURCE's IPv4 address, or the `lan` of the
    node carrying the role SOURCE names (lerobot_robot_tatbot.estop.udp_source reads it the same way)."""
    source = (urllib.parse.parse_qs(urllib.parse.urlsplit(device).query).get("from") or [""])[0]
    try:
        return str(ipaddress.IPv4Address(source))
    except ValueError:
        lan = nmap[nodes.require_role(nmap, source)].get("lan") if source else None
        if not lan:
            raise ValueError(f"e-stop {device}: no relay address for {source!r} in config/nodes.json") from None
        return str(ipaddress.IPv4Address(lan))


def _estop_snapshot_path(status_path: Path | None) -> Path:
    if status_path is not None:
        return Path(status_path).expanduser()
    override = os.environ.get("TATBOT_ESTOP_STATUS")
    if override:
        return Path(override).expanduser()
    # A normal login and a desktop/SSH terminal may disagree about XDG.
    # Discover the same user's monitor without opening its serial device.
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    fallback = Path("/tmp") / f"tatbot-{os.getuid()}"
    roots = [Path(runtime) if runtime else fallback, Path('/run/user') / str(os.getuid()), fallback]
    paths = list(dict.fromkeys(root / 'tatbot/estop-status.json' for root in roots))
    present = [path for path in paths if _exists(path)]
    if len(present) > 1:
        raise ValueError('multiple e-stop status snapshots found; identify the intended monitor explicitly')
    return present[0] if present else paths[0]


def _estop_status(device: str, *, status_path: Path | None = None) -> dict | None:
    path = _estop_snapshot_path(status_path)
    payload = _read_json(path)
    if payload is None:
        raise ValueError("no active e-stop monitor has published a status snapshot")
    if payload.get("schema") != "tatbot.estop-status/1":
        raise ValueError(f"unsupported e-stop status schema in {path}")
    if payload.get("device") != device:
        raise ValueError("e-stop status belongs to a different device")
    pid = payload.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("e-stop status has invalid pid")
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise ValueError("e-stop status producer is no longer running") from exc
    updated = payload.get("updated_unix")
    if not isinstance(updated, (int, float)) or isinstance(updated, bool) or not math.isfinite(updated):
        raise ValueError("e-stop status has invalid update time")
    age_s = time.time() - updated
    if age_s < -2 or age_s > 1:
        raise ValueError(f"e-stop status is stale ({age_s:.1f} s old)")
    state = payload.get("state")
    engaged = payload.get("engaged")
    if state not in {"ok", "pressed", "fault"} or engaged is not (state != "ok"):
        raise ValueError("e-stop status state and engaged flag disagree")
    heartbeat_age = payload.get("heartbeat_age_ms")
    sequence = payload.get("last_sequence")
    if state == "ok" and (
        not isinstance(heartbeat_age, (int, float)) or isinstance(heartbeat_age, bool)
        or not math.isfinite(heartbeat_age) or not 0 <= heartbeat_age < 100
        or not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0
    ):
        raise ValueError("healthy e-stop status lacks a valid heartbeat")
    return {"state": state, "engaged": engaged, "heartbeat_age_ms": heartbeat_age,
            "last_sequence": payload.get("last_sequence"), "producer_pid": pid,
            "snapshot_age_ms": round(max(0.0, age_s) * 1000, 1), "path": str(path)}


def _read_json(path: Path) -> dict | None:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"expected an object in {path}")
    return data


def _ping(ip: str, timeout_s: float = 1.0) -> bool:
    r = subprocess.run(["ping", "-c", "1", "-W", "1", ip], capture_output=True,
                       text=True, timeout=timeout_s)
    if r.returncode not in (0, 1):
        raise OSError(r.stderr.strip() or f"ping failed with {r.returncode}")
    return r.returncode == 0


def _git(repo: Path, timeout_s: float) -> dict:
    # Viewer nodes are deliberately installed from git archives. Report the
    # deployment receipt without inventing a Git cleanliness observation.
    if not (repo / ".git").exists():
        manifest = _read_json(repo / ".tatbot-deploy.json")
        if manifest is not None:
            sha = manifest.get("source_commit")
            if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
                raise ValueError("deployment manifest has invalid source_commit")
            return {"sha": sha[:7], "dirty": None, "source": "deployment_manifest",
                    "source_commit": sha, "content_verified": False}
    start = time.monotonic()
    def run(*args):
        remaining = timeout_s - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError("repository probe deadline expired")
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              timeout=remaining, check=True).stdout.strip()
    return {"sha": run("rev-parse", "--short", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def _tool(repo: Path) -> dict[str, str] | None:
    """The fitted tool each arm's workspace section records, keyed by arm.

    One section per arm (`right:`, `left:`); a bare first-match regex used to
    answer for whichever section came first, which is wrong the moment a
    second arm carries a tool."""
    path = repo / "config/workspace.yaml"
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    tools: dict[str, str] = {}
    section = None
    for line in text.splitlines():
        top = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(#.*)?$", line)
        if top:
            section = top.group(1)
            continue
        m = re.match(r"^\s+tool_id:\s*(\S+)", line)
        if m and section:
            tools[section] = m.group(1)
    if not tools:
        raise ValueError(f"no tool_id in {path}")
    return tools


def _runs(node: str) -> list[dict]:
    import tatbot_runlog as rl
    out = []
    for row in rl.index_runs(strict=True):
        status = rl.resolve_status(row, node=node, strict=True)
        if status.startswith("running"):
            out.append({"run_id": row["run_id"], "workflow": row.get("workflow"), "status": status})
    return out


def _unlanded_work(repo: Path) -> dict | None:
    """Worktrees on this node holding work that is not on origin/main (docs/work.md)."""
    try:
        import tatbot_work
    except ImportError:
        return None
    try:
        s = tatbot_work.summary(tatbot_work.mirror_of(repo))
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return None
    return {"unlanded": s["unlanded"], "parked": s["parked"], "stale_remote": s["stale_remote"],
            "oldest_unlanded_hours": s["oldest_unlanded_hours"], "mirror_clean": s["mirror"]["clean"]}


def _rig_power() -> dict | None:
    from tatbot_cli import rig
    state = rig.read_state()
    if state is None:
        return None
    return {"state": state["state"], "since": state.get("since"), "by": state.get("by"),
            "run_id": state.get("run_id"), "wake_at": state.get("wake_at"),
            "nodes": sorted(state.get("nodes", {}))}


def _ink_session() -> dict | None:
    import ink_session
    s = ink_session.current()
    if s is None:
        return None
    out = {k: v for k, v in vars(s).items() if isinstance(v, (str, int, float, bool)) or v is None}
    out["describe"] = ink_session.describe(s)
    return out


def _serve() -> dict | None:
    root = Path(os.environ.get("TATBOT_SERVE_ROOT", "~/il-serve")).expanduser()
    path = root / "current-server.json"
    payload = _read_json(path)
    if payload is None:
        return None
    pid = payload.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError(f"invalid pid in {path}")
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    return {"state_file": str(path), "pid": pid, "alive": alive,
            "policy": payload.get("policy") or payload.get("policy_path"),
            "port": payload.get("port"), "policy_type": payload.get("policy_type")}


def _inkgen(timeout_s: float = 2.0) -> dict | None:
    """Is a design generator up on this node, and how long until it stops itself?

    A stopped generator is a negative fact, not a failed collection: it is meant
    to come and go, and `tatbot design generate` starts one when it needs one.
    """
    import urllib.error
    import urllib.request

    from tatbot_cli.verbs.web import INKGEN_PORT

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{INKGEN_PORT}/api/health",  # noqa: S310
                                    timeout=max(0.001, timeout_s)) as response:
            document = json.loads(response.read(1 << 20))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    return {"running": True, "model": document.get("model"), "device": document.get("device"),
            "idle_stop_s": document.get("idle_stop_s"), "idle_stop_in_s": document.get("idle_stop_in_s"),
            "last_request_unix": document.get("last_request_unix"),
            "requests_in_flight": document.get("requests_in_flight")}


def _fleet_services(repo: Path, nmap: dict, timeout_s: float) -> dict:
    routers = [n for n in nmap.values() if isinstance(n, dict) and "bus-router" in n.get("roles", [])]
    if len(routers) != 1:
        raise ValueError("one bus router is required")
    binary = repo / "rust/target/release/fleetctl"
    if not os.access(binary, os.X_OK):
        raise ValueError("fleetctl is not deployed on this node")
    result = subprocess.run([str(binary), "--connect", "tcp/" + routers[0]["lan"] + ":7447", "services"],
                            capture_output=True, text=True, timeout=min(5.0, max(0.001, timeout_s)))
    if result.returncode:
        raise ValueError(f"service discovery exit {result.returncode}: {result.stderr.strip()}")
    value = json.loads(result.stdout)
    if value.get("schema") != "tatbot.services/1":
        raise ValueError("invalid service discovery schema")
    return value


def collect(repo: Path, *, cams: bool = False, timeout_s: float = 15.0, service_discovery: bool = True) -> dict:
    """Local only. Unknown collection failures make complete false; negative facts do not."""
    import tatbot_profile

    nmap = nodes.load(repo)
    node = nodes.this_node(nmap)
    roles = nodes.roles_of(nmap, node)
    started = time.monotonic()
    deadline = started + timeout_s
    observations = {}
    complete = True

    def observe(name, probe=None, *, applicable=True, reason=None, boolean=False, value=None,
                informational=False):
        nonlocal complete
        state = "ok"
        if not applicable:
            state, value = "not_applicable", None
            reason = reason or "not owned by this node's configured roles"
        elif probe is None:
            state = "unknown"
            if not informational:
                complete = False
        elif time.monotonic() >= deadline:
            state, value, reason = "unknown", None, "collection deadline expired"
            complete = False
        else:
            try:
                value = probe()
                if boolean and value is False:
                    state = "failed"
            except (OSError, ValueError, TypeError, KeyError, ImportError, subprocess.SubprocessError) as exc:
                state, value, reason = "unknown", None, f"{type(exc).__name__}: {exc}"
                if not informational:
                    complete = False
        observations[name] = {"node": node, "vantage_node": node, "checked_at": _utc(),
                              "status": state, "value": value, "reason": reason,
                              "required": applicable and not informational}
        return value

    profile = observe("profile", lambda: tatbot_profile.load(repo))
    driver = (profile or {}).get("driver") or {}
    # The profile object is used internally, only the name is a status observation.
    if profile:
        observations["profile"]["value"] = profile["name"]
    observe("repo", lambda: _git(repo, max(0.001, deadline - time.monotonic())))
    observe("host_resources", lambda: _resources(repo, deadline - time.monotonic()), informational=True)
    observe("systemd_services", lambda: _systemd_services(deadline - time.monotonic()), informational=True)
    for label, field in (("leader (left)", "leader_ip"), ("follower (right)", "follower_ip")):
        ip = driver.get(field)
        observe(f"arms.{label}", (lambda ip=ip: _ping(ip, min(1.5, deadline - time.monotonic()))) if ip else None,
                applicable="arm" in roles, boolean=True, reason=None if ip else "controller address not configured")
    estop = driver.get("estop_device")
    # A serial e-stop is present when its device node is; the palette relay's (udp://) when the relay answers.
    present = (lambda: _ping(_estop_relay(estop, nmap), min(1.5, deadline - time.monotonic()))) \
        if estop and estop.startswith("udp://") else (lambda: _exists(Path(estop)))
    observe("estop_device", present if estop else None,
            applicable="estop" in roles, boolean=True, reason=None if estop else "device path not configured")
    heartbeat = observe("estop_heartbeat", (lambda: _estop_status(estop)) if estop else None,
                        applicable="estop" in roles, informational=True,
                        reason=None if estop else "device path not configured")
    if heartbeat and heartbeat["state"] != "ok":
        observations["estop_heartbeat"]["status"] = "failed"
    observe("arm_connection_ready", applicable="arm" in roles, informational=True,
            reason="not probed: ping does not establish driver readiness; status never opens an arm connection")
    observe("fitted_tool_last_touchoff", lambda: _tool(repo), reason="last touch-off record, not physical tool verification")
    observe("ink_session", _ink_session)
    observe("runs_running", lambda: _runs(node))
    observe("unlanded_work", lambda: _unlanded_work(repo), informational=True,
            reason="worktrees with uncommitted or unpushed-to-main changes (tatbot work status)")
    rig_state = observe("rig_power", _rig_power, informational=True,
                        reason="the `tatbot rig sleep` marker on this node; absent means never slept here")
    if rig_state and rig_state.get("state") == "asleep":
        observations["rig_power"]["status"] = "failed"
        observations["rig_power"]["reason"] = "asleep: hardware verbs refuse until `tatbot rig wake`"
    root = gates.train_root()
    observe("training_lock", lambda: locks.held(root / ".tatbot-training.lock"), applicable="train" in roles,
            reason="point-in-time advisory-lock ownership; the launcher acquires its own lock")
    observe("sweep_pause", lambda: _exists(root / "SWEEP_PAUSE"), applicable="train" in roles)
    server = observe("policy_server", _serve, applicable="serve" in roles)
    if server and not server["alive"]:
        observations["policy_server"]["status"] = "failed"
    observe("inkgen", lambda: _inkgen(min(2.0, max(0.001, deadline - time.monotonic()))),
            applicable="inkgen" in roles,
            reason="the generator is started on demand and stops itself when idle; absent is normal")
    if service_discovery:
        observe("fleet_services", lambda: _fleet_services(repo, nmap, deadline - time.monotonic()), informational=True)
    if cams:
        for name, ip in nodes.cameras(repo).items():
            observe(f"cameras.{name}", lambda ip=ip: _ping(ip, min(1.5, deadline - time.monotonic())),
                    applicable="cameras-lan" in roles, boolean=True)
    return {"schema": SCHEMA, "schema_version": 2, "scope": "local", "node": node,
            "vantage_node": node, "checked_at": _utc(), "roles": roles, "complete": complete,
            "elapsed_s": round(time.monotonic() - started, 3), "observations": observations}


def legacy(report: dict) -> dict:
    """Compatibility projection; unknown/non-applicable values stay null, never false."""
    if report["scope"] == "fleet":
        return {"scope": "fleet", "node": report["node"], "complete": report["complete"],
                "nodes": {n: legacy(r) if r.get("scope") == "local" else r for n, r in report["nodes"].items()}}
    out = {"node": report["node"], "roles": report["roles"], "checked_at": report["checked_at"]}
    for name, obs in report["observations"].items():
        if name in ("estop_heartbeat", "arm_connection_ready"):
            continue
        if "." in name:
            group, key = name.split(".", 1)
            out.setdefault(group, {})[key] = obs["value"]
        else:
            out[name] = obs["value"]
    return out


def _validate_remote(payload, node):
    if isinstance(payload, dict) and "schema" not in payload and "node" in payload:
        raise ValueError("remote CLI returned legacy status; update the idle node's CLI to "
                         "status schema 2, then retry (status does not deploy or sync)")
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("scope") != "local":
        raise ValueError("collector must return a version-2 local status report")
    if payload.get("node") != node or not isinstance(payload.get("complete"), bool):
        raise ValueError("collector identity/completion mismatch")
    if payload.get("vantage_node") != node or not isinstance(payload.get("roles"), list):
        raise ValueError("collector vantage/roles missing")
    stamp = datetime.fromisoformat(payload["checked_at"].replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("collector timestamp lacks timezone")
    if not isinstance(payload.get("observations"), dict) or not payload["observations"]:
        raise ValueError("collector returned no observations")
    for obs in payload["observations"].values():
        if not isinstance(obs, dict) or obs.get("node") != node or obs.get("status") not in STATES:
            raise ValueError("invalid observation identity/state")
        stamp = datetime.fromisoformat(obs["checked_at"].replace("Z", "+00:00"))
        if (stamp.tzinfo is None or "value" not in obs or "reason" not in obs
                or not isinstance(obs.get("required"), bool) or obs.get("vantage_node") != node):
            raise ValueError("observation lacks timestamp timezone/value/reason")
    if payload["complete"] and any(o["required"] and o["status"] == "unknown" for o in payload["observations"].values()):
        raise ValueError("complete report contains required unknown observations")
    return payload


def collect_fleet(repo: Path, *, cams: bool = False, timeout_s: float = 15.0, runner=subprocess.run) -> dict:
    """At most four status collectors, all with the same overall deadline; no sync/TTY."""
    nmap = nodes.load(repo)
    here = nodes.this_node(nmap)
    targets = sorted(set(nmap) | {here})
    deadline = time.monotonic() + timeout_s
    try:
        services = {'status': 'ok', 'value': _fleet_services(repo, nmap, min(2.0, timeout_s)),
                    'vantage_node': here, 'checked_at': _utc(), 'reason': None}
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        services = {'status': 'unknown', 'value': None, 'vantage_node': here,
                    'checked_at': _utc(), 'reason': str(exc)}
    def one(node):
        checked = _utc()
        if node != here and nmap[node].get("checkout") is None:
            return {"node": node, "vantage_node": here, "checked_at": checked,
                    "status": "not_applicable", "reason": "no configured checkout", "complete": True}
        if node != here and not nodes.roles_of(nmap, node):
            # A node with no roles is retired from the fleet: nothing runs there
            # and no verb routes to it, so it is not probed. It is typically off,
            # and an ssh that waits out the whole deadline would make every
            # fleet status slow and incomplete for a host nobody depends on.
            return {"node": node, "vantage_node": here, "checked_at": checked, "roles": [],
                    "status": "retired", "reason": "no roles in config/nodes.json: not collected", "complete": True}
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("fleet collection deadline expired")
            args = ["--json", "status", "--schema-version", "2", "--timeout-s", str(remaining),
                    "--no-service-discovery"]
            if cams:
                args.append("--cams")
            cmd = ([sys.executable, str(repo / "scripts/lib/tatbot_cli"), *args] if node == here else
                   nodes.hop_argv(nmap, node, args, tty=False, sync=False))
            reply = runner(cmd, capture_output=True, text=True, timeout=remaining)
            if reply.returncode not in (0, 1):
                raise ValueError(f"collector exit {reply.returncode}: {reply.stderr.strip()[:300]}")
            payload = _validate_remote(json.loads(reply.stdout), node)
            if (reply.returncode == 0) != payload["complete"]:
                raise ValueError("collector exit code disagrees with completeness")
            payload["transport"] = {"vantage_node": here, "received_at": _utc(), "status": "ok"}
            remote_time = datetime.fromisoformat(payload["checked_at"].replace("Z", "+00:00"))
            payload["transport"]["reported_clock_offset_s"] = round(remote_time.timestamp() - time.time(), 3)
            return payload
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            return {"node": node, "vantage_node": here, "checked_at": _utc(), "status": "unknown",
                    "reason": f"{type(exc).__name__}: {exc}", "complete": False,
                    "roles": nodes.roles_of(nmap, node)}
    # subprocess.run kills/reaps timed-out local children. Queued work checks the
    # same deadline before spawning. Context shutdown therefore waits only for
    # bounded cleanup, not another per-node timeout period.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(zip(targets, pool.map(one, targets), strict=True))
    return {"schema": SCHEMA, "schema_version": 2, "scope": "fleet", "node": here,
            "vantage_node": here, "checked_at": _utc(), "timeout_s": timeout_s,
            "complete": all(r["complete"] for r in results.values()), "nodes": results,
            "fleet_services": services}


def render(report: dict) -> str:
    lines = [f"tatbot status — {report['scope']} from {report['vantage_node']}  {report['checked_at']}"]
    if report["scope"] == "fleet":
        for node, result in report["nodes"].items():
            if result.get("scope") == "local":
                lines.extend("  " + row for row in render(result).splitlines())
            else:
                lines.append(f"  {node}: {result['status']} — {result['reason']}")
    else:
        for name, obs in report["observations"].items():
            value = json.dumps(obs["value"], ensure_ascii=False, sort_keys=True)
            lines.append(f"  {name:<28} {obs['status']:<14} {value}")
            if obs["reason"]:
                lines.append(f"    {obs['reason']}")
    lines.append("  collection " + ("complete" if report["complete"] else "incomplete (see unknown observations)"))
    return "\n".join(lines)
