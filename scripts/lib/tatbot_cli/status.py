"""`tatbot status` — the fleet in one screen, cheap enough to call constantly."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tatbot_cli import gates, nodes


def _driver():
    import tatbot_profile

    from tatbot_cli.registry import repo_root
    try:
        return tatbot_profile.load(repo_root()).get("driver") or {}
    except tatbot_profile.ProfileError:
        return {}


_D = _driver()
ARMS = {k: str(_D[f]) for k, f in
        (("leader (left)", "leader_ip"), ("follower (right)", "follower_ip"))
        if _D.get(f)}
ESTOP = Path(_D.get("estop_device") or "/dev/tatbot-estop-unconfigured")


def _ping(ip: str, timeout_s: float = 1.0) -> bool:
    try:
        r = subprocess.run(["ping", "-c", "1", "-W", str(int(timeout_s)), ip],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_s + 2)
        return r.returncode == 0
    except Exception:
        return False


def _git(repo: Path) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    return {"sha": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def _tool(repo: Path) -> str | None:
    ws = repo / "config" / "workspace.yaml"
    if not ws.is_file():
        return None
    m = re.search(r"^\s*tool_id:\s*(\S+)", ws.read_text(), re.M)
    return m.group(1) if m else None


def _runs(repo: Path) -> list[dict]:
    sys.path.insert(0, str(repo / "scripts" / "lib"))
    try:
        import tatbot_runlog as rl
    except Exception:
        return []
    out = []
    try:
        for r in rl.index_runs():
            st = rl.resolve_status(r)
            if st.startswith("running"):
                out.append({"run_id": r.get("run_id"), "workflow": r.get("workflow"), "status": st})
    except Exception:
        pass
    return out


def _ink_session(repo: Path) -> dict | None:
    sys.path.insert(0, str(repo / "scripts" / "lib"))
    try:
        import ink_session
        s = ink_session.current()
    except Exception:
        return None
    if s is None:
        return None
    d = getattr(s, "__dict__", {})
    out = {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool)) or v is None}
    out["describe"] = ink_session.describe(s)
    return out


def _serve() -> dict | None:
    root = Path(os.environ.get("TATBOT_SERVE_ROOT", "~/il-serve")).expanduser()
    state = root / "current-server.json"
    if not state.is_file():
        return None
    try:
        payload = json.loads(state.read_text())
    except Exception:
        return {"state_file": str(state), "error": "unreadable"}
    pid = payload.get("pid")
    alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            alive = True
        except Exception:
            alive = False
    return {"state_file": str(state), "pid": pid, "alive": alive,
            "policy": payload.get("policy") or payload.get("policy_path"),
            "port": payload.get("port"), "policy_type": payload.get("policy_type")}


def _profile_name() -> str:
    """Which hardware profile these arm/e-stop values came from."""
    import tatbot_profile

    from tatbot_cli.registry import repo_root
    try:
        return tatbot_profile.load(repo_root())["name"]
    except tatbot_profile.ProfileError:
        return "(none resolved)"


def collect(repo: Path, *, cams: bool) -> dict:
    nmap = nodes.load(repo)
    node = nodes.this_node(nmap)
    root = gates.train_root()
    out = {
        "node": node,
        "profile": _profile_name(),
        "roles": nodes.roles_of(nmap, node),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": _git(repo),
        "arms": {name: _ping(ip) for name, ip in ARMS.items()},
        "estop_device": ESTOP.exists(),
        "fitted_tool_last_touchoff": _tool(repo),
        "ink_session": _ink_session(repo),
        "runs_running": _runs(repo),
        "training_lock": (root / ".tatbot-training.lock").exists(),
        "sweep_pause": (root / "SWEEP_PAUSE").exists(),
        "policy_server": _serve(),
    }
    if cams:
        out["cameras"] = {name: _ping(ip) for name, ip in nodes.cameras(repo).items()}
    return out


def render(s: dict) -> str:
    yes, no = "yes", "no"
    lines = [f"tatbot status — {s['node']} [{', '.join(s['roles']) or 'no roles'}]  {s['checked_at']}",
             f"  repo      {s['repo']['sha']}{' (dirty)' if s['repo']['dirty'] else ''}"]
    for name, ok in s["arms"].items():
        lines.append(f"  arm       {name:<18} {'reachable' if ok else 'unreachable'}")
    lines.append(f"  e-stop    {ESTOP} {'present' if s['estop_device'] else 'absent'}")
    lines.append(f"  profile   {s.get('profile', '-')}")
    lines.append(f"  tool      {s['fitted_tool_last_touchoff'] or '?'}  (last touch-off; state --ee-tool anyway)")
    ink = s["ink_session"]
    if ink:
        desc = ink["describe"].splitlines()
        lines.append(f"  ink       {desc[0]}")
        lines.extend(f"            {ln.strip()}" for ln in desc[1:])
    else:
        lines.append("  ink       no open session on this node")
    runs = s["runs_running"]
    lines.append(f"  runs      {len(runs)} running" + ("" if not runs else ": " + ", ".join(r['run_id'] for r in runs)))
    lines.append(f"  training  lock {yes if s['training_lock'] else no}, SWEEP_PAUSE {yes if s['sweep_pause'] else no}")
    srv = s["policy_server"]
    if srv is None:
        lines.append("  serve     no state file")
    else:
        lines.append(f"  serve     pid {srv.get('pid')} {'alive' if srv.get('alive') else 'DEAD'}  {srv.get('policy')}  :{srv.get('port')}")
    if "cameras" in s:
        for name, ok in s["cameras"].items():
            lines.append(f"  camera    {name:<10} {'reachable' if ok else 'unreachable'}")
    return "\n".join(lines)
