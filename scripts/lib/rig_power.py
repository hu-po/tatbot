#!/usr/bin/env python3
"""`tatbot rig sleep|wake|status|off` — idle the rig's cameras, trackers and demo window overnight.

Every node sleep and wake touch is a `power` record in config/nodes.json (see
tatbot_cli.rig). Nothing here moves an arm, stops the drawing stack or cuts arm
power: sleep refuses while the drawing stack holds an arm that has not landed
or a drawing or camera run is open, and leaves the arm as its last session left
it. Stdlib only, so it runs from any node's bare clone.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
from tatbot_cli import (  # noqa: E402
    EXIT_BUSY,
    EXIT_GATE_REFUSED,
    EXIT_HW_UNREACHABLE,
    EXIT_OK,
    EXIT_TOOL_FAILED,
    EXIT_WRONG_NODE,
    nodes,
    rig,
    tatbot_ros,
    viewer_path,
)

# ssh takes the FIRST value it sees for an option, so the connect timeout is
# given per call, never in this prefix.
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
REMOTE_MARKER = "~/tatbot-logs/rig/state.json"
# Long-lived runs that hold no arm: a manifested service's console and the drawing stack's.
STANDING_RUNS = ("fleet-service", "ros-stack")
# Runs that use what sleep stops or the arm it leaves up, by workflow prefix: the drawing stack's clients
# (`ros-cli` on the node the verb ran from; draw, touch, station and calibration runs on the ros node) and
# the camera tools. Sleep refuses while one runs anywhere it asks; training or a simulation never does.
RIG_RUNS = ("ros-", "vision", "calib", "surface-rgbd")
STOPPED = ("inactive", "failed")
ENABLED = ("enabled", "enabled-runtime")
BUS_WAIT_S = 20      # the router and its presence answer within seconds of a start
START_WAIT_S = 150   # the D555 owner's TimeoutStartSec is 120, and Restart= heals a start that lost the race in 5 s
STEADY_S = 5         # no restart in this long once every service is on the bus
PAGE_WAIT_S = 60     # stencild finding the print again after the D555 owner starts


class Remote:
    """One node's transport: a fresh, bounded ssh per call, or a local shell for
    the node this command runs on. A node holds no ssh key for itself, so
    probing it over ssh would report the very host running the command as
    asleep; it is plainly reachable, and its shell is right here."""

    def __init__(self, name: str, target: str | None, *, local: bool = False):
        self.name, self.target, self.local = name, target, local

    def run(self, script: str, *, timeout: float = 60, stdin: str | None = None) -> subprocess.CompletedProcess:
        # Never the operator's terminal: an ssh that inherits it forwards its keys to the remote command.
        kw = {"capture_output": True, "text": True, "timeout": timeout, "input": stdin or "", "check": False}
        if self.local:
            return subprocess.run(["bash", "-c", script], **kw)
        if not self.target:
            raise OSError(f"{self.name} has no ssh target")
        return subprocess.run([*SSH, "-o", "ConnectTimeout=8", self.target, "bash -c " + shlex.quote(script)], **kw)

    def reachable(self, timeout: float = 6) -> bool:
        if self.local:
            return True
        if not self.target:
            return False
        try:
            r = subprocess.run([*SSH, "-o", f"ConnectTimeout={int(timeout)}", self.target, "true"],
                               capture_output=True, text=True, timeout=timeout + 4, check=False)
        except (subprocess.SubprocessError, OSError):
            return False
        return r.returncode == 0

    def unit_states(self, units: list[str], *, user: bool = False) -> dict[str, dict]:
        """unit -> {"active", "enabled", "loaded", "restarts"}; a unit not installed reads loaded `not-found`."""
        if not units:
            return {}
        r = self.run(f"systemctl {'--user ' if user else ''}show --no-pager "
                     "--property=Id,LoadState,ActiveState,UnitFileState,NRestarts " + shlex.join(units))
        if r.returncode:
            raise OSError(f"{self.name}: systemctl show failed: {r.stderr.strip()[:200]}")
        states = {}
        for block in r.stdout.strip().split("\n\n"):
            f = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
            if f.get("Id"):
                states[f["Id"]] = {"active": f.get("ActiveState", "unknown"), "enabled": f.get("UnitFileState", ""),
                                   "loaded": f.get("LoadState", ""),
                                   "restarts": int(f["NRestarts"]) if f.get("NRestarts", "").isdigit() else None}
        return states

    def write_marker(self, state: dict) -> None:
        text = json.dumps({"schema": rig.SCHEMA, **state}, indent=2, sort_keys=True) + "\n"
        r = self.run(f"mkdir -p ~/tatbot-logs/rig && cat > {REMOTE_MARKER}.tmp && mv -f {REMOTE_MARKER}.tmp {REMOTE_MARKER}",
                     stdin=text)
        if r.returncode:
            raise OSError(f"{self.name}: marker write failed: {r.stderr.strip()[:200]}")

    def read_marker(self) -> dict | None:
        try:
            r = self.run(f"cat {REMOTE_MARKER} 2>/dev/null || true")
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode or not r.stdout.strip():
            return None
        try:
            data = json.loads(r.stdout)
        except ValueError:
            return None
        return data if isinstance(data, dict) and data.get("schema") == rig.SCHEMA else None


def _remote(nmap: dict, name: str, me: str) -> Remote:
    """`name`'s transport from `me`, the node this command runs on."""
    return Remote(name, nodes.ssh_target(nmap, name), local=name == me)


def _checkout(nmap: dict, remote: Remote) -> str:
    """The node's checkout as its own shell names it (this one's own, here)."""
    return shlex.quote(str(REPO)) if remote.local else nodes.remote_checkout(nmap, remote.name)


def _blank() -> dict:
    return {"units": None, "host": None, "note": None, "error": None}


def _emit(args, payload: dict, text: str) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else text)


def _say(text: str) -> None:
    print(f"rig: {text}", file=sys.stderr, flush=True)


def _render(verb: str, report: dict) -> str:
    lines = [f"tatbot rig {verb} — {report.get('state', '')} {report.get('at', '')}".rstrip()]
    for name, node in report["nodes"].items():
        bits = [f"host {node['host']}" if node.get("host") else None,
                f"units {node['units_verdict']}" if node.get("units_verdict") else None,
                f"window {node['window']}" if node.get("window") else None,
                node.get("kept_verdict"), node.get("note"), "ERROR " + node["error"] if node.get("error") else None]
        lines.append(f"  {name:<12} " + "  ".join(b for b in bits if b))
    lines += [f"  note {note}" for note in report.get("notes", [])]
    return "\n".join(lines)


def _ordered(plan: dict, nmap: dict) -> list[str]:
    """The bus router's node first: a service anywhere dials the router at start and exits without it."""
    return sorted(plan, key=lambda n: "bus-router" not in nmap.get(n, {}).get("roles", []))


def _runs_on(nmap: dict, remote: Remote) -> list[dict]:
    """The runs a node's own `tatbot status` says are running there (not elsewhere, not crashed).
    Raises when it answers but cannot say: sleeping or shutting down blind is not an option."""
    r = remote.run(f"cd {_checkout(nmap, remote)} && scripts/tatbot --no-hop --json status "
                   "--no-service-discovery --timeout-s 8", timeout=40)
    try:
        rows = json.loads(r.stdout)["observations"]["runs_running"]["value"]
    except (ValueError, KeyError, TypeError):
        raise OSError(f"could not read running workflows on {remote.name}: "
                      f"{(r.stderr or r.stdout).strip()[:200]}") from None
    if rows is None:
        raise OSError(f"{remote.name} could not enumerate its running workflows")
    return [row for row in rows if row.get("status") == "running"]


def _busy(nmap: dict, me: str, names: list[str], every: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """The runs on `names` that would lose what they use, and notes: on a node in `every` (off powers it down)
    any run but the standing ones, anywhere else a drawing or camera run (RIG_RUNS)."""
    running, notes = [], []
    for name in dict.fromkeys(n for n in names if n == me or nmap.get(n, {}).get("checkout")):
        remote = _remote(nmap, name, me)
        if not remote.reachable():
            notes.append(f"{name} did not answer ssh: not asked whether it is busy")
            continue
        running += [f"{name}: {row.get('workflow')} {row.get('run_id')}" for row in _runs_on(nmap, remote)
                    if row.get("workflow") not in STANDING_RUNS
                    and (name in every or str(row.get("workflow")).startswith(RIG_RUNS))]
    return running, notes


def _refuse_busy(verb: str, nmap: dict, me: str, names: list[str],
                 every: tuple[str, ...] = ()) -> tuple[int | None, list[str]]:
    """(an exit code to refuse with, or None to go on; notes)."""
    try:
        running, notes = _busy(nmap, me, names, every)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"rig {verb} refused: {exc}", file=sys.stderr)
        return EXIT_HW_UNREACHABLE, []
    if running:
        print(f"rig {verb} refused: runs are open that use the rig — finish or land them first:\n  "
              + "\n  ".join(running), file=sys.stderr)
        return EXIT_BUSY, []
    return None, notes


# --- the drawing stack: read, never touched ---------------------------------------


def _stack() -> dict | None:
    """`tatbot ros status --json` from here: the drawing stack's units, each arm's safety, the page on the bus."""
    try:
        r = subprocess.run([str(REPO / "scripts/tatbot"), "--json", "ros", "status"], capture_output=True,
                           text=True, timeout=90, check=False, stdin=subprocess.DEVNULL)
        return json.loads(r.stdout) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _page(stack: dict | None) -> dict:
    return ((stack or {}).get("page") or {}).get("summary") or {}


def _follows_stencil(stack: dict | None) -> bool:
    """The stack runs, on a page the stencil tracker measures (a fixed page never reaches the bus)."""
    return bool(stack) and (stack.get("units") or {}).get(rig.DRAWING_STACK[-1]) == "active" \
        and "page=stencil" in (stack.get("effective") or "").split()


def _stack_line(stack: dict | None) -> str:
    """One line: the stack's units, each arm latched or not and landed or not, the page on the bus."""
    if not stack:
        return "drawing stack: `tatbot ros status` did not answer"
    units = ", ".join(f"{u.removesuffix('.service')} {s}" for u, s in (stack.get("units") or {}).items())
    arms = [f"{arm} {'latched (' + str(s.get('latch_reason')) + ')' if s.get('latched') else 'not latched'}, "
            f"{'landed' if s.get('landed') else 'NOT landed'}"
            for arm, s in ((stack.get("stack") or {}).get("safety") or {}).items()]
    page = _page(stack)
    return (f"drawing stack: {units}; " + ("; ".join(arms) or "no arm safety state") +
            f"; page {page.get('measured', '?')} measured / {page.get('lost', '?')} lost in 2 s on the bus")


def _unlanded(stack: dict | None) -> list[str]:
    """The arms a running stack on real hardware holds that have not landed. Unknown (the stack down, not
    answering, or no arm reporting) is none: no driver then holds an arm that a night could drop."""
    if not stack or (stack.get("units") or {}).get(rig.DRAWING_STACK[-1]) != "active" \
            or tatbot_ros.stack_hardware(stack.get("effective") or "") in tatbot_ros.SIMULATED_HARDWARE:
        return []
    return sorted(arm for arm, s in ((stack.get("stack") or {}).get("safety") or {}).items() if not s.get("landed"))


def _refuse_unlanded(stack: dict | None) -> int | None:
    """Exit 3 while the stack holds an arm that has not landed: sleep moves no arm and leaves it so all night."""
    arms = _unlanded(stack)
    if not arms:
        return None
    print(f"rig sleep refused: {' and '.join(arms)} has not landed, and sleep leaves an arm as it is all night. "
          "Land it first, then sleep: `tatbot ros down` lands each arm that answers and stops the stack; an arm "
          "whose controller reports an error lands only after a restart, `tatbot ros up` with the stack's flags "
          f"({stack.get('effective') or 'stack.yaml defaults'}).\n  {_stack_line(stack)}", file=sys.stderr)
    return EXIT_GATE_REFUSED


def _stack_after_wake() -> str:
    """The stack's line once stencild measures the page again (bounded), so the morning's first draw is not
    planned on last night's pose. A note, never the gate: the stack's own page checks are the interlock."""
    deadline = time.monotonic() + PAGE_WAIT_S
    while True:
        stack = _stack()
        if not _follows_stencil(stack) or _page(stack).get("measured") or time.monotonic() >= deadline:
            break
        time.sleep(5)
    line = _stack_line(stack)
    if _follows_stencil(stack) and not _page(stack).get("measured"):
        line += f" — no measured page {PAGE_WAIT_S} s after the trackers started: is the print in the D555's view? (`tatbot ros page`)"
    return line


def _rig_check_notes() -> list[str]:
    """`tatbot rig check`, part by part: what a wake leaves NOT READY, read-only and never the gate."""
    try:
        r = subprocess.run([sys.executable, str(REPO / "scripts/lib/rig_check.py"), "--json"], capture_output=True,
                           text=True, timeout=180, check=False, stdin=subprocess.DEVNULL)
        return check_lines(json.loads(r.stdout))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return [f"rig check did not run: {exc}"]


def check_lines(report: dict) -> list[str]:
    """One line per part of a `tatbot rig check --json` report, naming each failed check."""
    out = []
    for part, v in (report.get("parts") or {}).items():
        failed = [x["what"] for x in report.get("results", []) if x.get("part") == part and x.get("status") == "fail"]
        out.append(f"rig check: {part} " + ("READY" if v.get("ready") else "NOT READY: " + "; ".join(failed))
                   + (f" ({v['warn']} warning(s))" if v.get("warn") else ""))
    return out


# --- plans ----------------------------------------------------------------------


def _operators(nmap: dict, skip, me: str) -> list[str]:
    """The operator nodes but `me` and those in `skip`."""
    return [n for n in nodes.nodes_with(nmap, "operator") if n not in skip and n != me]


def _sleep_steps(p: dict) -> list[str]:
    gating = [u for u in p["units"] if u not in p["display"]]
    kept = p["kept"] + [f"{u} (user)" for u in p["kept_user"]]
    return ([f"stop (display, best-effort) {' '.join(p['display'])}"] if p["display"] else []) + \
        ([f"stop {' -> '.join(gating)}"] if gating else []) + \
        (["close the demo viewer window (best-effort)"] if p["window"] else []) + \
        ([f"leaves up {' '.join(kept)}"] if kept else [])


def _wake_steps(p: dict) -> list[str]:
    return ([f"start {' '.join(p['bus'])}; wait <= {BUS_WAIT_S} s for {' '.join(_keys(p, p['bus']))} on the bus"]
            if p["bus"] else []) + \
        ([f"start {' '.join(p['start'])} (enabled units only); wait <= {START_WAIT_S} s until active, <= {BUS_WAIT_S} s "
          f"for {' '.join(_keys(p, p['bus'] + p['start']))} on the bus, then no restart in {STEADY_S} s"]
         if p["start"] else []) + \
        ([f"start (display, best-effort) {' '.join(p['display'] + p['ensure'])}"] if p["display"] or p["ensure"] else [])


WINDOW_STEP = ("open the demo viewer window, only over its direct link to the hub (best-effort): "
               f"tatbot viewer open {' '.join(rig.WINDOW_OPEN)}")


def _keys(p: dict, units: list[str]) -> list[str]:
    """The `<node>/<key>` liveliness names the units hold on the bus, once each."""
    return list(dict.fromkeys(p["liveliness"][u] for u in units if u in p["liveliness"]))


def _rows(rows: list[tuple[str, str]], numbered: bool) -> list[str]:
    """`  node  [n] step` lines, the node named on its first row only."""
    return [f"  {name if i == 0 or rows[i - 1][0] != name else '':<11} " + (f"{i + 1} " if numbered else "") + text
            for i, (name, text) in enumerate(rows)]


def _asked(nmap: dict, plan: dict, me: str) -> list[str]:
    """Where sleep asks what runs: the rig nodes, this node and every operator node (a `ros-cli` run lives on
    the node its verb ran from)."""
    return [n for n in dict.fromkeys([*plan, me, *_operators(nmap, plan, me)])
            if n == me or nmap.get(n, {}).get("checkout")]


def _plan_sleep(args, nmap: dict, plan: dict, me: str) -> int:
    busy = _asked(nmap, plan, me)
    never = [n for n in nmap if "estop-relay" in nmap[n].get("roles", [])]
    report = {"schema": "tatbot.rig-sleep-plan/3", "busy_check": busy, "busy_runs": list(RIG_RUNS),
              "nodes": {n: {"steps": _sleep_steps(p)} for n, p in plan.items()},
              "marker": {"required": [me, *plan], "best_effort": _operators(nmap, plan, me)}, "never": never}
    rows = [("stack", "read the drawing stack (`tatbot ros status`): refuse (exit 3) while it holds an arm that has "
                      "not landed; it stays up"),
            ("busy", f"refuse (exit 6) while a {'*, '.join(RIG_RUNS)}* run (not {' or '.join(STANDING_RUNS)}) "
                     "is open on " + ", ".join(busy))]
    rows += [(n, step) for n, p in plan.items() for step in _sleep_steps(p)]
    best = report["marker"]["best_effort"]
    rows.append(("marker", f"asleep here and on {', '.join(plan)}"
                 + (f"; best-effort on {', '.join(best)}" if best else "")))
    rows.append(("never", ", ".join([*(f"{n} (the e-stop relay)" for n in never),
                                     "the drawing stack and its arm connection", "the arm controllers"])))
    _emit(args, report, "\n".join(["tatbot rig sleep --plan (nothing executed)", *_rows(rows, False)]))
    return EXIT_OK


def _plan_wake(args, nmap: dict, plan: dict, me: str) -> int:
    rows = [(n, step) for n in _ordered(plan, nmap) for step in _wake_steps(plan[n])]
    rows += [(n, WINDOW_STEP) for n in plan if plan[n]["window"]]
    required = list(dict.fromkeys([me, *(n for n in plan if plan[n]["units"])]))
    best = [n for n in plan if n not in required] + _operators(nmap, plan, me)
    report = {"schema": "tatbot.rig-wake-plan/2", "steps": [{"node": n, "step": s} for n, s in rows],
              "then": ["tatbot ros status", "tatbot rig check"],
              "marker": {"required": required, "sleeper_required": True, "best_effort": best}}
    tail = [("then", f"the drawing stack's state and its page on the bus (`tatbot ros status`, <= {PAGE_WAIT_S} s "
                     "for a measured page), then `tatbot rig check`: notes, never the gate"),
            ("marker", f"awake on {', '.join(required)} and the node that ran `rig sleep`: one that fails keeps "
                       "the gate everywhere")]
    tail += [("", f"then {', '.join(best)}, best-effort; a rig node the sleep gated that misses it keeps refusing "
                  "there, named, and wake exits 1")] if best else []
    _emit(args, report, "\n".join(["tatbot rig wake --plan (nothing executed)", *_rows(rows, True), *_rows(tail, False)]))
    return EXIT_OK


# --- sleep ----------------------------------------------------------------------


def _stop_units(remote: Remote, p: dict, notes: list[str]) -> dict:
    """Stop the node's running units: displays first, best-effort and without waiting, then the rest."""
    before = remote.unit_states(p["units"])
    running = [u for u in p["units"] if before.get(u, {}).get("active", "inactive") not in STOPPED]
    shown = [u for u in running if u in p["display"]]
    gating = [u for u in running if u not in p["display"]]
    if shown:
        r = remote.run("sudo -n systemctl stop --no-block " + shlex.join(shown), timeout=60)
        if r.returncode:
            notes.append(f"{remote.name}: display stop failed, ignored: {r.stderr.strip()[:200]}")
    if gating:
        r = remote.run("sudo -n systemctl stop " + shlex.join(gating), timeout=120)
        if r.returncode:
            raise OSError(f"systemctl stop failed: {r.stderr.strip()[:200]}")
    after = remote.unit_states(p["units"])
    still = [u for u in gating if after.get(u, {}).get("active") not in STOPPED]
    if still:
        raise OSError("units still running: " + ", ".join(still))
    idle = len(p["units"]) - len(running)
    return {"units": after, "units_stopped": running,
            "units_verdict": f"{len(running)} stopped" + (f", {idle} already were" if idle else "")}


def _close_window(nmap: dict, remote: Remote, notes: list[str]) -> str:
    """`tatbot viewer stop` there: off the rerun-server node it closes the window and nothing else."""
    try:
        r = remote.run(f"cd {_checkout(nmap, remote)} && scripts/tatbot --no-hop viewer stop", timeout=60)
        why = (r.stderr or r.stdout).strip()[-200:] if r.returncode else None
    except (OSError, subprocess.SubprocessError) as exc:
        why = str(exc)
    if why is None:
        return "closed"
    notes.append(f"{remote.name}: window close failed, ignored: {why}")
    return "close failed"


def _sleep_node(nmap: dict, remote: Remote, p: dict, notes: list[str]) -> dict:
    node = _stop_units(remote, p, notes) if p["units"] else {}
    if p["window"]:
        node["window"] = _close_window(nmap, remote, notes)
    return node


def _publish(nmap: dict, me: str, marker: dict, targets: dict[str, bool], report: dict, run) -> list[str]:
    """Write `marker` on every target node but this one (the caller writes its own); the nodes that could not
    take it. A node already found awake is written directly, any other one is asked first."""
    missed = []
    for name, required in targets.items():
        if name == me:
            continue
        node = report["nodes"].setdefault(name, _blank())
        try:
            if name not in nmap:
                raise OSError("not in config/nodes.json")
            remote = _remote(nmap, name, me)
            if node.get("host") != "awake":
                if not remote.reachable():
                    node["host"] = "unreachable"
                    raise OSError("did not answer ssh")
                node["host"], node["note"] = "awake", "marker published" + (" (slept the rig from here)" if required else "")
            remote.write_marker(marker)
        except (OSError, subprocess.SubprocessError) as exc:
            missed.append(name)
            if required:
                node["error"] = node["error"] or f"marker: {exc}"
            else:
                node["note"] = f"marker not published: {exc}"
            run.event("marker.failed", node=name, required=required, error=str(exc))
    return missed


def do_sleep(args) -> int:
    nmap = nodes.load(REPO)
    me = nodes.this_node(nmap)
    plan = rig.plan(nmap)
    if not plan:
        print("rig sleep: no node in config/nodes.json carries a power record", file=sys.stderr)
        return EXIT_WRONG_NODE
    if args.plan:
        return _plan_sleep(args, nmap, plan, me)
    stack = _stack() if nodes.nodes_with(nmap, "ros") else None
    code = _refuse_unlanded(stack)
    if code is None:  # asked last, right before the stops: a run opened after this is not seen
        code, notes = _refuse_busy("sleep", nmap, me, _asked(nmap, plan, me))
    if code is not None:
        return code
    if nodes.nodes_with(nmap, "ros"):
        notes.insert(0, _stack_line(stack) + " — sleep leaves it as it is")
    import tatbot_runlog
    run = tatbot_runlog.init("rig-sleep", argv=sys.argv, meta={"plan": plan})
    at = rig.now()
    report = {"schema": "tatbot.rig-sleep/2", "state": "asleep", "at": at, "run_id": run.run_id, "by": me,
              "nodes": {}, "notes": notes}
    marker = {"state": "asleep", "since": at, "by": me, "run_id": run.run_id, "nodes": {}}
    failed = False
    for name, p in plan.items():
        remote, node = _remote(nmap, name, me), report["nodes"].setdefault(name, _blank())
        try:
            if not remote.reachable():
                node["host"], node["note"] = "unreachable", "did not answer ssh: nothing changed there"
                continue
            node["host"] = "awake"
            node.update(_sleep_node(nmap, remote, p, report["notes"]))
            marker["nodes"][name] = {"units_stopped": node.get("units_stopped", []), "window": node.get("window")}
            run.event("node.slept", node=name, units=node.get("units_stopped", []), window=node.get("window"))
        except (OSError, subprocess.SubprocessError) as exc:
            failed, node["error"] = True, str(exc)
            run.event("node.failed", node=name, error=str(exc))
    targets = dict.fromkeys((n for n in plan if report["nodes"][n]["host"] == "awake"), True)
    targets.update(dict.fromkeys(_operators(nmap, plan, me), False))
    failed |= any(targets[n] for n in _publish(nmap, me, marker, targets, report, run))
    rig.write_state(marker)
    (run.dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _emit(args, report, _render("sleep", report))
    run.finalize(EXIT_TOOL_FAILED if failed else EXIT_OK)
    return EXIT_TOOL_FAILED if failed else EXIT_OK


# --- wake -----------------------------------------------------------------------


def _start(remote: Remote, units: list[str], *, block: bool = True) -> None:
    r = remote.run(f"sudo -n systemctl start {'' if block else '--no-block '}" + shlex.join(units), timeout=120)
    if r.returncode:
        raise OSError(f"systemctl start failed: {r.stderr.strip()[:200]}")


def _on_bus(nmap: dict, remote: Remote, keys: list[str], notes: list[str]) -> None:
    """Wait (bounded) until every `<node>/<key>` answers on the bus, asked with the node's deployed fleetctl.
    Without one there the unit states alone decide, and a note says so."""
    if not keys:
        return
    try:
        endpoint = nodes.bus_endpoint(nmap, address="lan")
    except ValueError as exc:
        raise OSError(f"no bus to ask: {exc}") from None
    script = (f'f={_checkout(nmap, remote)}/rust/target/release/fleetctl; [ -x "$f" ] || exit 127; '
              f'"$f" --connect {shlex.quote(endpoint)} services ' + " ".join(f"--require {shlex.quote(k)}" for k in keys))
    deadline = time.monotonic() + BUS_WAIT_S
    while True:
        r = remote.run(script, timeout=30)
        if r.returncode == 0:
            return
        if r.returncode == 127:
            notes.append(f"{remote.name}: no deployed fleetctl in its checkout; bus presence not checked")
            return
        if time.monotonic() >= deadline:
            try:
                missing = json.loads(r.stdout).get("missing") or json.loads(r.stdout).get("reason")
            except ValueError:
                missing = (r.stderr or r.stdout).strip()[-200:]
            raise OSError(f"not on the bus {BUS_WAIT_S} s after the start: {missing}")
        time.sleep(2)


def _wait_active(remote: Remote, units: list[str]) -> dict:
    """The units' states once every one is active; raises after START_WAIT_S."""
    deadline = time.monotonic() + START_WAIT_S
    while True:
        states = remote.unit_states(units)
        waiting = [u for u in units if states.get(u, {}).get("active") != "active"]
        if not waiting:
            return states
        if time.monotonic() >= deadline:
            raise OSError(f"not active {START_WAIT_S} s after the start: " + ", ".join(
                f"{u} {states.get(u, {}).get('active')}" for u in waiting))
        time.sleep(3)


def unhealthy(units: list[str], first: dict, second: dict) -> list[str]:
    """Units not active at the second read, or restarted between the two."""
    return [u for u in units if second.get(u, {}).get("active") != "active"
            or (second[u]["restarts"] or 0) > (first.get(u, {}).get("restarts") or 0)]


def _wake_services(nmap: dict, remote: Remote, p: dict, notes: list[str]) -> dict:
    """The node's enabled units in three stages: the bus router until its key answers, the rest until each is
    active, on the bus and steady, then the displays best-effort. A unit not enabled was off before the
    night too, and is left as it is."""
    states = remote.unit_states(p["bus"] + p["start"] + p["display"])
    off = [u for u in states if states[u]["enabled"] not in ENABLED]
    if off:
        notes.append(f"{remote.name}: not enabled, left as it was: "
                     + ", ".join(f"{u} ({states[u]['enabled'] or states[u]['loaded']})" for u in off))
    bus, rest = [u for u in p["bus"] if u not in off], [u for u in p["start"] if u not in off]
    if bus:
        _say(f"{remote.name}: starting the bus router")
        _start(remote, bus)
        _on_bus(nmap, remote, _keys(p, bus), notes)
    if rest:
        _say(f"{remote.name}: starting {' '.join(rest)}")
        _start(remote, rest, block=False)
    first = _wait_active(remote, bus + rest)
    _on_bus(nmap, remote, _keys(p, bus + rest), notes)
    time.sleep(STEADY_S)
    second = remote.unit_states(bus + rest)
    bad = unhealthy(bus + rest, first, second)
    if bad:
        raise OSError("units not healthy after start: " + ", ".join(bad) + f" — `tatbot logs last fleet-service` on {remote.name}")
    shown = [u for u in p["display"] if u not in off] + p["ensure"]
    if shown:
        r = remote.run("sudo -n systemctl start --no-block " + shlex.join(shown), timeout=60)
        if r.returncode:
            notes.append(f"{remote.name}: display start failed, ignored: {r.stderr.strip()[:200]}")
    return {"units": second, "units_started": bus + rest, "units_verdict": f"{len(bus + rest)} running"}


def _open_window(nmap: dict, remote: Remote) -> str:
    """`tatbot viewer open` there, fullscreen behind whatever has the focus, and only over the node's direct
    cable to the hub when it has one: on the rig port the window's stream (and a reconnect's replay of the
    hub's whole buffer) delays the arm controller's packets queued behind it."""
    co = _checkout(nmap, remote)
    found = remote.run(f"cd {co} && PYTHONPATH=scripts/lib python3 -m tatbot_cli.viewer_path endpoint", timeout=20)
    path = found.stdout.strip().partition("\t")[2]
    if viewer_path.link(nmap, remote.name) and not path.startswith("direct link"):
        return (f"not opened: the hub would stream over {path or 'an unknown path'}; once the direct link answers, "
                f"`tatbot viewer open {' '.join(rig.WINDOW_OPEN)}` on {remote.name}")
    r = remote.run(f"cd {co} && scripts/tatbot --no-hop viewer open " + shlex.join(rig.WINDOW_OPEN), timeout=90)
    if r.returncode:
        return f"not opened (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-200:]}"
    return (r.stdout.strip().splitlines() or ["opened"])[-1]


def _wake_node(nmap: dict, me: str, name: str, p: dict, node: dict, notes: list[str]) -> dict | None:
    """Wake one node's services; the asleep marker it held, if any. A window-only node that does not answer
    is a note: its window is a display, and its marker keeps refusing there until a later wake reaches it."""
    remote = _remote(nmap, name, me)
    if not remote.reachable():
        node["host"] = "unreachable"
        if p["units"]:
            raise OSError("did not answer ssh")
        node["note"] = "did not answer ssh: its window and marker stay as they were until a later `rig wake`"
        return None
    node["host"] = "awake"
    found = remote.read_marker()
    if p["units"]:
        node.update(_wake_services(nmap, remote, p, notes))
    return found if found and found.get("state") == "asleep" else None


def _wake_all(nmap: dict, me: str, plan: dict, report: dict, run) -> tuple[bool, dict, dict]:
    """(failed, units started per node, the asleep markers the rig nodes held). The bus router's node goes
    first; when it fails, no other node's services are started (they would only churn through restarts)."""
    failed, started, slept, router_down = False, {}, {}, None
    for name in _ordered(plan, nmap):
        node = report["nodes"].setdefault(name, _blank())
        try:
            if router_down and plan[name]["units"]:
                raise OSError(f"not started: the bus router {router_down} is not up")
            found = _wake_node(nmap, me, name, plan[name], node, report["notes"])
            slept.update({name: found} if found else {})
            started[name] = node.get("units_started", [])
            run.event("node.woken", node=name, units=started[name])
        except (OSError, subprocess.SubprocessError) as exc:
            failed, node["error"] = True, str(exc)
            router_down = router_down or (name if "bus-router" in nmap[name].get("roles", []) else None)
            run.event("node.failed", node=name, error=str(exc))
    for name in (n for n in plan if plan[n]["window"] and report["nodes"][n]["host"] == "awake"):
        try:
            report["nodes"][name]["window"] = _open_window(nmap, _remote(nmap, name, me))
        except (OSError, subprocess.SubprocessError) as exc:
            report["nodes"][name]["window"] = f"not opened: {exc}"
    return failed, started, slept


def _provenance(previous: dict | None, slept: dict, me: str, notes: list[str]) -> dict | None:
    """The sleep this wake undoes, named first in the notes: this node's marker, else (the sleep ran elsewhere)
    the one the rig nodes hold."""
    if previous:
        lines = [f"asleep since {previous['since']} (rig sleep {previous.get('run_id')} from {previous.get('by')})"]
        if previous.get("wake_failed"):
            lines.append(f"last wake attempt failed: rig wake {previous['wake_failed'].get('run_id')}")
    elif slept:
        previous = next(iter(slept.values()))
        lines = [f"no marker on {me}; {', '.join(slept)} asleep since {previous.get('since')} "
                 f"(rig sleep {previous.get('run_id')} from {previous.get('by')})"]
    else:
        lines = []
    notes[:0] = lines
    return previous


def _wake_markers(nmap: dict, me: str, plan: dict, report: dict, run, ctx: dict) -> tuple[dict | None, bool, list[str]]:
    """Publish the wake's outcome; (this node's marker, None to leave it as it is; failed; the nodes whose gate
    stays set although the wake passed).

    Required first: each awake `services` node and the node that ran `rig sleep` (the marker's `by`, here or on
    the rig nodes); one that fails keeps the gate everywhere. Then, best-effort, every other rig node and
    operator node: a rig node the sleep gated (a window node that does not answer now) keeps refusing there
    until a later wake reaches it, and is named. A wake that fails on a rig that was not asleep sets no gate."""
    previous, slept, failed = ctx["previous"], ctx["slept"], ctx["failed"]
    if failed and not previous:
        report["notes"].append("the rig was not asleep: a failed wake sets no gate")
        return None, True, []

    def still_asleep() -> dict:
        """The gate as sleep left it: when the rig slept, from where and which run stay the sleep's own, so a
        retry's note still names the sleep, not the failed wake; the attempt is recorded alongside."""
        kept = {k: v for k, v in previous.items() if k != "schema"}
        return {**kept, "state": "asleep", "wake_failed": {"run_id": run.run_id, "at": ctx["at"], "by": me,
                                                           "nodes": ctx["done"]}}

    marker = still_asleep() if failed else {"state": "awake", "since": ctx["at"], "by": me, "run_id": run.run_id,
                                            "slept": (previous or {}).get("run_id"), "nodes": ctx["done"]}
    sleepers = {m["by"] for m in (previous, *slept.values()) if m and m.get("by")}
    awake = [n for n in plan if report["nodes"][n].get("host") == "awake"]
    required = dict.fromkeys([*(n for n in awake if plan[n]["units"]), *sorted(sleepers)], True)
    if _publish(nmap, me, marker, required, report, run) and not failed:
        if not previous:
            report["notes"].append("the rig was not asleep: a failed wake sets no gate")
            return None, True, []
        failed, marker = True, still_asleep()
    rest = [n for n in [*plan, *_operators(nmap, plan, me)] if n not in required]
    missed = _publish(nmap, me, marker, dict.fromkeys(rest, False), report, run)
    gated = {*slept, *(previous or {}).get("nodes", {})}
    return marker, failed, [] if failed else [n for n in missed if n in gated]


def do_wake(args) -> int:
    nmap = nodes.load(REPO)
    me = nodes.this_node(nmap)
    plan = rig.plan(nmap)
    if not plan:
        print("rig wake: no node in config/nodes.json carries a power record", file=sys.stderr)
        return EXIT_WRONG_NODE
    if args.plan:
        return _plan_wake(args, nmap, plan, me)
    import tatbot_runlog
    run = tatbot_runlog.init("rig-wake", argv=sys.argv)
    at = rig.now()
    report = {"schema": "tatbot.rig-wake/3", "state": "awake", "at": at, "run_id": run.run_id, "by": me,
              "nodes": {}, "notes": []}
    previous = rig.asleep()
    failed, started, slept = _wake_all(nmap, me, plan, report, run)
    previous = _provenance(previous, slept, me, report["notes"])
    if not failed:
        report["notes"] += ([_stack_after_wake()] if nodes.nodes_with(nmap, "ros") else []) + _rig_check_notes()
    done = {n: {"units_started": started.get(n, [])} for n in plan}
    marker, failed, kept = _wake_markers(nmap, me, plan, report, run,
                                         {"previous": previous, "slept": slept, "failed": failed, "at": at, "done": done})
    if failed:
        report["state"] = "wake_failed"
    if kept:
        report["gate_kept"] = kept
        report["notes"].append(f"the sleep gate stays set on {', '.join(kept)}: motion verbs refuse there until "
                               "`tatbot rig wake` reaches it")
    if marker:
        rig.write_state(marker)
    (run.dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _emit(args, report, _render("wake", report))
    code = EXIT_TOOL_FAILED if failed or kept else EXIT_OK
    run.finalize(code)
    return code


# --- status ---------------------------------------------------------------------


def _status_node(remote: Remote, p: dict) -> dict:
    node = {}
    if p["units"]:
        node["units"] = remote.unit_states(p["units"])
        active = sum(1 for s in node["units"].values() if s["active"] == "active")
        node["units_verdict"] = f"{active}/{len(node['units'])} active"
    kept = {**remote.unit_states(p["kept"]), **remote.unit_states(p["kept_user"], user=True)}
    down = [u.removesuffix(".service") for u, s in kept.items() if s["active"] != "active"]
    node["kept_verdict"] = f"stays up {len(kept) - len(down)}/{len(kept)}" + (f" (down: {' '.join(down)})" if down else "")
    if p["window"]:
        node["window"] = remote.unit_states([rig.WINDOW_UNIT], user=True).get(rig.WINDOW_UNIT, {}).get("active")
    marker = remote.read_marker()
    if marker:
        node["note"] = f"marker {marker.get('state')} since {marker.get('since')}"
    return node


def do_status(args) -> int:
    nmap = nodes.load(REPO)
    me = nodes.this_node(nmap)
    plan = rig.plan(nmap)
    try:
        local = rig.read_state()
    except (OSError, ValueError) as exc:
        local = {"state": "unreadable", "error": str(exc)}
    report = {"schema": "tatbot.rig-status/2", "state": (local or {}).get("state", "no marker"),
              "at": rig.now(), "marker": local, "nodes": {}}
    for name, p in plan.items():
        remote, node = _remote(nmap, name, me), report["nodes"].setdefault(name, _blank())
        if not remote.reachable(timeout=4):
            node["host"] = "unreachable"
            continue
        node["host"] = "awake"
        try:
            node.update(_status_node(remote, p))
        except (OSError, subprocess.SubprocessError) as exc:
            node["error"] = str(exc)
    _emit(args, report, _render("status", report))
    return EXIT_OK


# --- off ------------------------------------------------------------------------

OFF_WAIT_S = 90


def _off_land(nmap: dict, me: str, ros: str | None, run) -> tuple[int, str | None]:
    """`tatbot ros down` on the ros node: an arm that has not landed can drop when
    its driver dies with its host, so a failed landing powers nothing off."""
    if not ros:
        return EXIT_OK, None
    if not _remote(nmap, ros, me).reachable():
        return EXIT_OK, f"the ros node {ros} did not answer ssh: nothing to land there"
    print(f"rig off: landing the arms and stopping the drawing stack on {ros}", file=sys.stderr, flush=True)
    r = subprocess.run([str(REPO / "scripts/tatbot"), "ros", "down"], capture_output=True, text=True,
                       timeout=300, check=False)
    run.event("ros.down", code=r.returncode)
    if r.returncode:
        print(f"rig off refused: `tatbot ros down` failed (exit {r.returncode}), so nothing was shut down:\n"
              + (r.stderr or r.stdout).strip()[-1500:], file=sys.stderr)
        return r.returncode, None
    return EXIT_OK, "arms landed and drawing stack stopped"


def _off_host(remote: Remote) -> dict:
    """Shut one host down and wait until it stops answering ssh."""
    node = _blank()
    if not remote.reachable():
        node["host"], node["note"] = "unreachable", "did not answer ssh: already off?"
        return node
    print(f"rig off: shutting down {remote.name}", file=sys.stderr, flush=True)
    r = remote.run("sudo -n systemd-run --on-active=2 --quiet /usr/bin/systemctl poweroff")
    if r.returncode:
        raise OSError(f"could not schedule poweroff: {r.stderr.strip()[:200]}")
    deadline = time.monotonic() + OFF_WAIT_S
    while remote.reachable(timeout=3):
        if time.monotonic() >= deadline:
            raise OSError(f"still answers ssh {OFF_WAIT_S} s after the poweroff request")
        time.sleep(3)
    node["host"] = "off"
    return node


def do_off(args) -> int:
    """Land the arms, then shut down every `off` node so the strip can be cut."""
    nmap = nodes.load(REPO)
    me = nodes.this_node(nmap)
    order = [n for n in rig.off_order(nmap, me) if not (args.no_self and n == me)]
    if not order:
        print("rig off: no node in config/nodes.json is marked \"off\": true", file=sys.stderr)
        return EXIT_WRONG_NODE
    ros = (nodes.nodes_with(nmap, "ros") or [None])[0]
    if args.plan:
        report = {"schema": "tatbot.rig-off-plan/1", "land": f"tatbot ros down on {ros}" if ros else None,
                  "shutdown_order": order}
        _emit(args, report, "tatbot rig off --plan (nothing executed)\n"
              + (f"  land     tatbot ros down on {ros}\n" if ros else "") + "  shutdown " + " -> ".join(order))
        return EXIT_OK
    code, notes = _refuse_busy("off", nmap, me, [*order, me, *_operators(nmap, order, me)], every=tuple(order))
    if code is not None:
        return code

    import tatbot_runlog
    run = tatbot_runlog.init("rig-off", argv=sys.argv, meta={"order": order})
    report = {"schema": "tatbot.rig-off/1", "state": "off", "at": rig.now(), "run_id": run.run_id, "by": me,
              "nodes": {}, "notes": notes}
    code, note = _off_land(nmap, me, ros, run)
    if code:
        run.finalize(code)
        return code
    report["notes"] += [note] if note else []
    failed = False
    for name in (n for n in order if n != me):
        try:
            report["nodes"][name] = _off_host(_remote(nmap, name, me))
            run.event("node.off", node=name, host=report["nodes"][name]["host"])
        except (OSError, subprocess.SubprocessError) as exc:
            failed = True
            report["nodes"][name] = {**_blank(), "host": "on", "error": str(exc)}
            run.event("node.failed", node=name, error=str(exc))
    goes_self = me in order and not failed
    if me in order:
        report["nodes"][me] = {**_blank(), "host": "powers off in 5 s" if goes_self else "left on: another node did not shut down"}
    (run.dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _emit(args, report, _render("off", report))
    run.finalize(EXIT_TOOL_FAILED if failed else EXIT_OK)
    if goes_self:
        sys.stdout.flush()
        subprocess.run(["sudo", "-n", "systemd-run", "--on-active=5", "--quiet", "/usr/bin/systemctl", "poweroff"],
                       check=False)
    return EXIT_TOOL_FAILED if failed else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("sleep", help="stop the camera and tracker services, close the demo window")
    s.add_argument("--plan", action="store_true", help="print what sleep would do per node; no ssh")
    w = sub.add_parser("wake", help="start the services in stages, verify them on the bus, reopen the window")
    w.add_argument("--plan", action="store_true", help="print what wake would do per node; no ssh")
    sub.add_parser("status", help="marker, host reachability and unit state per rig node")
    o = sub.add_parser("off", help="land the arms, then shut down every node marked off")
    o.add_argument("--no-self", action="store_true", help="leave the node running this command on")
    o.add_argument("--plan", action="store_true", help="print the landing and shutdown order; no ssh")
    args = p.parse_args(argv)
    return {"sleep": do_sleep, "wake": do_wake, "status": do_status, "off": do_off}[args.verb](args)


if __name__ == "__main__":
    sys.exit(main())
