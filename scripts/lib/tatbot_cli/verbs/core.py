"""status · schema · check · logs · node — the verbs that are about the CLI and the fleet."""

from __future__ import annotations

import json
import math
import shlex
import sys
from pathlib import Path

from tatbot_cli import EXIT_OK, EXIT_USAGE, nodes
from tatbot_cli.registry import OFFLINE, REMOTE, SENSOR, Plan, verb
from tatbot_cli.verbs._common import sh

# --- status ----------------------------------------------------------------------


def _timeout(value):
    import argparse
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= 300:
        raise argparse.ArgumentTypeError("timeout must be finite and between 0 (exclusive) and 300 seconds")
    return seconds


_timeout.value_type = float
_timeout.bounds = (0.0, 300.0)
_timeout.exclusive_bounds = (True, False)


def _status_args(p):
    p.add_argument("--cams", action="store_true", help="also ping the five PoE cameras")
    p.add_argument("--fleet", action="store_true", help="collect local observations from configured nodes over read-only SSH (a node with no roles is reported retired, not probed)")
    p.add_argument("--timeout-s", type=_timeout, default=15.0, help="finite overall deadline, 0 < seconds <= 300 (default 15; at most four fleet collectors)")
    p.add_argument("--schema-version", type=int, choices=(1, 2), default=2, help="JSON result version; 1 is the legacy field projection")
    p.add_argument("--no-service-discovery", action="store_true", help="local collector only: fleet caller collects the shared service registry once")


def _status_effects(effects, ns, rest):
    return effects if ns.fleet else effects - {"remote_exec"}


@verb(refine_effects=_status_effects, effects=('read_files', 'network', 'sensor_read', 'remote_exec'), native=True, output="json", noun="status", verb="", tier=SENSOR, summary="node, arms, e-stop, tool, ink, runs, locks, policy server",
      args=_status_args, example=("--json",), doc="docs/cli.md",
      invariants=("Local observations by default; --fleet uses bounded read-only collectors, no sync, tty, or service start.",
                  "Pings controllers only on their arm node; cameras only with --cams.",
                  "The tool shown is the last touch-off record."))
def status(ctx, ns, rest):
    from tatbot_cli import status as st
    collector = st.collect_fleet if ns.fleet else st.collect
    options = {} if ns.fleet else {'service_discovery': not ns.no_service_discovery}
    s = collector(ctx.repo, cams=ns.cams, timeout_s=ns.timeout_s, **options)
    payload = st.legacy(s) if ns.schema_version == 1 else s
    print(json.dumps(payload, indent=2) if ctx.json else st.render(s))
    return EXIT_OK if s["complete"] else 1


# --- schema ----------------------------------------------------------------------


def _schema_args(p):
    p.add_argument("--schema-version", type=int, choices=(1, 2), default=2, help="JSON schema version; 1 retains the legacy field projection")
    p.add_argument("--md", action="store_true", help="Markdown (what docs/cli.md is generated from)")
    p.add_argument("--full", action="store_true",
                   help="with --md: the complete private table (docs/cli-full.md), not the public subset")
    p.add_argument("--disposition", action="store_true",
                   help="the whole-CLI command audit as Markdown: the audience of every command "
                        "and the reason it sits there")
    p.add_argument("--check", metavar="PATH", help="exit 1 if PATH differs from the generated Markdown")


def _schema_output(ns, rest):
    text = ns is not None and (ns.md or ns.disposition) and not ns.check
    return "text" if text else "json"


def _generated(sc, ctx, ns) -> str:
    return (sc.as_disposition(ctx.repo) if ns.disposition
            else sc.as_markdown(ctx.repo, full=ns.full))


@verb(effects=('read_files',), select_output=_schema_output, output_modes={"default": "json", "--md": "text", "--disposition": "text", "--check PATH": "json"}, native=True, output="json", noun="schema", verb="", tier=OFFLINE, summary="command tree, tiers, exit codes and nodes",
      args=_schema_args, example=("--json",), doc="docs/cli.md")
def schema(ctx, ns, rest):
    from tatbot_cli import schema as sc
    if ns.check:
        want = _generated(sc, ctx, ns)
        try:
            have = Path(ns.check).read_text()
        except OSError:
            have = ""
        if have != want:
            flags = "--disposition" if ns.disposition else "--md --full" if ns.full else "--md"
            print(f"{ns.check} is stale — regenerate: scripts/tatbot schema {flags} > {ns.check}", file=sys.stderr)
            return 1
        print(json.dumps({"schema": "tatbot.schema-check/1", "path": ns.check, "current": True})
              if ctx.json else f"{ns.check} is current")
        return EXIT_OK
    text = _generated(sc, ctx, ns) if (ns.md or ns.disposition) else sc.as_json(ctx.repo, ns.schema_version)
    sys.stdout.write(text.rstrip("\n") + "\n")
    return EXIT_OK


# --- check / logs (pure aliases) ------------------------------------------------


@verb(effects=('read_files', 'write_files', 'delete_files', 'network', 'start_process'), noun="check", verb="", tier=OFFLINE, summary="run every check this node can (PASS/FAIL/SKIP)",
      wraps=("scripts/check",), passthrough="scripts/check", example=("--list",), doc="docs/development.md")
def check(ctx, ns, rest):
    return sh(ctx, "scripts/check", *rest)


LOG_OUTPUT_MODES = {"default": "text", "list": "json", "last": "json"}


def _logs_output(ns, rest):
    return LOG_OUTPUT_MODES.get(rest[0] if rest else "", "text")


def _logs_effects(effects, ns, rest):
    if not rest:
        return effects
    reads = {"root", "list", "last", "show", "tail", "count", "du"}
    if rest[0] in reads or (rest[0] == "prune" and "--yes" not in rest):
        return effects - {"write_files", "delete_files"}
    return effects


@verb(refine_effects=_logs_effects, select_output=_logs_output, output_modes=LOG_OUTPUT_MODES, effects=('read_files', 'write_files', 'delete_files', 'network', 'remote_exec'), native=True, output="json", noun="logs", verb="", tier=OFFLINE, summary="list / last / show / tail / fetch / count / du / prune run logs",
      wraps=("scripts/tatbot-logs", "scripts/lib/tatbot_runlog.py"), passthrough="scripts/tatbot-logs",
      example=("last", "rollout"), doc="docs/run_logs.md",
      invariants=("Debug from the log, never from the operator: `tatbot logs last <workflow>` first.",
                  "`logs count <workflow> --expect N --before M` reconciles launch COUNT against the index before "
                  "anyone calls a launch uncommanded."))
def logs(ctx, ns, rest):
    sys.path.insert(0, str(ctx.repo / "scripts" / "lib"))
    import tatbot_runlog
    if ctx.json:
        if not rest or rest[0] not in ("list", "last"):
            from tatbot_cli.cli import UsageError
            raise UsageError("logs supports global --json for list and last; other subcommands are text-only")
        if "--json" not in rest:
            rest = [*rest, "--json"]
    return int(tatbot_runlog.main(list(rest)) or 0)


# --- node ------------------------------------------------------------------------


# Two verbs (2026-09-02; four before): `node list [<node>]` is the readable form
# of config/nodes.json (the old `node info` is the optional positional), and
# `node run <node> [cmd…]` is the only arbitrary-shell path into a node's
# checkout — with no command it opens an interactive shell there (the old
# `node ssh`).


def _node_list_args(p):
    p.add_argument("node", nargs="?", help="one node: print its full record (JSON)")


@verb(effects=('read_files',), native=True, output="json", noun="node", verb="list", tier=OFFLINE, summary="every node, its ssh target and roles; `node list <node>` is one node's record",
      args=_node_list_args, example=())
def node_list(ctx, ns, rest):
    nmap = nodes.load(ctx.repo)
    if ns.node:
        if ns.node not in nmap:
            print(f"unknown node {ns.node} (known: {', '.join(nmap)})", file=sys.stderr)
            return EXIT_USAGE
        print(json.dumps({ns.node: nmap[ns.node]}, indent=2))
        return EXIT_OK
    if ctx.json:
        print(json.dumps(nmap, indent=2))
        return EXIT_OK
    for n, rec in nmap.items():
        me = " (this node)" if n == ctx.node else ""
        print(f"{n:<11} {rec.get('ssh', ''):<24} {rec.get('arch', ''):<8} {', '.join(rec.get('roles', []))}{me}")
    return EXIT_OK


def _node_run_args(p):
    p.add_argument("node")
    p.add_argument("cmd", nargs="*", help="command to run in the node's checkout; none = an interactive shell there (ssh -t)")


@verb(effects=('network', 'remote_exec', 'arbitrary_command'), noun="node", verb="run", tier=REMOTE,
      summary="run a shell command in a node's checkout, or with no command open an interactive shell there",
      args=_node_run_args, passthrough="remote shell argv",
      example=(nodes.example_node(), "--", "git", "rev-parse", "--short", "HEAD"))
def node_run(ctx, ns, rest):
    nmap = nodes.load(ctx.repo)
    target = nodes.ssh_target(nmap, ns.node)
    if not target:
        print(f"unknown node {ns.node} (known: {', '.join(nmap)})", file=sys.stderr)
        return EXIT_USAGE
    cmd = [*ns.cmd, *rest]
    rec = nmap[ns.node]
    # `checkout: null` marks a node with no git checkout (the palette Pi): run in its home.
    checkout = None if "checkout" in rec and rec["checkout"] is None else rec.get("checkout") or "~/tatbot"
    cd = f"cd {checkout} && " if checkout else ""
    where = f"{ns.node}'s checkout {checkout}" if checkout else f"{ns.node}'s home directory (it has no checkout)"
    if not cmd:
        # Interactive: a tty is only wanted here, and Verb.tty is static (it is
        # about --on hops), so the -t is built into the argv for this case alone.
        return Plan(argv=["ssh", "-t", target, f"{cd}exec bash -l"],
                    notes=[f"no command given: an interactive login shell in {where}"])
    return Plan(argv=["ssh", "-o", "BatchMode=yes", target, cd + " ".join(shlex.quote(c) for c in cmd)],
                notes=[] if checkout else [f"runs in {where}"])


@verb(effects=('read_files',), visibility="public", native=True, output="text",
      noun="completion", verb="bash", tier=OFFLINE,
      summary="print Bash completion generated from canonical command metadata",
      example=(), doc="docs/cli.md",
      invariants=("Prints shell code; installing or sourcing it is explicit and never edits startup files.",))
def completion_bash(ctx, ns, rest):
    from tatbot_cli.completion import bash
    print(bash(ctx.repo), end="")
    return EXIT_OK
