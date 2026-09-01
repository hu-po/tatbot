"""status · schema · check · logs · node — the verbs that are about the CLI and the fleet."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from tatbot_cli import EXIT_OK, EXIT_USAGE, nodes
from tatbot_cli.registry import OFFLINE, SENSOR, Plan, verb
from tatbot_cli.verbs._common import sh

# --- status ----------------------------------------------------------------------


def _status_args(p):
    p.add_argument("--cams", action="store_true", help="also ping the five PoE cameras")


@verb(noun="status", verb="", tier=SENSOR, summary="node, arms, e-stop, tool, ink, runs, locks, policy server",
      args=_status_args, example=("--json",), doc="docs/cli.md",
      invariants=("Pings the two arm controllers (1 s each); cameras only with --cams.",
                  "The tool shown is the last touch-off's, not proof of what is fitted — always state --ee-tool."))
def status(ctx, ns, rest):
    from tatbot_cli import status as st
    if ctx.dry_run:
        return Plan(argv=["<native>", "tatbot", "status"], notes=["pings the profile's arms; reads run index, locks, serve state"])
    s = st.collect(ctx.repo, cams=ns.cams)
    print(json.dumps(s, indent=2) if ctx.json else st.render(s))
    return EXIT_OK


# --- schema ----------------------------------------------------------------------


def _schema_args(p):
    p.add_argument("--md", action="store_true", help="Markdown (what docs/cli.md is generated from)")
    p.add_argument("--check", metavar="PATH", help="exit 1 if PATH differs from the generated Markdown")


@verb(noun="schema", verb="", tier=OFFLINE, summary="command tree, tiers, exit codes and nodes",
      args=_schema_args, example=("--json",), doc="docs/cli.md")
def schema(ctx, ns, rest):
    from tatbot_cli import schema as sc
    if ns.check:
        want = sc.as_markdown(ctx.repo)
        try:
            have = Path(ns.check).read_text()
        except OSError:
            have = ""
        if have != want:
            print(f"{ns.check} is stale — regenerate: scripts/tatbot schema --md > {ns.check}", file=sys.stderr)
            return 1
        print(f"{ns.check} is current")
        return EXIT_OK
    sys.stdout.write((sc.as_markdown(ctx.repo) if ns.md else sc.as_json(ctx.repo)).rstrip("\n") + "\n")
    return EXIT_OK


# --- check / logs (pure aliases) ------------------------------------------------


@verb(noun="check", verb="", tier=OFFLINE, summary="run every check this node can (PASS/FAIL/SKIP)",
      wraps=("scripts/check",), passthrough="scripts/check", example=("--list",), doc="docs/development.md")
def check(ctx, ns, rest):
    return sh(ctx, "scripts/check", *rest)


@verb(noun="logs", verb="", tier=OFFLINE, summary="list / last / show / tail / fetch / du / prune run logs",
      wraps=("scripts/tatbot-logs", "scripts/lib/tatbot_runlog.py"), passthrough="scripts/tatbot-logs",
      example=("last", "rollout"), doc="docs/run_logs.md",
      invariants=("Debug from the log, never from the operator: `tatbot logs last <workflow>` first.",))
def logs(ctx, ns, rest):
    if ctx.dry_run:
        return Plan(argv=["<native>", "tatbot_runlog.main", *rest], notes=["in-process; scripts/tatbot-logs is a shim for this"])
    sys.path.insert(0, str(ctx.repo / "scripts" / "lib"))
    import tatbot_runlog
    return int(tatbot_runlog.main(list(rest)) or 0)


# --- node ------------------------------------------------------------------------


@verb(noun="node", verb="list", tier=OFFLINE, summary="every node, its ssh target and roles", example=())
def node_list(ctx, ns, rest):
    nmap = nodes.load(ctx.repo)
    if ctx.json:
        print(json.dumps(nmap, indent=2))
        return EXIT_OK
    for n, rec in nmap.items():
        me = " (this node)" if n == ctx.node else ""
        print(f"{n:<11} {rec.get('ssh', ''):<24} {rec.get('arch', ''):<8} {', '.join(rec.get('roles', []))}{me}")
    return EXIT_OK


def _node_arg(p):
    p.add_argument("node")


@verb(noun="node", verb="info", tier=OFFLINE, summary="one node's record", args=_node_arg, example=(nodes.example_node(),))
def node_info(ctx, ns, rest):
    nmap = nodes.load(ctx.repo)
    if ns.node not in nmap:
        print(f"unknown node {ns.node} (known: {', '.join(nmap)})", file=sys.stderr)
        return EXIT_USAGE
    print(json.dumps({ns.node: nmap[ns.node]}, indent=2))
    return EXIT_OK


@verb(noun="node", verb="ssh", tier=OFFLINE, summary="interactive ssh to a node", args=_node_arg, example=(nodes.example_node(),), tty=True)
def node_ssh(ctx, ns, rest):
    nmap = nodes.load(ctx.repo)
    target = nodes.ssh_target(nmap, ns.node)
    if not target:
        print(f"unknown node {ns.node} (known: {', '.join(nmap)})", file=sys.stderr)
        return EXIT_USAGE
    return Plan(argv=["ssh", target, *rest])


def _node_run_args(p):
    p.add_argument("node")
    p.add_argument("cmd", nargs="*", help="command to run in the node's checkout")


@verb(noun="node", verb="run", tier=OFFLINE, summary="run a shell command in a node's checkout", args=_node_run_args,
      example=(nodes.example_node(), "--", "git", "rev-parse", "--short", "HEAD"))
def node_run(ctx, ns, rest):
    nmap = nodes.load(ctx.repo)
    target = nodes.ssh_target(nmap, ns.node)
    if not target:
        print(f"unknown node {ns.node} (known: {', '.join(nmap)})", file=sys.stderr)
        return EXIT_USAGE
    cmd = [*ns.cmd, *rest]
    if not cmd:
        print("node run: give a command (after -- if it has flags)", file=sys.stderr)
        return EXIT_USAGE
    checkout = nmap[ns.node].get("checkout") or "~/tatbot"
    return Plan(argv=["ssh", "-o", "BatchMode=yes", target, f"cd {checkout} && " + " ".join(shlex.quote(c) for c in cmd)])
