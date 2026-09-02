"""Root parser, dispatch, gates, node routing, --dry-run/--explain/--json."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

from tatbot_cli import (
    EXIT_GATE_REFUSED,
    EXIT_NAMES,
    EXIT_OK,
    EXIT_USAGE,
    EXIT_WRONG_NODE,
    __version__,
    gates,
    nodes,
)
from tatbot_cli.registry import (
    MOTION_AUTO,
    MOTION_TIERS,
    NOUN_ORDER,
    NOUN_SUMMARY,
    TIER_MEANING,
    Ctx,
    Plan,
    Verb,
    nouns,
    repo_root,
    verbs_of,
)

GLOBAL_HELP = """\
global flags (before the noun):
  --json            machine-readable output; refusals are one JSON line on stderr
  --dry-run         resolve node, tool and gates, print the exact command, exec nothing
  --on <node>       run this command on another node over ssh (config/nodes.json)
  --ee-tool <id>    the tool in the mount — stated, never inferred (or TATBOT_EE_TOOL)
  --explain         print the verb's tier, gates, invariants and docs, then exit
  -q / -v           quieter / louder
  --version

exit codes:
  0 ok   1 tool failed   2 usage   3 safety gate refused   4 wrong node
  5 hardware unreachable   6 busy (arm held, training lock, SWEEP_PAUSE)
"""


def _root_usage(out=sys.stdout) -> None:
    print("usage: tatbot [global flags] <noun> [verb] [args] [-- passthrough]\n", file=out)
    print("nouns, in the order an operator meets them in a day:", file=out)
    for n in nouns():
        tiers = sorted({v.tier for v in verbs_of(n)}, key=lambda t: list(TIER_MEANING).index(t))
        print(f"  {n:<9} {NOUN_SUMMARY.get(n, ''):<52} [{', '.join(tiers)}]", file=out)
    print(file=out)
    print(GLOBAL_HELP, file=out)
    print("`tatbot <noun> --help` lists its verbs; `tatbot <noun> <verb> --explain` says what it can do.", file=out)


BARE_GLOBALS = {"--json": "json", "--dry-run": "dry_run", "--explain": "explain", "--no-hop": "no_hop"}


def parse_globals(argv: list[str]) -> tuple[dict, str | None, list[str]]:
    g: dict = {"json": False, "dry_run": False, "on": None, "ee_tool": None, "explain": False,
               "quiet": False, "verbose": False, "no_hop": False, "help": False, "version": False}
    # The bare global flags are accepted anywhere before `--`, so
    # `tatbot rollout run --explain` reads as naturally as `tatbot --explain rollout run`.
    head, tail = split_dashdash(argv)
    kept = []
    for a in head:
        if a in BARE_GLOBALS:
            g[BARE_GLOBALS[a]] = True
        else:
            kept.append(a)
    argv = kept + (["--", *tail] if "--" in argv else [])
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            g["help"] = True
        elif a == "--version":
            g["version"] = True
        elif a == "--json":
            g["json"] = True
        elif a == "--dry-run":
            g["dry_run"] = True
        elif a == "--explain":
            g["explain"] = True
        elif a == "--no-hop":
            g["no_hop"] = True
        elif a == "-q":
            g["quiet"] = True
        elif a == "-v":
            g["verbose"] = True
        elif a == "--on" or a == "--ee-tool":
            if i + 1 >= len(argv):
                raise SystemExit(_usage_error(f"{a} needs a value"))
            g["on" if a == "--on" else "ee_tool"] = argv[i + 1]
            i += 1
        elif a.startswith("--on="):
            g["on"] = a[5:]
        elif a.startswith("--ee-tool="):
            g["ee_tool"] = a[10:]
        elif a.startswith("-"):
            raise SystemExit(_usage_error(f"unknown global flag {a} (global flags go before the noun)"))
        else:
            return g, a, argv[i + 1:]
        i += 1
    return g, None, []


def _usage_error(msg: str) -> int:
    print(f"tatbot: {msg}", file=sys.stderr)
    return EXIT_USAGE


def split_dashdash(args: list[str]) -> tuple[list[str], list[str]]:
    if "--" in args:
        i = args.index("--")
        return args[:i], args[i + 1:]
    return args, []


# --- noun parser ---------------------------------------------------------------

def _attach(parser: argparse.ArgumentParser, entries: list[tuple[list[str], Verb]], prog: str) -> None:
    """Build (possibly nested) subparsers from verb names split on spaces.

    A token can be both a verb and a prefix (`ink session` shows the session,
    `ink session start` opens one): the leaf becomes the parser's default and
    its sub-verbs are optional. Subparser defaults win over the parent's, so a
    named sub-verb still resolves to itself."""
    leaves = [v for toks, v in entries if not toks]
    deeper = [(toks, v) for toks, v in entries if toks]
    if leaves:
        v = leaves[0]
        if v.args:
            v.args(parser)
        parser.set_defaults(_verb=v)
        if not deeper:
            return
    sub = parser.add_subparsers(dest="_sub", metavar="<verb>")
    sub.required = not leaves
    groups: dict[str, list[tuple[list[str], Verb]]] = {}
    for toks, v in deeper:
        groups.setdefault(toks[0], []).append((toks[1:], v))
    for tok, group in groups.items():
        first = group[0][1]
        help_ = f"[{first.tier}] {first.summary}" if len(group) == 1 else f"{tok} …"
        p = sub.add_parser(tok, help=help_, description=help_,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        _attach(p, group, f"{prog} {tok}")


def resolve_verb(noun: str, own: list[str]) -> Verb | None:
    """The verb whose (possibly multi-token) name is the longest prefix of `own`."""
    best = None
    for v in verbs_of(noun):
        toks = v.verb.split()
        if own[:len(toks)] == toks and (best is None or len(toks) > len(best.verb.split())):
            best = v
    return best


def build_noun_parser(noun: str) -> argparse.ArgumentParser:
    vs = verbs_of(noun)
    # A pure passthrough noun (logs, ink, check) forwards --help to the tool it wraps.
    pure = len(vs) == 1 and vs[0].verb == "" and vs[0].args is None and vs[0].passthrough
    parser = argparse.ArgumentParser(
        prog=f"tatbot {noun}", description=NOUN_SUMMARY.get(noun, ""), add_help=not pure,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="args after `--` pass through untouched to the underlying tool")
    entries = [(v.verb.split() if v.verb else [], v) for v in vs]
    _attach(parser, entries, f"tatbot {noun}")
    return parser


# --- output helpers --------------------------------------------------------------

def refuse(ctx: Ctx, code: int, gate: str, reason: str, fix: str | None = None) -> int:
    if ctx.json:
        print(json.dumps({"code": code, "status": EXIT_NAMES[code], "gate": gate,
                          "reason": reason, "fix": fix}), file=sys.stderr)
    else:
        print(f"tatbot: {EXIT_NAMES[code]} ({gate}): {reason}", file=sys.stderr)
        if fix:
            print(f"  {fix}", file=sys.stderr)
    return code


def normalize(res: Plan | list[str]) -> Plan:
    return res if isinstance(res, Plan) else Plan(argv=list(res))


def print_plan(ctx: Ctx, v: Verb, plan: Plan, *, hop_to: str | None = None) -> None:
    role_note = ""
    if v.role:
        ok = v.role in nodes.roles_of(nodes.load(ctx.repo), ctx.node)
        role_note = f" (needs role {v.role}: {'ok' if ok else 'MISSING'})"
    if ctx.json:
        print(json.dumps({
            "dry_run": True, "verb": v.name, "tier": v.tier, "node": ctx.node, "hop": hop_to,
            "role": v.role, "argv": plan.argv, "cwd": str(plan.cwd) if plan.cwd else None,
            "env": plan.env, "gates": list(v.gates), "wraps": list(v.wraps), "notes": plan.notes,
            "busy": gates.busy_reasons(),
        }))
        return
    target = f"→ {', '.join(v.wraps)}" if v.wraps else "(native)"
    print(f"tatbot {v.name} [{v.tier}] {target}   dry run: nothing executed")
    print(f"  node   {ctx.node}{role_note}" + (f" → hop to {hop_to}" if hop_to else ""))
    if plan.cwd:
        print(f"  cwd    {plan.cwd}")
    for k, val in plan.env.items():
        print(f"  env    {k}={val}")
    for g in v.gates:
        print(f"  gate   {g}")
    for b in gates.busy_reasons():
        print(f"  busy   {b}")
    print(f"  exec   {shlex.join(plan.argv)}")
    for n in plan.notes:
        print(f"  note   {n}")


def print_explain(ctx: Ctx, v: Verb) -> None:
    nmap = nodes.load(ctx.repo)
    where = nodes.nodes_with(nmap, v.role) if v.role else ["any node"]
    if ctx.json:
        print(json.dumps({
            "verb": v.name, "tier": v.tier, "tier_meaning": TIER_MEANING[v.tier],
            "summary": v.summary, "role": v.role, "nodes": where, "gates": list(v.gates),
            "needs_tool": v.needs_tool, "nonce": v.nonce, "nonce_exempt": v.nonce_exempt,
            "dip_hook": v.dip_hook, "passthrough": v.passthrough,
            "wraps": list(v.wraps), "doc": v.doc, "invariants": list(v.invariants),
            "example": f"tatbot {v.name} {' '.join(v.example)}".strip(),
        }, indent=2))
        return
    print(f"tatbot {v.name} — {v.summary}")
    print(f"  tier       {v.tier}: {TIER_MEANING[v.tier]}")
    print(f"  runs on    {', '.join(where)}" + (f"  (role {v.role})" if v.role else ""))
    for g in v.gates:
        print(f"  gate       {g}")
    if v.needs_tool:
        print("  requires   --ee-tool <id> (stated, never inferred)")
    if v.nonce:
        print("  requires   --nonce <literal> (single-use, ledgered; armed on the arm node, also over --on)"
              + (f" — not with {v.nonce_exempt_flags}, which command nothing" if v.nonce_exempt else ""))
    if v.dip_hook:
        print("  with --dip a scripted dip runs first: --nonce <literal> required (armed on the arm node, also over --on)")
    if v.passthrough:
        print(f"  after --   passes through to {v.passthrough}")
    if v.wraps:
        print(f"  wraps      {', '.join(v.wraps)}")
    if v.doc:
        print(f"  docs       {v.doc}")
    for inv in v.invariants:
        print(f"  invariant  {inv}")
    if v.example:
        print(f"  example    tatbot {v.name} {' '.join(v.example)}")


# --- dispatch --------------------------------------------------------------------

def dispatch(ctx: Ctx, v: Verb, ns: argparse.Namespace, passthrough: list[str]) -> int:
    if ctx.explain:
        print_explain(ctx, v)
        return EXIT_OK

    # Gates the CLI can decide itself. The launcher re-checks every one of them.
    if v.tier in MOTION_TIERS:
        bad = gates.estop_overrides(passthrough)
        if bad:
            return refuse(ctx, EXIT_GATE_REFUSED, "estop_guard",
                          f"refusing E-stop override in a production launcher: {' '.join(bad)}",
                          "use the lower-level component command for an intentional hardware-free bench run")
        # Hardware profile gate (plan Phase 2): motion needs a complete
        # hardware profile, resolved and validated BEFORE anything connects.
        # A dry run only plans, so it reports the problem instead of refusing
        # (the public tree has no default profile and must still --dry-run).
        import tatbot_profile
        try:
            profile = tatbot_profile.load(ctx.repo)
            perrs = tatbot_profile.hardware_errors(profile)
            pwhy = (f"profile '{profile['name']}' cannot drive hardware: "
                    + "; ".join(perrs)) if perrs else None
        except tatbot_profile.ProfileError as e:
            profile, pwhy = None, str(e)
        if pwhy:
            if not ctx.dry_run:
                return refuse(ctx, EXIT_GATE_REFUSED, "profile", pwhy)
            print(f"tatbot: dry-run note (profile gate would refuse): {pwhy}",
                  file=sys.stderr)
        else:
            os.environ.update(tatbot_profile.env_exports(profile))
    if v.needs_tool:
        tool, err = gates.resolve_tool(ctx.repo, ctx.ee_tool)
        if err:
            return refuse(ctx, EXIT_GATE_REFUSED, "ee_tool", err, "pass --ee-tool <id> before the noun")
        ctx.ee_tool = tool
    nonce = getattr(ns, "nonce", None)
    needs_nonce = v.needs_nonce(ns)
    if needs_nonce:
        err = gates.nonce_error(nonce)
        if err:
            if v.dip_hook and v.tier != MOTION_AUTO:
                err = f"--dip is a scripted, autonomous dip before the session, so {err}"
            return refuse(ctx, EXIT_GATE_REFUSED, "arm_gate", err)

    nmap = nodes.load(ctx.repo)

    # Node routing.
    if ctx.on:
        # Autonomous motion hops too (operator decision 2026-09-02: the draw
        # sessions are launched from the viewer node). The hop is `ssh -t`, the
        # nonce is written on the arm node right before exec and consumed once
        # by its arm_gate, so the single-use ledger is unchanged; what moves is
        # the keyboard, not the e-stop operator, who stays at the rig.
        if v.autonomous(ns) and not ctx.quiet and not ctx.dry_run:
            print(f"tatbot: autonomous motion over --on {ctx.on}: the nonce is armed there right before "
                  "exec; the e-stop operator stays at the rig", file=sys.stderr)
        if ctx.on not in nmap:
            return _usage_error(f"unknown node '{ctx.on}' (known: {', '.join(nmap)})")
        if ctx.on == ctx.node:
            return _usage_error(f"--on {ctx.on}: that is this node")
        if "checkout" in nmap[ctx.on] and nmap[ctx.on]["checkout"] is None:
            return refuse(ctx, EXIT_USAGE, "node", f"{ctx.on} has no git checkout to run in",
                          nmap[ctx.on].get("note"))
        remote_argv = [a for a in ctx.argv if a not in ("--on", ctx.on) and not a.startswith("--on=")]
        plan = Plan(argv=nodes.hop_argv(nmap, ctx.on, remote_argv, tty=v.tty or v.tier in MOTION_TIERS, sync=v.sync),
                    notes=[f"remote exit code is returned as-is; run ids carry the node ({ctx.on})"]
                          + ([f"the {ctx.on} checkout is fast-forwarded to origin/main first"] if v.sync else []))
        if ctx.dry_run:
            print_plan(ctx, v, plan, hop_to=ctx.on)
            return EXIT_OK
        os.execvp(plan.argv[0], plan.argv)
    if v.role and v.role not in nodes.roles_of(nmap, ctx.node):
        cands = nodes.nodes_with(nmap, v.role)
        hint = f"tatbot --on {cands[0]} {shlex.join(ctx.argv)}" if cands else "no node in config/nodes.json has that role"
        if v.auto_hop and cands and not ctx.no_hop and v.tier not in MOTION_TIERS and not v.autonomous(ns):
            # The verb asked to be routed: run it where the role lives, as if --on had been given.
            plan = Plan(argv=nodes.hop_argv(nmap, cands[0], list(ctx.argv), tty=v.tty, sync=v.sync),
                        notes=[f"{v.name} runs on {cands[0]} (role {v.role}); this node ({ctx.node}) has no {v.role}"]
                              + ([f"the {cands[0]} checkout is fast-forwarded to origin/main first"] if v.sync else []))
            if ctx.dry_run:
                print_plan(ctx, v, plan, hop_to=cands[0])
                return EXIT_OK
            if not ctx.quiet:
                print(f"tatbot: {v.name} → {cands[0]}" + (" (syncing its checkout first)" if v.sync else ""), file=sys.stderr)
            os.execvp(plan.argv[0], plan.argv)
        if ctx.no_hop:
            print(f"tatbot: warning: {ctx.node} lacks role {v.role} (continuing: --no-hop)", file=sys.stderr)
        else:
            return refuse(ctx, EXIT_WRONG_NODE, "node",
                          f"{v.name} needs role \"{v.role}\" — this node ({ctx.node}) has "
                          f"[{', '.join(nodes.roles_of(nmap, ctx.node)) or 'no roles'}]", hint)

    res = v.run(ctx, ns, passthrough)
    if isinstance(res, int):
        return res
    plan = normalize(res)
    plan.env.setdefault("TATBOT_VIA_CLI", "1")
    # Propagate the canonical config/nodes.json identity to wrapped tools.
    # A host whose hostname differs from its node name cannot safely
    # reconstruct it from hostname alone, and training profiles are keyed by
    # the canonical node name.
    plan.env.setdefault("TATBOT_NODE", ctx.node)
    if ctx.ee_tool:
        plan.env.setdefault("TATBOT_EE_TOOL", ctx.ee_tool)
    if ctx.dry_run:
        print_plan(ctx, v, plan)
        return EXIT_OK
    if needs_nonce and nonce:
        gates.write_nonce(nonce)
    env = dict(os.environ)
    env.update(plan.env)
    if plan.cwd:
        os.chdir(plan.cwd)
    try:
        os.execvpe(plan.argv[0], plan.argv, env)
    except FileNotFoundError:
        return refuse(ctx, EXIT_USAGE, "exec", f"not found: {plan.argv[0]}",
                      "build it first (scripts/check builds cpp/rust) or check the path")


def main(argv: list[str]) -> int:
    try:
        g, noun, rest = parse_globals(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if g["version"]:
        print(f"tatbot {__version__}")
        return EXIT_OK
    if noun is None:
        _root_usage(sys.stdout if g["help"] else sys.stderr)
        return EXIT_OK if g["help"] else EXIT_USAGE
    if noun not in nouns():
        close = [n for n in NOUN_ORDER if n.startswith(noun[:2])]
        return _usage_error(f"unknown noun '{noun}'" + (f" (did you mean: {', '.join(close)}?)" if close else "")
                            + " — `tatbot --help` lists them")
    ctx = Ctx(repo=repo_root(), node=nodes.this_node(nodes.load(repo_root())), json=g["json"], dry_run=g["dry_run"], on=g["on"],
              ee_tool=g["ee_tool"], explain=g["explain"], quiet=g["quiet"], verbose=g["verbose"],
              no_hop=g["no_hop"], argv=list(argv))
    own, passthrough = split_dashdash(rest)
    if g["explain"]:
        # --explain needs only the verb, not its positionals.
        v = resolve_verb(noun, own)
        if v is None:
            return _usage_error(f"tatbot {noun}: which verb? (`tatbot {noun} --help`)")
        print_explain(ctx, v)
        return EXIT_OK
    if g["help"]:
        own = own + ["--help"]
    parser = build_noun_parser(noun)
    try:
        ns, unknown = parser.parse_known_args(own)
    except SystemExit as e:
        return EXIT_OK if e.code == 0 else EXIT_USAGE
    v: Verb = ns._verb
    return dispatch(ctx, v, ns, unknown + passthrough)
