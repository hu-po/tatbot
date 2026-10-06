"""Root parser, dispatch, gates, node routing, --dry-run/--explain/--json."""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import io
import json
import os
import shlex
import sys
from pathlib import Path

from tatbot_cli import (
    EXIT_BUSY,
    EXIT_GATE_REFUSED,
    EXIT_HW_UNREACHABLE,
    EXIT_NAMES,
    EXIT_OK,
    EXIT_USAGE,
    EXIT_WRONG_NODE,
    __version__,
    gates,
    nodes,
    rig,
)
from tatbot_cli.aliases import AliasError, for_command, translate
from tatbot_cli.audience import visible as is_visible
from tatbot_cli.metadata import operation_effects, output_capability, requirements
from tatbot_cli.registry import (
    GLOBAL_ORDER,
    MOTION_TIERS,
    NOUN_SUMMARY,
    OFFLINE,
    PREFER_LOCAL,
    TIER_MEANING,
    Ctx,
    Invocation,
    Plan,
    Verb,
    all_verbs,
    nouns,
    repo_root,
    verbs_of,
)

GLOBAL_HELP = """\
global flags (anywhere before `--`; `--flag value` or `--flag=value`):
  --json            machine-readable output; refusals are one JSON line on stderr
  --dry-run         resolve node, tool and gates, print the exact command, exec nothing
  --on <node>       run this command on another node over ssh (config/nodes.json);
                    a command that owns a fleet role already routes itself there
  --ee-tool <id>    the tool in the mount (or TATBOT_EE_TOOL); omitted, the
                    execution owner's configured fitted tool is used
  --explain         print the verb's tier, gates, invariants and docs, then exit
  --help-all        help including advanced, compatibility, internal and
                    experimental commands (nothing is hidden from the schema)
  -q / -v           quieter / louder
  --version

exit codes:
  0 ok   1 tool failed   2 usage   3 safety gate refused   4 wrong node
  5 hardware unreachable   6 busy (arm held, training lock, SWEEP_PAUSE)
"""


def _root_usage(out=sys.stdout, show_all: bool = False) -> None:
    print("usage: tatbot [global flags] <noun> [verb] [args] [-- passthrough]\n", file=out)
    hidden = 0
    for group, title in (("operator", "Operator work"), ("development", "Offline development"),
                         ("administration", "Administration")):
        print(title + ":", file=out)
        for n in nouns():
            if not any(v.group == group for v in verbs_of(n)):
                continue
            tiers = sorted({v.tier for v in verbs_of(n)}, key=lambda t: list(TIER_MEANING).index(t))
            print(f"  {n:<11} {NOUN_SUMMARY.get(n, ''):<52} [{', '.join(tiers)}]", file=out)
            if not show_all:
                hidden += sum(1 for v in verbs_of(n) if not is_visible(v))
                continue
            # --help-all: every command, grouped under its noun, so the whole
            # tree is readable in one place without reading the JSON schema.
            for v in verbs_of(n):
                mark = "" if v.audience == "primary" else f"  ({v.audience})"
                print(f"      {v.verb or '(bare)':<24} {v.summary[:60]}{mark}", file=out)
        print(file=out)
    print("Examples: tatbot status; tatbot teleop start; tatbot ros status; tatbot logs last rollout", file=out)
    print(file=out)
    print(GLOBAL_HELP, file=out)
    if hidden:
        print(f"Ordinary help shows everyday commands. {hidden} advanced, compatibility, internal and\n"
              "experimental commands are one flag away: `tatbot --help-all`, or `tatbot <noun> --help-all`.\n"
              "Every one of them is in `tatbot schema --json`.\n", file=out)
    print("`tatbot <noun> --help` lists its verbs; `tatbot <noun> <verb> --explain` says what it can do.", file=out)


BARE_GLOBALS = {"--json": "json", "--dry-run": "dry_run", "--explain": "explain", "--no-hop": "no_hop"}
VALUED_GLOBALS = {"--on": "on", "--ee-tool": "ee_tool"}
# -q/-v are extracted after the command too, except out of a remainder
# passthrough: there every unrecognized single-dash flag belongs to the backend,
# and short flags collide far more often than the long global spellings do.
SWITCH_GLOBALS = {"-q": "quiet", "-v": "verbose"}


def _set_valued(g: dict, key: str, value: str) -> None:
    """Record a valued global once. Repeating it is fine; disagreeing is not."""
    if not value:
        raise UsageError(f"{key} needs a value")
    dest = VALUED_GLOBALS[key]
    previous = g[dest]
    if previous is not None and previous != value:
        raise UsageError(f"conflicting {key} values: '{previous}' and '{value}' "
                         f"(state {key} once)")
    g[dest] = value


def _global_tokens(g: dict) -> tuple[list[str], list[str]]:
    """(non-routing globals, routing tokens) in one canonical spelling."""
    tokens = [flag for flag in GLOBAL_ORDER
              if g.get(BARE_GLOBALS.get(flag) or SWITCH_GLOBALS.get(flag, ""), False)]
    if g["ee_tool"]:
        tokens += ["--ee-tool", g["ee_tool"]]
    return tokens, (["--on", g["on"]] if g["on"] else [])


_JSON_OUTPUT = contextvars.ContextVar("cli_json_output", default=False)


class UsageError(ValueError):
    """A CLI-owned argument error, rendered once by main."""


class Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message):
        raise UsageError(f"{self.prog}: {message}")


def parse_globals(argv: list[str]) -> tuple[dict, str | None, list[str]]:
    """Split an argv into globals, the command, and the backend tail.

    Globals may appear anywhere before the first ``--``: before the noun,
    between noun and verb, or after the verb, spelled ``--flag value`` or
    ``--flag=value``. Placement is resolved against the real parser tree, never
    by searching the raw string, so a filename that happens to equal a node
    name, a tool id, or an option spelling stays an operand of the command that
    declared it. Everything after the first ``--`` is the backend's, in order.
    """
    g = {"json": False, "dry_run": False, "on": None, "ee_tool": None, "explain": False,
         "quiet": False, "verbose": False, "no_hop": False, "help": False, "help_all": False,
         "version": False}
    i = 0
    noun = None
    while i < len(argv):
        token = argv[i]
        key, equal, value = token.partition("=")
        if token in BARE_GLOBALS:
            g[BARE_GLOBALS[token]] = True
            _JSON_OUTPUT.set(g["json"])
        elif token in ("-h", "--help", "--help-all", "--version", "-q", "-v"):
            g[{"-h": "help", "--help": "help", "--version": "version",
               "-q": "quiet", "-v": "verbose", "--help-all": "help"}[token]] = True
            g["help_all"] = g["help_all"] or token == "--help-all"
        elif key in VALUED_GLOBALS:
            if not equal:
                if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                    raise UsageError(f"{key} needs a value")
                i += 1
                value = argv[i]
            _set_valued(g, key, value)
        elif token.startswith("-"):
            raise UsageError(f"unknown global flag {token} (`tatbot --help` lists them)")
        else:
            noun = token
            break
        i += 1
    rest = list(argv[i + 1:]) if noun else []
    try:
        rest, deprecations = translate(noun, rest, BARE_GLOBALS, explain=g["explain"])
    except AliasError as exc:
        # These legacy sim selectors own no JSON option. A trailing --json
        # still selects structured usage output, even when translation refuses.
        head = rest[:rest.index("--")] if "--" in rest else rest
        if "--json" in head:
            _JSON_OUTPUT.set(True)
        raise UsageError(str(exc)) from exc
    head, tail = split_dashdash(rest)
    kept: list[str] = []
    ownership_error = None
    # Walk the parser tree without executing handlers. A command-owned option
    # and its values take precedence over any global spelled the same way.
    parser = build_noun_parser(noun, show_all=True) if noun in nouns() else None
    remainder = _is_remainder_passthrough(parser)
    j = 0
    while j < len(head):
        token = head[j]
        key, equal, value = token.partition("=")
        action = parser._option_string_actions.get(key) if parser else None
        if action is not None:
            kept.append(token)
            j += 1
            count = action.nargs
            if count != 0 and not equal:
                if count is None:
                    count = 1
                if isinstance(count, int):
                    # argparse will report a missing/invalid value; globals
                    # never get to steal a declared option's operand.
                    consumed = 0
                    for _ in range(count):
                        if j >= len(head) or head[j].startswith("--"):
                            break
                        kept.append(head[j])
                        j += 1
                        consumed += 1
                    if consumed < count:
                        ownership_error = f"{key} needs {count} value(s) before the next option"
                elif count in ("?", "*", "+"):
                    start = j
                    while j < len(head) and not head[j].startswith("-"):
                        kept.append(head[j])
                        j += 1
                        if count == "?":
                            break
                    if count == "+" and start == j:
                        ownership_error = f"{key} needs a value before the next option"
            continue
        if key in VALUED_GLOBALS:
            if not equal:
                if j + 1 >= len(head) or head[j + 1].startswith("-"):
                    raise UsageError(f"{key} needs a value")
                j += 1
                value = head[j]
            _set_valued(g, key, value)
        elif token in BARE_GLOBALS:
            g[BARE_GLOBALS[token]] = True
            _JSON_OUTPUT.set(g["json"])
        elif token == "--help-all":
            g["help"] = g["help_all"] = True
        elif token in SWITCH_GLOBALS and not remainder:
            g[SWITCH_GLOBALS[token]] = True
        else:
            kept.append(token)
            if parser:
                sub = next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)
                if sub and token in sub.choices:
                    parser = sub.choices[token]
                    remainder = _is_remainder_passthrough(parser)
        j += 1
    if ownership_error and not g["explain"] and not g["help"] and not {"-h", "--help"}.intersection(head):
        raise UsageError(ownership_error)
    globals_, routing = _global_tokens(g)
    g["invocation"] = Invocation(list(argv), globals_, ([noun] if noun else []) + kept, tail,
                                 routing_tokens=routing, has_passthrough="--" in rest,
                                 deprecations=deprecations)
    return g, noun, kept + (["--", *tail] if "--" in rest else [])


def _is_remainder_passthrough(parser) -> bool:
    """True where the command declares no options of its own and forwards the
    remainder verbatim (`tatbot logs …`, `tatbot vision d …`)."""
    v = parser.get_default("_verb") if parser is not None else None
    return bool(v is not None and v.args is None and v.passthrough)


def _usage_error(msg: str) -> int:
    if _JSON_OUTPUT.get():
        print(json.dumps({"schema": "tatbot.cli-error/1", "code": EXIT_USAGE,
                          "status": EXIT_NAMES[EXIT_USAGE], "reason": msg}), file=sys.stderr)
    else:
        print(f"tatbot: {msg}", file=sys.stderr)
    return EXIT_USAGE


def split_dashdash(args: list[str]) -> tuple[list[str], list[str]]:
    if "--" in args:
        i = args.index("--")
        return args[:i], args[i + 1:]
    return args, []


# --- noun parser ---------------------------------------------------------------

def _attach(parser: argparse.ArgumentParser, entries: list[tuple[list[str], Verb]], prog: str,
            show_all: bool = False) -> None:
    """Build (possibly nested) subparsers from verb names split on spaces.

    A token can be both a verb and a prefix (`ink session` shows the session,
    `ink session start` opens one): the leaf becomes the parser's default and
    its sub-verbs are optional. Subparser defaults win over the parent's, so a
    named sub-verb still resolves to itself.

    A command outside this help level is SUPPRESSed, never omitted: it still
    parses, still runs, and still appears in the schema and the full reference.
    """
    leaves = [v for toks, v in entries if not toks]
    deeper = [(toks, v) for toks, v in entries if toks]
    if leaves:
        v = leaves[0]
        v.argument_spec.install(parser)
        parser.set_defaults(_verb=v)
        aliases = for_command(v.name)
        if aliases:
            parser.epilog = (parser.epilog or "") + "\nDeprecated compatibility forms:\n" + "\n".join(
                "  tatbot " + a.source + " -> tatbot " + a.canonical_form for a in aliases
            )
        if not deeper:
            return
    sub = parser.add_subparsers(dest="_sub", metavar="<verb>")
    # Never argparse-required: a bare noun should answer with what the noun is
    # FOR, not with "the following arguments are required: <verb>". `_verb`
    # stays unset and the dispatcher renders the everyday commands instead.
    sub.required = False
    groups: dict[str, list[tuple[list[str], Verb]]] = {}
    for toks, v in deeper:
        groups.setdefault(toks[0], []).append((toks[1:], v))
    for tok, group in groups.items():
        members = [v for _, v in group]
        shown = [v for v in members if is_visible(v, show_all=show_all)]
        summary = (f"[{members[0].tier}] {members[0].summary}" if len(group) == 1 else f"{tok} …")
        # A group nobody at this level should read gets NO help entry at all,
        # which is how argparse omits a choice while still parsing it (a
        # `help=SUPPRESS` renders the literal "==SUPPRESS==" instead).
        options = {} if not shown else {"help": summary}
        p = sub.add_parser(tok, description=summary,
                           formatter_class=argparse.RawDescriptionHelpFormatter, **options)
        _attach(p, group, f"{prog} {tok}", show_all)


def suggest_noun(token: str) -> str | None:
    """The closest command, named but never run.

    A mistyped noun is often a verb of another one (`sweep`, `adopt`), so the
    whole tree is searched, not just the first level.
    """
    import difflib
    # An exact verb token beats a fuzzy noun: someone typing `sweep` means
    # `work sweep`, not `serve`.
    owners = sorted({v.name for v in all_verbs()
                     if token in v.verb.split() and is_visible(v, show_all=True)})
    if owners:
        return " (did you mean: " + ", ".join(f"tatbot {name}" for name in owners[:2]) + "?)"
    close = difflib.get_close_matches(token, list(nouns()), n=2, cutoff=0.7)
    return f" (did you mean: {', '.join(close)}?)" if close else None


def missing_verb_help(noun: str) -> str:
    """What this noun is actually for, rather than a pointer to another page."""
    everyday = [v for v in verbs_of(noun) if is_visible(v)]
    if not everyday:
        return f" (`tatbot {noun} --help-all` lists its commands)"
    lines = [f"\n  tatbot {v.name}" + (f" {shlex.join(v.example)}" if v.example else "")
             + f"\n      {v.summary}" for v in everyday[:5]]
    more = f"\n  … `tatbot {noun} --help` for the rest" if len(everyday) > 5 else ""
    return "".join(lines) + more


def resolve_verb(noun: str, own: list[str]) -> Verb | None:
    """The verb whose (possibly multi-token) name is the longest prefix of `own`."""
    best = None
    for v in verbs_of(noun):
        toks = v.verb.split()
        if own[:len(toks)] == toks and (best is None or len(toks) > len(best.verb.split())):
            best = v
    return best


def build_noun_parser(noun: str, *, show_all: bool = False) -> argparse.ArgumentParser:
    """The parser for a noun and its verbs."""
    vs = verbs_of(noun)
    # A pure passthrough noun (logs, ink, check) forwards --help to the tool it wraps.
    pure = len(vs) == 1 and vs[0].verb == "" and vs[0].args is None and vs[0].passthrough
    parser = Parser(
        prog=f"tatbot {noun}", description=NOUN_SUMMARY.get(noun, ""), add_help=not pure,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="args after `--` pass through untouched to the underlying tool")
    entries = [(v.verb.split() if v.verb else [], v) for v in vs]
    _attach(parser, entries, f"tatbot {noun}", show_all)
    hidden = sum(1 for v in vs if not is_visible(v, show_all=show_all))
    if hidden:
        parser.epilog = (f"{hidden} more command(s) here: `tatbot {noun} --help-all` "
                         "(all of them are in `tatbot schema --json`)\n\n" + (parser.epilog or ""))
    return parser


def example_role(v: Verb) -> str | None:
    """The role the verb's registered example needs — the declared one, or the
    one the example's own operands resolve — for the checks that dry-run every
    example on a node that owns it."""
    if v.role_for is None:
        return v.role
    own = [*v.verb.split(), *v.example]
    ns, _ = build_noun_parser(v.noun).parse_known_args(own)
    return v.required_role(ns)


# --- output helpers --------------------------------------------------------------

def refuse(ctx: Ctx, code: int, gate: str, reason: str, fix: str | None = None) -> int:
    if ctx.json:
        print(json.dumps({"schema": "tatbot.cli-error/1", "code": code, "status": EXIT_NAMES[code], "gate": gate,
                          "reason": reason, "fix": fix}), file=sys.stderr)
    else:
        print(f"tatbot: {EXIT_NAMES[code]} ({gate}): {reason}", file=sys.stderr)
        if fix:
            print(f"  {fix}", file=sys.stderr)
    return code


def normalize(res: Plan | list[str]) -> Plan:
    return res if isinstance(res, Plan) else Plan(argv=list(res))


def _home_relative(arg: str) -> str:
    """A path under this node's HOME travels as `~/...` so it resolves under the remote HOME
    (every node keeps its checkout and logs at the same HOME-relative paths; the shell expanded ~
    before we saw it)."""
    home = os.path.expanduser("~")
    if home != "/" and (arg == home or arg.startswith(home + "/")):
        return "~" + arg[len(home):]
    return arg


def print_plan(ctx: Ctx, v: Verb, plan: Plan, *, hop_to: str | None = None,
               launch_id: str | None = None, ns=None, passthrough=()) -> None:
    _warn_deprecations(ctx)
    role = v.required_role(ns)
    role_note = ""
    if role:
        ok = role in nodes.roles_of(nodes.load(ctx.repo), ctx.node)
        role_note = f" (needs role {role}: {'ok' if ok else 'MISSING'})"
    if ctx.json:
        print(json.dumps({
            "schema": "tatbot.cli-plan/2", "kind": plan.kind,
            "effects": operation_effects(v, ns, passthrough, plan=plan, remote=bool(hop_to)),
            "requirements": requirements(v, ns, passthrough, plan=plan, remote=bool(hop_to)),
            "passthrough": list(passthrough),
            "deprecations": ctx.invocation.deprecations if ctx.invocation else [],
            "invocation": ctx.invocation.command_tokens if ctx.invocation else ctx.argv,
            "dry_run": True, "verb": v.name, "tier": v.tier, "node": ctx.node, "hop": hop_to,
            "role": role, "argv": plan.argv, "cwd": str(plan.cwd) if plan.cwd else None,
            "env": plan.env, "gates": list(v.gates), "wraps": list(v.wraps), "notes": plan.notes,
            "files": plan.files,
            "busy": gates.busy_reasons(), "launch_id": launch_id, "tool": tool_selection(ctx, v, hop_to),
        }))
        return
    target = f"→ {', '.join(v.wraps)}" if v.wraps else "(native)"
    print(f"tatbot {v.name} [{v.tier}] {target}   dry run: nothing executed")
    print(f"  node   {ctx.node}{role_note}" + (f" → hop to {hop_to}" if hop_to else ""))
    selection = tool_selection(ctx, v, hop_to)
    if selection:
        print(f"  tool   {selection['id'] or 'resolved on ' + str(selection['resolved_on'])}"
              + (f" ({selection['source_label']})" if selection["id"] else "")
              + ("  [deferred: nothing was asked of that node]" if selection["deferred"] else ""))
    if plan.cwd:
        print(f"  cwd    {plan.cwd}")
    for k, val in plan.env.items():
        print(f"  env    {k}={val}")
    for g in v.gates:
        print(f"  gate   {g}")
    for b in gates.busy_reasons():
        print(f"  busy   {b}")
    if launch_id:
        print(f"  launch {launch_id}  (example; a fresh id is minted at exec, ledgered + audited by arm_gate)")
    _print_files(plan)
    print(f"  exec   {shlex.join(plan.argv)}" if plan.kind == "exec" else f"  native {v.name}")
    for n in plan.notes:
        print(f"  note   {n}")


def _print_files(plan: Plan) -> None:
    for path in plan.files:
        print(f"  file   {path}  (written right before the exec)")


def write_files(plan: Plan) -> None:
    """What the verb composed for its launcher, written after every gate and
    immediately before the exec (a dry run only shows it)."""
    for path, text in plan.files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)


def tool_selection(ctx: Ctx, v: Verb, hop_to: str | None = None) -> dict | None:
    """What tool this invocation uses, and on whose authority.

    Across a hop the answer belongs to the execution owner, so planning says so
    instead of sending this node's default as if the operator had chosen it, and
    instead of claiming to have checked anything over there.
    """
    # A verb whose tool requirement is per-invocation (a calibration whose arm
    # phases need one, a camera-only capture that does not) resolves it in its
    # own prepare hook. Report whatever was actually selected, however it was.
    if not v.needs_tool and not ctx.ee_tool:
        return None
    deferred = bool(hop_to) and not ctx.ee_tool
    return {"id": None if deferred else ctx.ee_tool,
            "source": None if deferred else ctx.tool_source,
            "source_label": gates.TOOL_SOURCES.get(ctx.tool_source or "", ctx.tool_source),
            "deferred": deferred, "resolved_on": hop_to if deferred else ctx.node}


def print_explain(ctx: Ctx, v: Verb) -> None:
    _warn_deprecations(ctx)
    nmap = nodes.load(ctx.repo)
    where = (nodes.nodes_with(nmap, v.role) if v.role
             else [v.role_for.describe] if v.role_for else ["any node"])
    if ctx.json:
        print(json.dumps({
            "schema": "tatbot.cli-explain/2", "effects": operation_effects(v), "requirements": requirements(v),
            "output": v.output, "deprecations": ctx.invocation.deprecations if ctx.invocation else [],
            "audience": v.audience, "disposition": v.disposition,
            "verb": v.name, "tier": v.tier, "tier_meaning": TIER_MEANING[v.tier],
            "summary": v.summary, "role": v.role, "role_from": v.role_for.describe if v.role_for else None,
            "nodes": where, "gates": list(v.gates),
            "needs_tool": v.needs_tool, "launch_id": v.launch_id,
            "ink_hook": v.ink_hook, "passthrough": v.passthrough,
            "wraps": list(v.wraps), "doc": v.doc, "invariants": list(v.invariants),
            "example": f"tatbot {v.name} {shlex.join(v.example)}".strip(),
        }, indent=2))
        return
    print(f"tatbot {v.name} — {v.summary}")
    print(f"  tier       {v.tier}: {TIER_MEANING[v.tier]}")
    print(f"  audience   {v.audience}" + (f": {v.disposition}" if v.disposition else ""))
    print(f"  runs on    {', '.join(where)}" + (f"  (role {v.role})" if v.role else
                                               "  (a role resolved from the operands)" if v.role_for else ""))
    for g in v.gates:
        print(f"  gate       {g}")
    if v.launch_id:
        print("  launch id  automatic: minted on the arm node right before exec (also over --on), written to "
              "/tmp/tatbot-arm-token, ledgered + audited by arm_gate (pid chain, SSH origin); --tag <label> adds a label")
    if v.ink_hook:
        print("  ink        --no-ink: no ink session, no stroke debit; the run is stamped tracking=false")
    if v.passthrough:
        print(f"  after --   passes through to {v.passthrough}")
    if v.wraps:
        print(f"  wraps      {', '.join(v.wraps)}")
    if v.doc:
        print(f"  docs       {v.doc}")
    for inv in v.invariants:
        print(f"  invariant  {inv}")
    if v.example:
        print(f"  example    tatbot {v.name} {shlex.join(v.example)}")


# --- dispatch --------------------------------------------------------------------


def resolve_owner(nmap: dict, v: Verb, role: str, node: str, on: str | None):
    """(owner, reason, fix) — which node runs a command that declares a role.

    Ownership is a contract, not a preference: an `owner` command refuses when
    the role has no owner or several, even though some node carries the role,
    because "whichever node answered" is the wrong node to move an arm from.
    An explicit --on may name the owner; it may not appoint a different one.
    """
    owners = nodes.nodes_with(nmap, role)
    if v.routing == PREFER_LOCAL:
        if node in owners:
            return node, None, None
        if on in owners:
            return on, None, None
        if len(owners) == 1:
            return owners[0], None, None
        if not owners:
            return None, f"{v.name} needs role \"{role}\"; no node in config/nodes.json has it", None
        return (None, f"{v.name} needs role \"{role}\", which {len(owners)} nodes share",
                f"name one: tatbot --on <{'|'.join(owners)}> {v.name}")
    if len(owners) != 1:
        return (None, f"{v.name} belongs to the single {role} owner; "
                      f"config/nodes.json describes {len(owners)}",
                f"eligible: {', '.join(owners)}" if owners else
                f"give exactly one node the \"{role}\" role in config/nodes.json")
    owner = owners[0]
    if on and on != owner:
        return (None, f"{v.name} belongs to role {role} on {owner}, not {on}",
                f"drop --on, or use: tatbot --on {owner} {v.name}")
    return owner, None, None


def routing_notice(ctx: Ctx, v: Verb, hop_to: str | None, tool_source: str | None = None,
                   role: str | None = None) -> None:
    """One line saying where this ran and which tool it used, when either was
    resolved rather than typed. Silent under -q, --json and --dry-run (the plan
    already states both)."""
    if ctx.quiet or ctx.json or ctx.dry_run:
        return
    parts = []
    if hop_to:
        parts.append(f"{v.name} runs on the {role or v.role} owner ({hop_to})")
    if ctx.ee_tool and tool_source:
        parts.append(f"tool {ctx.ee_tool} ({tool_source})")
    if parts:
        print("tatbot: " + "; ".join(parts), file=sys.stderr)



def _admit(ctx: Ctx, v: Verb, ns: argparse.Namespace) -> tuple[str | None, int | None]:
    """What the operator asked for, checked against the command's own
    declarations before anything is routed: an impossible selection must not
    cost an ssh, and a question ("which phases?") must be answered here. Then
    the role this invocation needs — declared, or resolved from its operands
    (`vision capture --arm`: the arm's wrist camera owner in the vision
    registry). Returns (role, exit code to stop with)."""
    if v.validate:
        try:
            stop = v.validate(ctx, v, ns)
        except UsageError as error:
            return None, _usage_error(str(error))
        if stop is not None:
            return None, stop
    try:
        return v.required_role(ns), None
    except ValueError as error:
        return None, refuse(ctx, EXIT_WRONG_NODE, "node", str(error))


def dispatch(ctx: Ctx, v: Verb, ns: argparse.Namespace, passthrough: list[str]) -> int:
    if ctx.explain:
        print_explain(ctx, v)
        return EXIT_OK

    role, stop = _admit(ctx, v, ns)
    if stop is not None:
        return stop
    nmap = nodes.load(ctx.repo)

    # Route before binding files or probing hardware: these belong to the
    # execution node. The remote CLI repeats validation; a hop is no authority.
    if v.auto_hop and role:
        owner, reason, fix = resolve_owner(nmap, v, role, ctx.node, ctx.on)
        if owner is None:
            return refuse(ctx, EXIT_WRONG_NODE, "node", reason, fix)
        if owner != ctx.node:
            if ctx.no_hop:
                return refuse(ctx, EXIT_WRONG_NODE, "node",
                              f"{v.name} requires role {role}; refusing a second hop from {ctx.node}")
            ctx.on = owner
        else:
            # The owner is this node. An explicit --on naming it is honoured by
            # running here, not refused as an unnecessary self-hop.
            ctx.on = None

    # Node routing.
    if ctx.on and v.needs_tool and (ctx.ee_tool or os.environ.get("TATBOT_EE_TOOL")) \
            and gates.known_tools(ctx.repo):
        # A typo is refused here rather than after an ssh, stated by flag or
        # carried from TATBOT_EE_TOOL (which crosses the hop as --ee-tool); an
        # omitted tool is deliberately NOT resolved: the owner's configuration decides.
        _, _, err = gates.resolve_tool(ctx.repo, ctx.ee_tool)
        if err:
            return refuse(ctx, EXIT_GATE_REFUSED, "ee_tool", err, gates.FLAG_HINT)
    if ctx.on:
        if ctx.on not in nmap:
            return _usage_error(f"unknown node '{ctx.on}' (known: {', '.join(nmap)})")
        if role and role not in nodes.roles_of(nmap, ctx.on):
            return refuse(ctx, EXIT_WRONG_NODE, "node", f"{ctx.on} does not own required role {role}")
        if ctx.on == ctx.node:
            ctx.on = None
    if ctx.on:
        # Autonomous motion hops too (operator decision 2026-09-02: the draw
        # sessions are launched from the viewer node). The hop is `ssh -t`; the
        # launch id is minted on the arm node right before exec and ledgered +
        # audited by its arm_gate; what moves is the keyboard, not the e-stop
        # operator, who stays at the rig.
        if v.autonomous(ns) and v.launch_id and not ctx.quiet and not ctx.dry_run and not ctx.json:
            print(f"tatbot: autonomous motion over --on {ctx.on}: the launch id is minted there right before "
                  "exec; the e-stop operator stays at the rig", file=sys.stderr)
        if "checkout" in nmap[ctx.on] and nmap[ctx.on]["checkout"] is None:
            return refuse(ctx, EXIT_USAGE, "node", f"{ctx.on} has no git checkout to run in",
                          nmap[ctx.on].get("note"))
        if not nodes.ssh_target(nmap, ctx.on):
            return refuse(ctx, EXIT_WRONG_NODE, "node", f"{ctx.on} has no configured transport address")
        invocation = ctx.invocation or parse_globals(ctx.argv)[0]["invocation"]
        remote_argv = [_home_relative(a) for a in invocation.remote_tokens()]
        notes = [f"remote exit code is returned as-is; run ids carry the node ({ctx.on})"]
        carried = os.environ.get("TATBOT_EE_TOOL")
        if v.needs_tool and not ctx.ee_tool and carried:
            # Valued globals lead, which every released parser accepts.
            remote_argv[:0] = ["--ee-tool", carried]
            notes.append(f"TATBOT_EE_TOOL={carried} is carried over as an explicit --ee-tool")
        elif v.needs_tool and not ctx.ee_tool:
            notes.append(f"no tool stated: {ctx.on} resolves its own configured fitted tool")
        if v.sync:
            notes.append(f"the {ctx.on} checkout is fast-forwarded to origin/main first")
        plan = Plan(argv=nodes.hop_argv(nmap, ctx.on, remote_argv, tty=v.tty or v.tier in MOTION_TIERS,
                                        sync=v.sync),
                    notes=notes)
        if ctx.dry_run:
            print_plan(ctx, v, plan, hop_to=ctx.on, ns=ns, passthrough=passthrough)
            return EXIT_OK
        if ctx.json and output_capability(v, ns, passthrough) == "text":
            return refuse(ctx, EXIT_USAGE, "output", f"{v.name} has text-only execution; use --dry-run --json")
        _warn_deprecations(ctx)
        routing_notice(ctx, v, ctx.on, role=role)
        os.execvp(plan.argv[0], plan.argv)
    if role and role not in nodes.roles_of(nmap, ctx.node):
        cands = nodes.nodes_with(nmap, role)
        hint = f"tatbot --on {cands[0]} {shlex.join(ctx.argv)}" if cands else "no node in config/nodes.json has that role"
        if ctx.no_hop:
            if not ctx.json:
                print(f"tatbot: warning: {ctx.node} lacks role {role} (continuing: --no-hop)", file=sys.stderr)
        else:
            return refuse(ctx, EXIT_WRONG_NODE, "node",
                          f"{v.name} needs role \"{role}\" — this node ({ctx.node}) has "
                          f"[{', '.join(nodes.roles_of(nmap, ctx.node)) or 'no roles'}]", hint)

    if v.prepare:
        try:
            v.prepare(ctx, v, ns)
        except UsageError as error:
            # An impossible selection is the operator's typing, not a gate.
            return _usage_error(str(error))
        except (ValueError, KeyError, OSError) as error:
            return refuse(ctx, EXIT_GATE_REFUSED, v.noun, str(error))

    profile_env = {}
    # Gates the CLI can decide itself. The launcher re-checks every one of them.
    # The rig marker: `tatbot rig sleep` switched the cameras, screen and hosts
    # off, so a hardware verb would only report exit 5 piecemeal; say why once.
    effects = set(operation_effects(v, ns, passthrough))
    diagnostic = (v.verb == "status" and v.tier == OFFLINE and
                  effects <= {"read_files", "network", "remote_exec"})
    if v.noun != "rig" and (v.tier in MOTION_TIERS or
                                                 (role and role in rig.SLEEPING_ROLES and not diagnostic)):
        asleep = rig.asleep()
        if asleep:
            why = (f"the rig is asleep since {asleep.get('since')} "
                   f"(rig sleep {asleep.get('run_id')} from {asleep.get('by')})")
            if not ctx.dry_run:
                return refuse(ctx, EXIT_HW_UNREACHABLE, "rig", why, "tatbot rig wake")
            print(f"tatbot: dry-run note (rig gate would refuse): {why}", file=sys.stderr)
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
        elif profile is not None:
            profile_env = tatbot_profile.env_exports(profile)
    if v.needs_tool:
        tool, source, err = gates.resolve_tool(ctx.repo, ctx.ee_tool, arm=v.tool_arm)
        if err:
            return refuse(ctx, EXIT_GATE_REFUSED, "ee_tool", err, gates.FLAG_HINT)
        # A read-only prepare hook may already have bound an authoritative
        # identity (a saved session's recorded tool); keep whose it was.
        ctx.ee_tool, ctx.tool_source = tool, ctx.tool_source or source
    tag = getattr(ns, "tag", None)
    err = gates.tag_error(tag)
    if err:
        return refuse(ctx, EXIT_USAGE, "tag", err)
    needs_launch_id = v.launch_id

    if ctx.json and not ctx.dry_run and output_capability(v, ns, passthrough) == "text":
        return refuse(ctx, EXIT_USAGE, "output", f"{v.name} executes a text-only tool; JSON execution is unsupported",
                      "use --dry-run --json or --explain --json, or run without global --json")
    if ctx.tool_source and ctx.tool_source != "flag":
        routing_notice(ctx, v, None, gates.TOOL_SOURCES[ctx.tool_source])

    def _run_native() -> int:
        res = v.run(ctx, ns, passthrough)
        return res if isinstance(res, int) else EXIT_OK

    res = (Plan(argv=[], kind="native", action=_run_native)
           if v.native else _call_handler(ctx, v, ns, passthrough))
    if isinstance(res, int):
        return res
    plan = normalize(res)
    plan.env.update(getattr(ns, "job_env", {}))
    plan.env.update(profile_env)
    plan.env.setdefault("TATBOT_VIA_CLI", "1")
    # Propagate the canonical config/nodes.json identity to wrapped tools.
    # A host whose hostname differs from its node name cannot safely
    # reconstruct it from hostname alone, and training profiles are keyed by
    # the canonical node name.
    plan.env.setdefault("TATBOT_NODE", ctx.node)
    if ctx.ee_tool:
        plan.env.setdefault("TATBOT_EE_TOOL", ctx.ee_tool)
    # The launch id: minted here, on the node that execs, right before the exec.
    # The launcher's arm_gate reads it from the token file, ledgers and audits it.
    launch_id = gates.mint_launch_id(tag) if needs_launch_id else None
    if ctx.dry_run:
        print_plan(ctx, v, plan, launch_id=launch_id, ns=ns, passthrough=passthrough)
        return EXIT_OK
    if plan.kind == "native":
        return _call_native(ctx, v.name, plan.action)
    if launch_id:
        gates.write_launch_id(launch_id)
        if ctx.json:
            print(json.dumps({"launch_id": launch_id}), flush=True)
        elif not ctx.quiet:
            print(f"tatbot: launch id {launch_id}", flush=True)
    _warn_deprecations(ctx)
    env = dict(os.environ)
    env.update(plan.env)
    if plan.cwd:
        os.chdir(plan.cwd)
    write_files(plan)
    try:
        os.execvpe(plan.argv[0], plan.argv, env)
    except FileNotFoundError:
        return refuse(ctx, EXIT_USAGE, "exec", f"not found: {plan.argv[0]}",
                      "build it first (scripts/check builds cpp/rust) or check the path")


def _warn_deprecations(ctx):
    if ctx.invocation and ctx.invocation.deprecations:
        # Once per invocation, including multiple deprecated option spellings.
        print("tatbot: " + "; ".join(dict.fromkeys(ctx.invocation.deprecations)), file=sys.stderr)


def _call_handler(ctx, v, ns, passthrough):
    return _call_native(ctx, v.name, lambda: v.run(ctx, ns, passthrough))


def _call_native(ctx, name, action):
    if not ctx.json:
        return action()
    # Only CLI-owned handlers run here. No workflow subprocess is captured or
    # interposed in the signal path. Existing domain JSON refusals pass through.
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            result = action()
        except SystemExit as exc:
            result = exc.code if isinstance(exc.code, int) else EXIT_USAGE
            if not isinstance(exc.code, int) and exc.code is not None:
                print(str(exc.code), file=sys.stderr)
    message = stderr.getvalue()
    if isinstance(result, int) and result != 0:
        try:
            error = json.loads(message)
            assert isinstance(error, dict)
        except (ValueError, AssertionError):
            error = {"schema": "tatbot.cli-error/1", "code": result,
                     "status": EXIT_NAMES.get(result, "tool_failed"),
                     "reason": message.strip() or f"{name} failed"}
        print(json.dumps(error), file=sys.stderr)
    elif message:
        sys.stderr.write(message)
    return result


def main(argv: list[str]) -> int:
    token = _JSON_OUTPUT.set(False)
    try:
        return _main(argv)
    except UsageError as exc:
        return _usage_error(str(exc))
    except (OSError, ValueError) as exc:
        code = EXIT_BUSY if isinstance(exc, BlockingIOError) else EXIT_USAGE
        if _JSON_OUTPUT.get():
            print(json.dumps({"schema": "tatbot.cli-error/1", "code": code,
                              "status": EXIT_NAMES[code], "reason": str(exc)}), file=sys.stderr)
        else:
            print(f"tatbot: {exc}", file=sys.stderr)
        return code
    finally:
        _JSON_OUTPUT.reset(token)


def _main(argv: list[str]) -> int:
    try:
        g, noun, rest = parse_globals(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if g["version"]:
        print(json.dumps({"schema": "tatbot.version/1", "version": __version__}) if g["json"] else f"tatbot {__version__}")
        return EXIT_OK
    if noun is None:
        if g["json"] and not g["help"]:
            return _usage_error("a command is required; tatbot --help lists them")
        _root_usage(sys.stdout if g["help"] else sys.stderr, show_all=g["help_all"])
        return EXIT_OK if g["help"] else EXIT_USAGE
    if noun not in nouns():
        return _usage_error(f"unknown command '{noun}'" + (suggest_noun(noun) or "")
                            + " — `tatbot --help` lists them")
    ctx = Ctx(repo=repo_root(), node=nodes.this_node(nodes.load(repo_root())), json=g["json"], dry_run=g["dry_run"], on=g["on"],
              ee_tool=g["ee_tool"], explain=g["explain"], quiet=g["quiet"], verbose=g["verbose"],
              no_hop=g["no_hop"], argv=list(argv), invocation=g["invocation"])
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
    parser = build_noun_parser(noun, show_all=g["help_all"])
    try:
        ns, unknown = parser.parse_known_args(own)
    except SystemExit as e:
        return EXIT_OK if e.code == 0 else EXIT_USAGE
    v: Verb | None = getattr(ns, "_verb", None)
    if v is None:
        return _usage_error(f"tatbot {noun}: which one?" + missing_verb_help(noun))
    if unknown and not (v.args is None and v.passthrough and v.tier not in MOTION_TIERS):
        # A hardware verb forwards nothing it does not know: a mistyped flag used to
        # ride through to the launcher and reach the executor after the arm gate had
        # ledgered the launch id. Launcher/executor flags go after an explicit `--`.
        return refuse(ctx, EXIT_USAGE, "args",
                      f"tatbot {v.name}: unknown argument(s) {' '.join(unknown)} — this {v.tier} verb forwards "
                      "only the flags it declares",
                      f"`tatbot {v.name} --help` lists them; {v.passthrough or 'launcher'} flags go after `--`")
    if passthrough and not v.passthrough:
        raise UsageError(f"tatbot {v.name}: this command has no passthrough arguments")
    return dispatch(ctx, v, ns, unknown + passthrough)
