"""ink — inks, caps, palette, the ledger and the session (scripts/ink.py).

Six verbs over ink.py's seventeen subcommands, one per thing an operator
means (2026-09-02; the 08-29 one-verb-per-subcommand tree had 17 verbs, 12
of them never typed):

- `ink status` reads — palette, stock, the session (`--session`) and the
  ledger (`--ledger`); all three are `ink.py status|session status|ledger`.
- `ink mise-en-place` is the pre-session checklist for the stated tool;
  `--strokes` turns it into the dip planner's dry run (`ink.py plan`).
- `ink session start|end` open and close what the next run debits; `start`
  takes the stated `--ee-tool` like every other tool-bearing verb.
- `ink edit -- <load|dump|bottle|cartridge|caps|reconcile|weigh> …` is the
  one `mutates-config` verb: those seven rewrite tracked config or append a
  ledger event, and the tier is the only thing the CLI adds — ink.py validates
  its own arguments. Any other subcommand after `edit --` is refused, so a
  read never masquerades as an edit.
- `ink sync` scp's other nodes' ledgers (`remote`).
- bare `ink -- …` passes anything else ink.py offers (`fit`, `session
  rebuild <id>`, `plan`, `--help`) through as `offline`; it refuses the edit
  subcommands and `sync`, so a write never masquerades as offline.
"""

from __future__ import annotations

import sys

from tatbot_cli import EXIT_OK, EXIT_USAGE, nodes
from tatbot_cli.registry import MUTATES_CONFIG, OFFLINE, REMOTE, verb
from tatbot_cli.verbs._common import py, tool_flag

INK = "scripts/ink.py"
WRAPS = (INK, "scripts/lib/ink_spec.py", "scripts/lib/ink_session.py")
DOC = "docs/ink.md"
LEDGER_INV = "The ledger is append-only; every event carries an id, so re-reading or re-syncing never double-counts."
COMMIT_INV = "Writes the file at once (no --write); commit config/palette_load.yaml / inventory.yaml afterwards — they are session facts every node should agree on."

# ink.py subcommands that rewrite tracked config or append a ledger event.
EDIT_SUBS = ("load", "dump", "bottle", "cartridge", "caps", "reconcile", "weigh")
EDIT_USAGE = """\
usage: tatbot ink edit -- <subcommand> …        (mutates-config; commit the result)

  load <slot> <ink_id> --ul <n> [--bottle <id>] [--cap-stock <id>]   put ink in a cap → palette_load.yaml + cap.fill
  dump <slot>                                     empty a cap → palette_load.yaml + cap.dump
  bottle add <id> --ink <ink_id> --ml <n> | open <id> | retire <id>   → inventory.yaml
  cartridge add|fit|count|retire <id> [n]         needle cartridges → inventory.yaml
  caps count <id> <n>                             blank caps → inventory.yaml
  weigh <slot|bottle_id> <grams> --when before|after   one `weigh` ledger event (the input of `ink -- fit`)
  reconcile [--write]                             ledger use vs inventory.yaml; --write folds it in

`tatbot ink edit -- <subcommand> --help` shows that subcommand's options (ink.py validates them).
Reads are `tatbot ink status`; anything else ink.py offers is `tatbot ink -- …`."""


def _ink(ctx, sub: str, *args, tool: bool = False, **kw):
    return py(ctx, INK, sub, *(tool_flag(ctx) if tool else []), *args, **kw)


# --- read-only -----------------------------------------------------------------


def _status_args(p):
    p.add_argument("--session", action="store_true", help="the open session on this node (ink.py session status)")
    p.add_argument("--ledger", action="store_true", help="the append-only event ledger, local + synced remote copies (ink.py ledger)")
    p.add_argument("--since", metavar="ISO", help="ledger: events since this time (implies --ledger)")
    p.add_argument("--mode", choices=["real", "rehearsal", "sim"], help="ledger: one mode (implies --ledger)")
    p.add_argument("-n", type=int, default=0, metavar="N", help="ledger: last N events (implies --ledger)")


@verb(effects=('read_files',), noun="ink", verb="status", tier=OFFLINE,
      summary="palette load, stock, the ledger tail; --session the open session; --ledger [-n N] the events",
      wraps=WRAPS, passthrough="ink.py status | session status | ledger", args=_status_args, example=(), doc=DOC,
      invariants=(LEDGER_INV,))
def ink_status(ctx, ns, rest):
    ledger = ns.ledger or bool(ns.since or ns.mode or ns.n)
    if ledger and ns.session:
        print("tatbot ink status: --ledger and --session are two different reads; pick one", file=sys.stderr)
        return EXIT_USAGE
    if ns.session:
        return _ink(ctx, "session", "status", *rest)
    if ledger:
        flags = (["--since", ns.since] if ns.since else []) + (["--mode", ns.mode] if ns.mode else []) + \
            (["-n", str(ns.n)] if ns.n else [])
        return _ink(ctx, "ledger", *flags, *rest)
    return _ink(ctx, "status", *rest)


def _mise_args(p):
    p.add_argument("--strokes", nargs="+", metavar="MM,S[,INK]",
                   help="planned strokes: dry-run the dip planner for the stated tool instead of the checklist (ink.py plan)")


@verb(effects=('read_files',), noun="ink", verb="mise-en-place", tier=OFFLINE,
      summary="the human setup checklist for a session: caps to fill, cartridge, weigh-in; --strokes dry-runs the dip planner",
      wraps=WRAPS, passthrough="ink.py mise-en-place | plan", args=_mise_args, needs_tool=True,
      example=("--", "--need", "nighthawk_black=600"), doc=DOC,
      invariants=("Reads state and changes nothing; the need comes from --need or --program, or from --strokes for the planner.",))
def ink_mise(ctx, ns, rest):
    if ns.strokes:
        return _ink(ctx, "plan", "--strokes", *ns.strokes, *rest, tool=True)
    return _ink(ctx, "mise-en-place", *rest, tool=True)


# --- the session -----------------------------------------------------------------


@verb(effects=('read_files', 'write_files'), noun="ink", verb="session start", tier=OFFLINE, summary="open the session for the stated tool (one per node; --need-ul / --program)",
      wraps=WRAPS, passthrough="ink.py session start", needs_tool=True, example=(), doc=DOC,
      invariants=("One open session per node — one tool in the mount; another tool's session is refused until ended.",
                  "Start one by hand to declare the planned need.",
                  "Read it back with `tatbot ink status --session`; prove it from the ledger with `tatbot ink -- session rebuild <id>`."))
def ink_session_start(ctx, ns, rest):
    return _ink(ctx, "session", "start", *rest, tool=True)


@verb(effects=('read_files', 'write_files'), noun="ink", verb="session end", tier=OFFLINE, summary="close the open session with its totals",
      wraps=WRAPS, passthrough="ink.py session end", example=(), doc=DOC)
def ink_session_end(ctx, ns, rest):
    return _ink(ctx, "session", "end", *rest)


# --- tracked config ---------------------------------------------------------------


@verb(effects=('read_files', 'write_config'), noun="ink", verb="edit", tier=MUTATES_CONFIG,
      summary="-- load | dump | bottle | cartridge | caps | reconcile | weigh …: palette_load.yaml, inventory.yaml, ledger events",
      wraps=WRAPS, passthrough="ink.py " + "|".join(EDIT_SUBS),
      example=("--", "load", "inkcap_medium_1", "nighthawk_black", "--ul", "400"), doc=DOC,
      invariants=(COMMIT_INV, LEDGER_INV,
                  "Only the seven writing subcommands are accepted after --; a read is `ink status`, anything else is `ink -- …`.",
                  "reconcile shows the drift and only --write changes inventory.yaml; it explains a mismatch, it never rewrites the ledger."))
def ink_edit(ctx, ns, rest):
    if not rest or rest[0] in ("-h", "--help"):
        print(EDIT_USAGE, file=sys.stdout if rest else sys.stderr)
        return EXIT_OK if rest else EXIT_USAGE
    if rest[0] not in EDIT_SUBS:
        print(f"tatbot ink edit: `{rest[0]}` does not edit anything — edit takes {'|'.join(EDIT_SUBS)}; "
              f"reads are `tatbot ink status`, the rest `tatbot ink -- {' '.join(rest)}`", file=sys.stderr)
        return EXIT_USAGE
    return _ink(ctx, rest[0], *rest[1:])


# --- other nodes ------------------------------------------------------------------


def _sync_args(p):
    p.add_argument("nodes", nargs="+", help="ssh targets (node name or user@host)")


@verb(effects=('read_files', 'write_files', 'network', 'remote_exec'), noun="ink", verb="sync", tier=REMOTE, summary="scp other nodes' ledgers into <ledger dir>/remote/ so every reader sees them",
      wraps=WRAPS, passthrough="ink.py sync", args=_sync_args, example=(nodes.example_node(),), doc=DOC,
      invariants=(LEDGER_INV, "Read-only on the remote node: it copies the file, it never writes there."))
def ink_sync(ctx, ns, rest):
    return _ink(ctx, "sync", *ns.nodes, *rest, notes=[f"scp from {', '.join(ns.nodes)} (BatchMode, 8 s timeout)"])


# --- everything else ink.py offers ---------------------------------------------------


def _bare_args(p):
    p.description = (p.description or "") + (
        "\n\nanything else ink.py offers passes through after `--` (offline):\n"
        "  tatbot ink -- fit                       refit uptake/deposit/bleed from weigh events\n"
        "  tatbot ink -- session rebuild <id>      prove a session from its ledger events\n"
        "  tatbot ink -- --help                    ink.py's own usage")


@verb(effects=('read_files', 'write_files'), noun="ink", verb="", tier=OFFLINE, summary="anything else ink.py offers: `-- fit`, `-- session rebuild <id>`, `-- --help`",
      wraps=WRAPS, passthrough="ink.py", args=_bare_args, example=("--", "fit"), doc=DOC,
      invariants=("Offline passthrough only: the writing subcommands are `ink edit --`, sync is `ink sync`; both are refused here.",))
def ink_passthrough(ctx, ns, rest):
    if not rest:
        print("tatbot ink: which verb? (`tatbot ink --help`) — or `tatbot ink -- <ink.py subcommand> …`", file=sys.stderr)
        return EXIT_USAGE
    if rest[0] in EDIT_SUBS:
        print(f"tatbot ink: `{rest[0]}` writes tracked config or the ledger — that is "
              f"`tatbot ink edit -- {' '.join(rest)}` (mutates-config)", file=sys.stderr)
        return EXIT_USAGE
    if rest[0] == "sync":
        print(f"tatbot ink: sync reaches other nodes — that is `tatbot ink sync {' '.join(rest[1:])}` (remote)".rstrip(),
              file=sys.stderr)
        return EXIT_USAGE
    return _ink(ctx, rest[0], *rest[1:])
