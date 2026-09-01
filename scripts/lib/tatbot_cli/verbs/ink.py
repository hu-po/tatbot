"""ink — inks, caps, palette, the ledger and the session (scripts/ink.py).

Until 2026-08-29 `tatbot ink` was one `offline` passthrough over the whole
ink.py tree. It was not offline: `load`/`dump`/`bottle`/`cartridge`/`caps`
rewrite tracked config, `reconcile --write` rewrites the inventory, `session
start`/`end` change what the next rollout debits, and `sync` scp's other nodes'
ledgers. One verb per subcommand makes the tier, `--explain` and `--dry-run`
honest, and lets `session start`, `plan` and `mise-en-place` take the stated
`--ee-tool` like every other tool-bearing verb. Options are passed through by
position, so `tatbot ink load <slot> <ink> --ul 800` is `ink.py load …`.
"""

from __future__ import annotations

from tatbot_cli import nodes
from tatbot_cli.registry import MUTATES_CONFIG, OFFLINE, REMOTE, verb
from tatbot_cli.verbs._common import py, tool_flag

INK = "scripts/ink.py"
WRAPS = (INK, "scripts/lib/ink_spec.py", "scripts/lib/ink_session.py")
DOC = "docs/ink.md"
LEDGER_INV = "The ledger is append-only; every event carries an id, so re-reading or re-syncing never double-counts."
COMMIT_INV = "Writes the file at once (no --write); commit config/palette_load.yaml / inventory.yaml afterwards — they are session facts every node should agree on."


def _ink(ctx, sub: str, *args, tool: bool = False, **kw):
    return py(ctx, INK, sub, *(tool_flag(ctx) if tool else []), *args, **kw)


# --- read-only -----------------------------------------------------------------


@verb(noun="ink", verb="status", tier=OFFLINE, summary="palette load, stock, the open session and the ledger tail",
      wraps=WRAPS, passthrough="ink.py status", example=(), doc=DOC)
def ink_status(ctx, ns, rest):
    return _ink(ctx, "status", *rest)


def _ledger_args(p):
    p.add_argument("--since", help="ISO time")
    p.add_argument("--mode", choices=["real", "rehearsal", "sim"])
    p.add_argument("-n", type=int, default=0, help="last N events")


@verb(noun="ink", verb="ledger", tier=OFFLINE, summary="the append-only event ledger (local + synced remote copies)",
      wraps=WRAPS, passthrough="ink.py ledger", args=_ledger_args, example=("-n", "20"), doc=DOC, invariants=(LEDGER_INV,))
def ink_ledger(ctx, ns, rest):
    flags = (["--since", ns.since] if ns.since else []) + (["--mode", ns.mode] if ns.mode else []) + \
        (["-n", str(ns.n)] if ns.n else [])
    return _ink(ctx, "ledger", *flags, *rest)


@verb(noun="ink", verb="fit", tier=OFFLINE, summary="fit uptake / deposit / bleed from before-after cap weighings; prints the datasheet edit",
      wraps=WRAPS, passthrough="ink.py fit", example=(), doc=DOC,
      invariants=("Prints the edit; the datasheet (config/tools/<id>.yaml) is changed by hand, deliberately.",))
def ink_fit(ctx, ns, rest):
    return _ink(ctx, "fit", *rest)


def _plan_args(p):
    p.add_argument("--strokes", nargs="+", required=True, metavar="MM,S[,INK]", help="planned strokes")


@verb(noun="ink", verb="plan", tier=OFFLINE, summary="dry-run the dip planner for the stated tool over planned strokes",
      wraps=WRAPS, passthrough="ink.py plan", args=_plan_args, needs_tool=True, example=("--strokes", "120,6", "300,15"), doc=DOC)
def ink_plan(ctx, ns, rest):
    return _ink(ctx, "plan", "--strokes", *ns.strokes, *rest, tool=True)


@verb(noun="ink", verb="mise-en-place", tier=OFFLINE, summary="the human setup checklist for a session: caps to fill, cartridge, weigh-in",
      wraps=WRAPS, passthrough="ink.py mise-en-place", needs_tool=True, example=("--need", "nighthawk_black=600"), doc=DOC,
      invariants=("Reads state and changes nothing; the need comes from --need or --program.",))
def ink_mise(ctx, ns, rest):
    return _ink(ctx, "mise-en-place", *rest, tool=True)


# --- the session -----------------------------------------------------------------


@verb(noun="ink", verb="session", tier=OFFLINE, summary="what the needle carries now: the open session on this node",
      wraps=WRAPS, passthrough="ink.py session", example=(), doc=DOC)
def ink_session_status(ctx, ns, rest):
    return _ink(ctx, "session", "status", *rest)


@verb(noun="ink", verb="session start", tier=OFFLINE, summary="open the session for the stated tool (one per node; --need-ul / --program)",
      wraps=WRAPS, passthrough="ink.py session start", needs_tool=True, example=(), doc=DOC,
      invariants=("One open session per node — one tool in one gripper; another tool's session is refused until ended.",
                  "il_dip opens a session itself when none is open; start one by hand to declare the planned need."))
def ink_session_start(ctx, ns, rest):
    return _ink(ctx, "session", "start", *rest, tool=True)


@verb(noun="ink", verb="session end", tier=OFFLINE, summary="close the open session with its totals",
      wraps=WRAPS, passthrough="ink.py session end", example=(), doc=DOC)
def ink_session_end(ctx, ns, rest):
    return _ink(ctx, "session", "end", *rest)


def _rebuild_args(p):
    p.add_argument("id", help="session id")
    p.add_argument("--write", action="store_true", help="also write session.json")


@verb(noun="ink", verb="session rebuild", tier=OFFLINE, summary="prove a session from its ledger events (the file is a cache)",
      wraps=WRAPS, passthrough="ink.py session rebuild", args=_rebuild_args, example=(f"20260829_120000-{nodes.example_node()}-ab12",), doc=DOC)
def ink_session_rebuild(ctx, ns, rest):
    return _ink(ctx, "session", "rebuild", ns.id, *(["--write"] if ns.write else []), *rest)


# --- tracked config ---------------------------------------------------------------


def _load_args(p):
    p.add_argument("slot", help="palette slot, e.g. inkcap_right_medium_0")
    p.add_argument("ink", help="ink id from config/inks.yaml")


@verb(noun="ink", verb="load", tier=MUTATES_CONFIG, summary="put ink in a cap: --ul, --bottle, --cap-stock → palette_load.yaml + cap.fill",
      wraps=WRAPS, passthrough="ink.py load", args=_load_args,
      example=("inkcap_right_medium_0", "nighthawk_black", "--ul", "400"), doc=DOC, invariants=(COMMIT_INV, LEDGER_INV))
def ink_load(ctx, ns, rest):
    return _ink(ctx, "load", ns.slot, ns.ink, *rest)


def _slot_arg(p):
    p.add_argument("slot")


@verb(noun="ink", verb="dump", tier=MUTATES_CONFIG, summary="empty a cap → palette_load.yaml + cap.dump",
      wraps=WRAPS, passthrough="ink.py dump", args=_slot_arg, example=("inkcap_right_medium_0",), doc=DOC,
      invariants=(COMMIT_INV,))
def ink_dump(ctx, ns, rest):
    return _ink(ctx, "dump", ns.slot, *rest)


def _bottle_args(p):
    p.add_argument("action", choices=["add", "open", "retire"])
    p.add_argument("id", help="bottle id")


@verb(noun="ink", verb="bottle", tier=MUTATES_CONFIG, summary="add / open / retire a bottle → inventory.yaml",
      wraps=WRAPS, passthrough="ink.py bottle", args=_bottle_args, example=("open", "nighthawk_black_01"), doc=DOC,
      invariants=(COMMIT_INV,))
def ink_bottle(ctx, ns, rest):
    return _ink(ctx, "bottle", ns.action, ns.id, *rest)


def _cartridge_args(p):
    p.add_argument("action", choices=["add", "fit", "count", "retire"])
    p.add_argument("id", help="cartridge box id")
    p.add_argument("n", nargs="?", help="count (fit/count/retire)")


@verb(noun="ink", verb="cartridge", tier=MUTATES_CONFIG, summary="add / fit / count / retire needle cartridges → inventory.yaml",
      wraps=WRAPS, passthrough="ink.py cartridge", args=_cartridge_args, example=("count", "quelle_1003rl_box01", "18"), doc=DOC,
      invariants=(COMMIT_INV,))
def ink_cartridge(ctx, ns, rest):
    return _ink(ctx, "cartridge", ns.action, ns.id, *([ns.n] if ns.n else []), *rest)


def _caps_args(p):
    p.add_argument("action", choices=["count"])
    p.add_argument("id", help="cap stock id")
    p.add_argument("n", help="count")


@verb(noun="ink", verb="caps", tier=MUTATES_CONFIG, summary="count blank caps → inventory.yaml",
      wraps=WRAPS, passthrough="ink.py caps", args=_caps_args, example=("count", "emalla_15mm", "40"), doc=DOC,
      invariants=(COMMIT_INV,))
def ink_caps(ctx, ns, rest):
    return _ink(ctx, "caps", ns.action, ns.id, ns.n, *rest)


def _weigh_args(p):
    p.add_argument("target", help="slot or bottle id")
    p.add_argument("grams", help="0.01 g scale reading")
    p.add_argument("--when", choices=["before", "after"], required=True)


@verb(noun="ink", verb="weigh", tier=OFFLINE, summary="record a cap or bottle weighing (the calibration input for `ink fit`)",
      wraps=WRAPS, passthrough="ink.py weigh", args=_weigh_args, example=("inkcap_right_medium_0", "4.31", "--when", "before"),
      doc=DOC, invariants=("Appends one `weigh` event to the ledger; nothing else changes.",))
def ink_weigh(ctx, ns, rest):
    return _ink(ctx, "weigh", ns.target, ns.grams, "--when", ns.when, *rest)


def _reconcile_args(p):
    p.add_argument("--write", action="store_true", help="fold ledger consumption into inventory.yaml")


@verb(noun="ink", verb="reconcile", tier=MUTATES_CONFIG, summary="ledger consumption vs inventory.yaml and weighings; --write folds it in",
      wraps=WRAPS, passthrough="ink.py reconcile", args=_reconcile_args, example=(), doc=DOC,
      invariants=("Shows the drift; only --write changes inventory.yaml. It explains a mismatch, it never rewrites the ledger.",))
def ink_reconcile(ctx, ns, rest):
    return _ink(ctx, "reconcile", *(["--write"] if ns.write else []), *rest)


# --- other nodes ------------------------------------------------------------------


def _sync_args(p):
    p.add_argument("nodes", nargs="+", help="ssh targets (node name or user@host)")


@verb(noun="ink", verb="sync", tier=REMOTE, summary="scp other nodes' ledgers into <ledger dir>/remote/ so every reader sees them",
      wraps=WRAPS, passthrough="ink.py sync", args=_sync_args, example=(nodes.example_node(),), doc=DOC,
      invariants=(LEDGER_INV, "Read-only on the remote node: it copies the file, it never writes there."))
def ink_sync(ctx, ns, rest):
    return _ink(ctx, "sync", *ns.nodes, *rest, notes=[f"scp from {', '.join(ns.nodes)} (BatchMode, 8 s timeout)"])
