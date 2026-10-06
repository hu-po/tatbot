"""The verb registry: what exists, what it can physically do, what it wraps."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tatbot_cli.arguments import Arguments

# --- safety tiers -----------------------------------------------------------
#
# Every verb carries exactly one. Shown in --help, `schema` and --explain, and
# used to decide which gates the CLI itself enforces before exec.

OFFLINE = "offline"          # files only
SENSOR = "sensor"            # hardware observations
MOTION_HUMAN = "motion-human"  # arm moves under direct human guidance
MOTION_AUTO = "motion-auto"    # arm moves autonomously (rollouts, dips)
MUTATES_CONFIG = "mutates-config"  # writes a tracked config from a measurement
REMOTE = "remote"            # changes another node or service

TIERS = (OFFLINE, SENSOR, MOTION_HUMAN, MOTION_AUTO, MUTATES_CONFIG, REMOTE)

TIER_MEANING = {
    OFFLINE: "files only",
    SENSOR: "hardware observations",
    MOTION_HUMAN: "arm moves under direct human guidance",
    MOTION_AUTO: "arm moves autonomously",
    MUTATES_CONFIG: "writes a tracked config file from a measurement",
    REMOTE: "changes another node or a long-running service",
}

# What a tier implies, for the tier table. A verb's own gates (``Verb.gates``)
# are computed from what it actually declares — ``needs_tool``, ``launch_id``,
# ``ink_hook`` — so a verb never advertises a gate its launcher does not run.
GATE_ESTOP = "e-stop required (launcher: estop_guard)"
GATE_TOOL = "tool named to the launcher: --ee-tool, else TATBOT_EE_TOOL, else the owner's fitted tool"
GATE_LAUNCH_ID = "launch id ledgered and audited (launcher: arm_gate)"

TIER_GATES = {
    OFFLINE: (),
    SENSOR: (),
    MOTION_HUMAN: (GATE_ESTOP, GATE_TOOL),
    MOTION_AUTO: (GATE_ESTOP, GATE_TOOL, GATE_LAUNCH_ID),
    MUTATES_CONFIG: ("writes a tracked config file; commit the result",),
    REMOTE: ("names the target node/service in the plan",),
}

MOTION_TIERS = (MOTION_HUMAN, MOTION_AUTO)

# --- routing policies -------------------------------------------------------
#
# What `auto_hop` means for a verb whose role has more than one owner in
# config/nodes.json. Neither policy ever picks a node by position in the map.
#
#   owner        exactly one node may own this role. Absent or ambiguous
#                ownership refuses (exit 4) even when some node happens to
#                carry the role: an arm workflow that ran on "whichever node
#                answered" would command hardware the operator is not at.
#   prefer-local this node runs it when it owns the role; otherwise a single
#                owner is hopped to, and a shared role is refused with the
#                eligible targets so `--on` can name one.
OWNER = "owner"
PREFER_LOCAL = "prefer-local"
ROUTING_POLICIES = (OWNER, PREFER_LOCAL)

@dataclass
class Plan:
    """What a verb would exec. `--dry-run` prints it; a real run execs it."""

    argv: list[str]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    kind: str = "exec"
    action: Callable[[], int] | None = field(default=None, repr=False)
    # What the verb writes right before the exec, `{path: text}` (a launch
    # manifest the argv names): shown by `--dry-run`, written by a real run.
    files: dict[str, str] = field(default_factory=dict)


# The canonical order globals are re-emitted in, whatever order they were
# typed in. One spelling for execution, remote replay, plans and messages.
GLOBAL_ORDER = ("--json", "--dry-run", "--explain", "--no-hop", "-q", "-v")


@dataclass
class Invocation:
    """Token ownership, normalized once so every consumer renders the same command.

    Globals may be typed anywhere before the first ``--``; they are collected
    here by meaning rather than by position, so ``canonical`` and
    ``remote_tokens`` are rebuilt rather than sliced out of the original argv.
    ``routing`` (``--on <node>``) is held apart because the hop consumes it:
    the remote is already the named node and must not hop again.
    """

    original: list[str]
    global_tokens: list[str]          # normalized globals, routing excluded
    command_tokens: list[str]         # [noun, *owned tokens], after alias translation
    passthrough: list[str]
    routing_tokens: list[str] = field(default_factory=list)
    has_passthrough: bool = False     # an explicit `--` was typed, even with an empty tail
    deprecations: list[str] = field(default_factory=list)

    @property
    def tail_tokens(self) -> list[str]:
        return ["--", *self.passthrough] if self.has_passthrough else []

    @property
    def canonical(self) -> list[str]:
        """Exactly what this invocation means, in one normalized spelling."""
        return [*self.global_tokens, *self.routing_tokens, *self.command_tokens, *self.tail_tokens]

    def remote_tokens(self) -> list[str]:
        """The canonical form minus the routing that this hop is satisfying."""
        return [*self.global_tokens, *self.command_tokens, *self.tail_tokens]


@dataclass
class Ctx:
    repo: Path
    node: str
    json: bool = False
    dry_run: bool = False
    on: str | None = None
    ee_tool: str | None = None
    tool_source: str | None = None  # flag / environment / configured; see gates.TOOL_SOURCES
    explain: bool = False
    quiet: bool = False
    verbose: bool = False
    no_hop: bool = False
    argv: list[str] = field(default_factory=list)  # the full original argv
    invocation: Invocation | None = None

    def path(self, rel: str) -> str:
        return str(self.repo / rel)


Handler = Callable[[Ctx, argparse.Namespace, list[str]], "Plan | list[str] | int"]


@dataclass(frozen=True)
class RoleFrom:
    """A fleet role resolved from the invocation's own operands, for a verb no
    one node owns for every invocation (`vision capture`: the wrist camera owner
    of `--arm`). `describe` is what the tables and `--explain` print; `resolve`
    returns the role the parsed operands need, or raises ValueError when they
    name no single owner role."""
    describe: str
    resolve: Callable[[argparse.Namespace], str]


@dataclass
class Verb:
    noun: str
    verb: str                      # "" for a noun that is its own command (status)
    tier: str
    summary: str
    run: Handler
    role: str | None = None        # config/nodes.json role this verb needs
    role_for: RoleFrom | None = None  # or the role each invocation resolves from its operands
    wraps: tuple[str, ...] = ()    # repo-relative files this verb delegates to
    doc: str | None = None         # where the human documentation lives
    example: tuple[str, ...] = ()  # args after the verb; used by selfcheck + docs
    args: Callable[[argparse.ArgumentParser], None] | None = None
    needs_tool: bool = False       # --ee-tool must be stated
    tool_arm: str = "right"        # physical arm whose fitted-tool pointer supplies the default
    launch_id: bool = False        # the launch carries an automatic launch id (ledgered + audited by arm_gate); --tag labels it
    ink_hook: bool = False         # takes --no-ink (scripts/lib/ink_hook.sh)
    passthrough: str | None = None # what receives the args after `--`
    tty: bool = False              # --on hops with `ssh -t`
    auto_hop: bool = False         # route to the role owner before execution-node gates
    routing: str = "owner"         # how auto_hop resolves the owner; see ROUTING_POLICIES
    sync: bool = False             # a hop fast-forwards the remote checkout first (git pull --ff-only)
    invariants: tuple[str, ...] = ()  # printed by --explain
    argument_spec: Arguments = field(default_factory=Arguments)
    group: str = ""
    effects: tuple[str, ...] = ()
    visibility: str = "private"   # PUBLIC EXPORT visibility; not the help audience
    audience: str = ""            # who the command is for (tatbot_cli.audience)
    disposition: str = ""         # why it sits there; part of the command audit
    canonical_name: str | None = None
    deprecated: str | None = None
    native: bool = False           # defer native execution until after planning
    prepare: Callable | None = None  # read-only input binding on the EXECUTION node, before gates
    validate: Callable | None = None  # pure check of the parsed namespace, BEFORE routing:
                                      # an impossible selection must not cost an ssh. Returns an
                                      # exit code to stop, or None to continue.
    refine_effects: Callable | None = None  # (effects, namespace, backend args) -> effects
    select_output: Callable | None = None
    output_modes: dict[str, str] = field(default_factory=dict)
    output: str = "text"           # json, json-lines, or text; execution only

    @property
    def name(self) -> str:
        return f"{self.noun} {self.verb}".strip()

    def required_role(self, ns: argparse.Namespace | None = None) -> str | None:
        """The fleet role this invocation needs: the declared `role`, or what
        `role_for` resolves from the parsed operands (without them, the
        declared role, which is None for such a verb)."""
        if self.role_for is not None and ns is not None:
            return self.role_for.resolve(ns)
        return self.role

    @property
    def gates(self) -> tuple[str, ...]:
        """The gates this verb really runs, from its declaration, not its tier."""
        g: list[str] = []
        if self.tier in MOTION_TIERS:
            g.append(GATE_ESTOP)
        if self.needs_tool:
            g.append(GATE_TOOL)
        if self.launch_id:
            g.append(GATE_LAUNCH_ID)
        if self.tier in (MUTATES_CONFIG, REMOTE):
            g.extend(TIER_GATES[self.tier])
        return tuple(g)

    def autonomous(self, ns) -> bool:
        """Does THIS invocation move the arm on its own? A motion-auto verb does."""
        return self.tier == MOTION_AUTO



# One noun declaration owns ordering, help summary, and command grouping.
NOUNS = {
    "status": ("operator", "local observations; explicit fleet collection"),
    "schema": ("administration", "the command tree, tiers and nodes as JSON or Markdown"),
    "completion": ("administration", "generate opt-in shell completion"),
    "check": ("administration", "every check this repo has (scripts/check)"),
    "logs": ("operator", "find and read the full log of any run"),
    "estop": ("operator", "the Pico e-stop, bench-checked"),
    "arm": ("operator", "reach, recover and land the Trossen arms"),
    "tool": ("operator", "the end-effector tool registry"),
    "body": ("development", "immutable human body model assets and cache audit"),
    "ink": ("operator", "inks, caps, palette and the ledger"),
    "teleop": ("operator", "leader→follower teleoperation"),
    "ros": ("operator", "the ROS 2 drawing stack: deploy, up/down, compile, draw, touch, decide"),
    "rollout": ("operator", "run and read trained policies on the arm"),
    "serve": ("administration", "the async policy server"),
    "train": ("development", "policy training on the training nodes"),
    "data": ("development", "datasets: the hub archive and the LeRobot dataset tools"),
    "sim": ("development", "scenario, dataset, evaluation and render tools"),
    "travel": ("development", "travel demo previews and synthetic tracing episodes"),
    "calib": ("operator", "native arm measurements and retained overhead registration"),
    "vision": ("operator", "cameras, calibration, tracking, deploy"),
    "live": ("operator", "every live sensor in one Rerun viewer"),
    "node": ("administration", "the node→role map and ssh dispatch"),
    "design": ("development", "an image or an SVG to a portable design, without a browser"),
    "research": ("development", "DBV3 paired drawing research with frozen inputs and durable recovery"),
    "drawingbot": ("development", "DrawingBotV3 stroke experiments and preference review"),
    "inkmap": ("development", "ground InkLang placements and preview tattoos on canonical bodies"),
    "inkgen": ("development", "generate tattoo artwork for Inkmap"),
    "profile": ("administration", "inspect and validate hardware profiles"),
    "viewer": ("operator", "the one persistent Rerun viewer every workflow streams into"),
    "deploy": ("administration", "build and deploy manifested fleet services"),
    "rig": ("operator", "sleep and wake the rig: camera and bus services, rig hosts"),
    "work": ("development", "one session, one worktree, one branch, pushed always"),
    "release": ("administration", "publish the public export of main to hu-po/tatbot"),
}
NOUN_ORDER = tuple(NOUNS)
NOUN_SUMMARY = {name: summary for name, (_, summary) in NOUNS.items()}


_REGISTRY: list[Verb] = []


def verb(**kw) -> Callable[[Handler], Handler]:
    """Register a verb. ``@verb(noun=…, verb=…, tier=…, summary=…)``."""

    def deco(fn: Handler) -> Handler:
        record = Verb(run=fn, **kw)
        record.argument_spec = Arguments.capture(record.args)
        from tatbot_cli.metadata import apply_metadata
        apply_metadata(record)
        _REGISTRY.append(record)
        return fn

    return deco


# Against the tree THIS code shipped in, not TATBOT_REPO: that env points
# tools at another checkout's config, but a verb's backing scripts live
# (or were export-excluded) beside the registry itself.
_TREE = Path(__file__).resolve().parents[3]


def _available(v: "Verb") -> bool:
    """A verb exists only where its wrapped scripts do. The public export
    excludes fleet-orchestration scripts (plan Phase 5), and the CLI's help
    must be truthful on that tree: a verb missing any backing script is
    hidden rather than advertised and broken. Native verbs (no wraps) are
    always available."""
    if not v.wraps:
        return True
    return all((_TREE / w).exists() for w in v.wraps)


# (registry object, its length, every verb in order, the available ones).
# Ordering and availability are facts of the shipped tree, fixed for the
# life of a process; recomputing them per call made `--help` stat every
# backing script ~37,000 times (verbs_of() per noun, each re-sorting with an
# O(n) `_REGISTRY.index`), several seconds per CLI invocation. The snapshot
# keys on the registry list itself, so a test that swaps `_REGISTRY` sees
# its own verbs.
_SNAPSHOT: tuple[list, int, tuple[Verb, ...], tuple[Verb, ...]] | None = None


def all_verbs(*, include_unavailable: bool = False) -> list[Verb]:
    from tatbot_cli import verbs  # noqa: F401  (import registers everything)

    global _SNAPSHOT
    if _SNAPSHOT is None or _SNAPSHOT[0] is not _REGISTRY or _SNAPSHOT[1] != len(_REGISTRY):
        order = {n: i for i, n in enumerate(NOUN_ORDER)}
        ranked = sorted(enumerate(_REGISTRY), key=lambda iv: (order.get(iv[1].noun, 99), iv[0]))
        every = tuple(v for _, v in ranked)
        _SNAPSHOT = (_REGISTRY, len(_REGISTRY), every, tuple(v for v in every if _available(v)))
    return list(_SNAPSHOT[2] if include_unavailable else _SNAPSHOT[3])


def nouns() -> list[str]:
    seen: dict[str, None] = {}
    for v in all_verbs():
        seen.setdefault(v.noun, None)
    return list(seen)


def verbs_of(noun: str) -> list[Verb]:
    return [v for v in all_verbs() if v.noun == noun]


def find(noun: str, verb_name: str) -> Verb | None:
    for v in verbs_of(noun):
        if v.verb == verb_name:
            return v
    return None


def repo_root() -> Path:
    """TATBOT_REPO, else this checkout -- tatbot_paths owns the rule."""
    from tatbot_paths import repo_root as _root
    return _root()
