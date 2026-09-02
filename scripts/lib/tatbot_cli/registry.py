"""The verb registry: what exists, what it can physically do, what it wraps."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# --- safety tiers -----------------------------------------------------------
#
# Every verb carries exactly one. Shown in --help, `schema` and --explain, and
# used to decide which gates the CLI itself enforces before exec.

OFFLINE = "offline"          # files only
SENSOR = "sensor"            # reads hardware, nothing moves
MOTION_HUMAN = "motion-human"  # arm moves, human on the leader arm
MOTION_AUTO = "motion-auto"    # arm moves autonomously (rollouts, dips)
MUTATES_CONFIG = "mutates-config"  # writes a tracked config from a measurement
REMOTE = "remote"            # changes another node or service

TIERS = (OFFLINE, SENSOR, MOTION_HUMAN, MOTION_AUTO, MUTATES_CONFIG, REMOTE)

TIER_MEANING = {
    OFFLINE: "files only",
    SENSOR: "reads hardware, nothing moves",
    MOTION_HUMAN: "arm moves, human on the leader arm with the e-stop",
    MOTION_AUTO: "arm moves autonomously",
    MUTATES_CONFIG: "writes a tracked config file from a measurement",
    REMOTE: "changes another node or a long-running service",
}

# What a tier implies, for the tier table. A verb's own gates (``Verb.gates``)
# are computed from what it actually declares — ``needs_tool``, ``nonce``,
# ``dip_hook`` — so a verb never advertises a gate its launcher does not run.
TIER_GATES = {
    OFFLINE: (),
    SENSOR: (),
    MOTION_HUMAN: ("e-stop required (launcher: estop_guard)", "--ee-tool required"),
    MOTION_AUTO: (
        "e-stop required (launcher: estop_guard)",
        "--ee-tool required",
        "single-use arm nonce (launcher: arm_gate)",
    ),
    MUTATES_CONFIG: ("writes a tracked config file; commit the result",),
    REMOTE: ("names the target node/service in the plan",),
}

MOTION_TIERS = (MOTION_HUMAN, MOTION_AUTO)

GATE_ESTOP = "e-stop required (launcher: estop_guard)"
GATE_TOOL = "--ee-tool required"
GATE_NONCE = "single-use arm nonce (launcher: arm_gate)"
GATE_DIP = "with --dip: motion-auto — single-use arm nonce (launcher: arm_gate), armed on the arm node also over --on"


@dataclass
class Plan:
    """What a verb would exec. `--dry-run` prints it; a real run execs it."""

    argv: list[str]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Ctx:
    repo: Path
    node: str
    json: bool = False
    dry_run: bool = False
    on: str | None = None
    ee_tool: str | None = None
    explain: bool = False
    quiet: bool = False
    verbose: bool = False
    no_hop: bool = False
    argv: list[str] = field(default_factory=list)  # the full original argv

    def path(self, rel: str) -> str:
        return str(self.repo / rel)


Handler = Callable[[Ctx, argparse.Namespace, list[str]], "Plan | list[str] | int"]


@dataclass
class Verb:
    noun: str
    verb: str                      # "" for a noun that is its own command (record, dip)
    tier: str
    summary: str
    run: Handler
    role: str | None = None        # config/nodes.json role this verb needs
    wraps: tuple[str, ...] = ()    # repo-relative files this verb delegates to
    doc: str | None = None         # where the human documentation lives
    example: tuple[str, ...] = ()  # args after the verb; used by selfcheck + docs
    args: Callable[[argparse.ArgumentParser], None] | None = None
    needs_tool: bool = False       # --ee-tool must be stated
    nonce: bool = False            # --nonce <literal> arms the launcher's arm_gate
    nonce_exempt: tuple[str, ...] = ()  # args dests under which nothing is commanded, so no nonce (dip --plan)
    dip_hook: bool = False         # takes --dip/--no-ink (scripts/lib/dip_hook.sh); --dip escalates to motion-auto
    passthrough: str | None = None # what receives the args after `--`
    tty: bool = False              # --on hops with `ssh -t`
    auto_hop: bool = False         # lacking the role here, hop to the role's node instead of refusing (never motion)
    sync: bool = False             # a hop fast-forwards the remote checkout first (git pull --ff-only)
    invariants: tuple[str, ...] = ()  # printed by --explain

    @property
    def name(self) -> str:
        return f"{self.noun} {self.verb}".strip()

    @property
    def gates(self) -> tuple[str, ...]:
        """The gates this verb really runs, from its declaration, not its tier."""
        g: list[str] = []
        if self.tier in MOTION_TIERS:
            g.append(GATE_ESTOP)
        if self.needs_tool:
            g.append(GATE_TOOL)
        if self.nonce:
            g.append(GATE_NONCE + (f" (not with {self.nonce_exempt_flags})" if self.nonce_exempt else ""))
        if self.dip_hook:
            g.append(GATE_DIP)
        if self.tier in (MUTATES_CONFIG, REMOTE):
            g.extend(TIER_GATES[self.tier])
        return tuple(g)

    def autonomous(self, ns) -> bool:
        """Does THIS invocation move the arm on its own? A motion-auto verb
        always does; a motion-human verb does the moment --dip is given (the
        scripted dip runs before the human touches the leader arm)."""
        return self.tier == MOTION_AUTO or (self.dip_hook and bool(getattr(ns, "dip", False)))

    @property
    def nonce_exempt_flags(self) -> str:
        return " / ".join("--" + d.replace("_", "-") for d in self.nonce_exempt)

    def needs_nonce(self, ns) -> bool:
        if any(getattr(ns, d, False) for d in self.nonce_exempt):
            return False
        return self.nonce or (self.dip_hook and bool(getattr(ns, "dip", False)))


# Nouns, in the order an operator meets them in a day. --help and docs follow it.
NOUN_ORDER = (
    "status", "schema", "check", "logs", "estop", "arm", "tool", "ink",
    "teleop", "draw", "record", "dip", "rollout", "serve", "train", "data", "sim",
    "vision", "live", "depth", "audio", "net", "node", "inkmap",
    "inkgen",
)

NOUN_SUMMARY = {
    "status": "the fleet in one screen",
    "schema": "the command tree, tiers and nodes as JSON or Markdown",
    "check": "every check this repo has (scripts/check)",
    "logs": "find and read the full log of any run",
    "estop": "the Pico e-stop, bench-checked or simulated",
    "arm": "reach, recover and land the Trossen arms",
    "tool": "the end-effector tool registry",
    "ink": "inks, caps, palette and the ledger",
    "teleop": "leader→follower teleoperation and tuning",
    "draw": "map a surface with the wrist D405s, then draw on it",
    "record": "record imitation-learning episodes",
    "dip": "dip the fitted tool into the palette",
    "rollout": "run and read trained policies on the arm",
    "serve": "the async policy server",
    "train": "policy training on the training nodes",
    "data": "datasets: hub, split, aggregate, audit",
    "sim": "the ManiSkill data factory (x86_64 only)",
    "vision": "cameras, calibration, tracking, deploy",
    "live": "every live sensor in one Rerun viewer",
    "depth": "wrist D405 depth gates",
    "audio": "piezo contact-mic analysis",
    "net": "edge/home network status",
    "node": "the node→role map and ssh dispatch",
}

_REGISTRY: list[Verb] = []


def verb(**kw) -> Callable[[Handler], Handler]:
    """Register a verb. ``@verb(noun=…, verb=…, tier=…, summary=…)``."""

    def deco(fn: Handler) -> Handler:
        _REGISTRY.append(Verb(run=fn, **kw))
        return fn

    return deco


def _available(v: "Verb") -> bool:
    """A verb exists only where its wrapped scripts do. The public export
    excludes fleet-orchestration scripts (plan Phase 5), and the CLI's help
    must be truthful on that tree: a verb whose every backing script is
    absent is hidden rather than advertised and broken. Native verbs (no
    wraps) are always available."""
    if not v.wraps:
        return True
    # Against the tree THIS code shipped in, not TATBOT_REPO: that env points
    # tools at another checkout's config, but a verb's backing scripts live
    # (or were export-excluded) beside the registry itself.
    tree = Path(__file__).resolve().parents[3]
    return all((tree / w).exists() for w in v.wraps)


def all_verbs() -> list[Verb]:
    from tatbot_cli import verbs  # noqa: F401  (import registers everything)

    order = {n: i for i, n in enumerate(NOUN_ORDER)}
    return sorted((v for v in _REGISTRY if _available(v)),
                  key=lambda v: (order.get(v.noun, 99), _REGISTRY.index(v)))


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
    env = os.environ.get("TATBOT_REPO")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]
