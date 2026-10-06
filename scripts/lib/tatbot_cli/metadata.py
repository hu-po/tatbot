"""Command grouping and invocation-specific effects; never an authorization gate."""

from __future__ import annotations

from pathlib import Path

from tatbot_cli.registry import NOUNS

GROUPS = {group: tuple(n for n, (g, _) in NOUNS.items() if g == group)
          for group in ("operator", "development", "administration")}


def apply_metadata(v):
    if not v.group:
        v.group = next((g for g, nouns in GROUPS.items() if v.noun in nouns), "administration")
    if v.canonical_name is None:
        v.canonical_name = v.name
    # Help audience is a separate concept from export visibility: a public
    # command can be internal protocol, and a private one can be everyday work.
    if not v.audience:
        from tatbot_cli.audience import disposition
        v.audience, v.disposition = disposition(v.name)


def output_capability(v, ns=None, rest=()):
    return v.select_output(ns, rest) if v.select_output else v.output


def operation_effects(v, ns=None, rest=(), *, plan=None, remote=False):
    """Refine documented mixed operations; unknown backend flags stay conservative."""
    effects = set(v.effects)
    setup_effects = {"remote_exec"} if remote else set()
    if plan is not None and plan.argv and Path(plan.argv[0]).name in ("uv", "cargo"):
        setup_effects.update(("network", "write_files", "environment_setup"))
    if ns is None:
        return sorted(effects | setup_effects)
    if v.refine_effects:
        effects = set(v.refine_effects(effects, ns, rest))
    return sorted(effects | setup_effects)


def requirements(v, ns=None, rest=(), *, plan=None, remote=False):
    effects = operation_effects(v, ns, rest, plan=plan, remote=remote)
    return {"role": v.required_role(ns), "network": bool({"network", "remote_exec", "remote_write"} & set(effects)),
            "gpu": "gpu" in effects, "auto_hop": v.auto_hop, "sync": v.sync,
            "tty": v.tty,
            "environment": "native-stdlib" if v.native else "backend-owned; see plan argv",
            "dependency_setup": ("runtime may install dependencies, build artifacts, and write caches" if "environment_setup" in effects
                                 else "none" if v.native else "backend-owned; provisioning may use network and caches"),
            "capability_check": "at execution; planning does not probe interpreters"}
