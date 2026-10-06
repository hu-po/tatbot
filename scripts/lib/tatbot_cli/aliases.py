"""Deprecated spellings translate once to canonical commands, never handlers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Alias:
    source: str
    canonical_name: str
    canonical_form: str
    noun: str
    command: str
    flag: str | None = None
    replacement_flag: str | None = None

    def schema(self):
        return {
            "source": self.source, "canonical_name": self.canonical_name,
            "canonical_form": self.canonical_form, "deprecated_since": "2026-09-05",
            "minimum_days": 30, "minimum_reviewed_releases": 2,
            "removal_requires": ["consumer_review", "release_note"],
        }

    @property
    def warning(self):
        return f"deprecated: tatbot {self.source}; use tatbot {self.canonical_form}"


ALIASES = (
    Alias("sim compile --sample N", "sim sample", "sim sample --count N", "sim", "compile", "--sample", "--count"),
    Alias("sim preview --reach", "sim reach", "sim reach", "sim", "preview", "--reach"),
    Alias("sim cinematic --viewer DIR…", "sim viewer", "sim viewer DIR…", "sim", "cinematic", "--viewer"),
    Alias("sim audit --samples DATASET", "sim samples", "sim samples DATASET", "sim", "audit", "--samples"),
    Alias("sim eval --policy-rollout -- ARGS", "sim eval policy", "sim eval policy -- ARGS", "sim", "eval", "--policy-rollout"),
    Alias("sim eval DATASET…", "sim eval dataset", "sim eval dataset DATASET…", "sim", "eval"),
)


def for_command(name):
    return [alias for alias in ALIASES if alias.canonical_name == name]


class AliasError(ValueError):
    """Malformed legacy syntax must not become a valid canonical invocation."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise AliasError(message)


def _legacy_args(alias, args, bare_globals):
    # Compatibility declarations preserve the old option ownership (including
    # last-wins valued flags). They never contain executable domain logic.
    parser = _Parser(add_help=False, allow_abbrev=False)
    parser.add_argument("--help", "-h", action="store_true")
    for flag in bare_globals:
        parser.add_argument(flag, action="store_true")
    if alias.command == "compile":
        parser.add_argument("placement", nargs="?")
    if alias.command == "eval":
        parser.add_argument("dataset", nargs="*")
    parser.add_argument(alias.flag, dest="selected",
                        **({"action": "store_true"} if alias.flag in {"--reach", "--policy-rollout"}
                           else {"nargs": "+"} if alias.flag == "--viewer" else {}))
    ns = parser.parse_args(args)
    if getattr(ns, "placement", None) or getattr(ns, "dataset", None):
        raise AliasError(f"tatbot {alias.source}: this mode cannot be combined with a positional operand")
    return ns


def translate(noun, tokens, bare_globals, *, explain=False):
    """Translate only CLI-owned mode tokens, preserving the -- tail verbatim."""
    if noun != "sim":
        return list(tokens), []
    stop = tokens.index("--") if "--" in tokens else len(tokens)
    head, tail = list(tokens[:stop]), list(tokens[stop:])
    index = next((i for i, token in enumerate(head) if token not in bare_globals), None)
    if index is None:
        return list(tokens), []
    command = head[index]
    args = head[index + 1:]
    meaningful = [t for t in args if t not in bare_globals]
    # Known new selectors win over the legacy dataset form, including paths
    # literally named dataset/policy (which need the explicit dataset syntax).
    if command == "eval" and meaningful and meaningful[0] in {"dataset", "policy"}:
        return list(tokens), []
    for alias in ALIASES:
        if alias.noun != noun or alias.command != command:
            continue
        if alias.flag is None:
            if not meaningful or meaningful == ["--help"] or meaningful == ["-h"]:
                continue
            converted = args
        else:
            if not any(t.partition("=")[0] == alias.flag for t in args):
                continue
            ns = None
            if not explain and "--explain" not in args:
                try:
                    ns = _legacy_args(alias, args, bare_globals)
                except argparse.ArgumentError as exc:
                    raise AliasError(str(exc)) from exc
            converted = []
            for token in args:
                key, equal, value = token.partition("=")
                if key != alias.flag:
                    converted.append(token)
                elif alias.replacement_flag:
                    converted.append(alias.replacement_flag + ("=" + value if equal else ""))
                elif equal:
                    # Valued selectors become positional arguments. Boolean
                    # selector=VALUE was never valid; leave it for usage refusal.
                    if alias.flag in {"--reach", "--policy-rollout"}:
                        return list(tokens), []
                    converted.append(value)
            if ns is not None and alias.flag in {"--viewer", "--samples"}:
                selected = ns.selected if isinstance(ns.selected, list) else [ns.selected]
                converted = [*selected, *(t for t in args if t in bare_globals)]
                if ns.help:
                    converted.append("--help")
        replacement = alias.canonical_name.split()[1:]
        return [*head[:index], *replacement, *converted, *tail], [alias.warning]
    return list(tokens), []
