"""One declaration for argparse, machine schema, and completion (stdlib only)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Argument:
    names: tuple[str, ...]
    options: dict[str, Any]
    group: int | None = None

    def schema(self) -> dict:
        # Let argparse supply its documented action defaults and destination
        # rules, using exactly this definition; never serialize callable reprs.
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        action = parser.add_argument(*self.names, **self.options)
        convert = self.options.get("type", str)
        value_type = getattr(convert, "value_type", convert)
        return {
            "name": action.dest, "option_strings": action.option_strings,
            "positional": not bool(action.option_strings), "nargs": action.nargs,
            "type": "bool" if self.options.get("action") in ("store_true", "store_false") else getattr(value_type, "__name__", "string"),
            "default": action.default, "const": action.const,
            # A type may carry the choices itself when argparse cannot enforce
            # them directly — a phase name accepts compatibility aliases that
            # must be translated but never advertised, which `choices=` would
            # either reject or list.
            "choices": list(action.choices) if action.choices is not None
            else list(convert.choices) if hasattr(convert, "choices") else None,
            "range": list(convert.bounds) if hasattr(convert, "bounds") else None,
            "exclusive_bounds": list(getattr(convert, "exclusive_bounds", (False, False))) if hasattr(convert, "bounds") else None,
            "required": action.required, "conflict_group": self.group,
            "action": self.options.get("action", "store"), "help": action.help,
            "deprecated_options": ["--json"] if "--output" in self.names and "--json" in self.names else [],
        }


class _Group:
    def __init__(self, owner, index):
        self.owner, self.index = owner, index

    def add_argument(self, *names, **options):
        arg = Argument(names, options, self.index)
        self.owner.arguments.append(arg)
        return arg


@dataclass
class Arguments:
    """Recorder for existing declarative argument builders; no parsing here."""

    arguments: list[Argument] = field(default_factory=list)
    groups: list[dict] = field(default_factory=list)
    description: str | None = None

    @classmethod
    def capture(cls, builder):
        out = cls()
        if builder:
            builder(out)
        return out

    def add_argument(self, *names, **options):
        arg = Argument(names, options)
        self.arguments.append(arg)
        return arg

    def add_mutually_exclusive_group(self, **options):
        index = len(self.groups)
        self.groups.append(options)
        return _Group(self, index)

    def install(self, parser):
        groups = [parser.add_mutually_exclusive_group(**opts) for opts in self.groups]
        for arg in self.arguments:
            owner = parser if arg.group is None else groups[arg.group]
            owner.add_argument(*arg.names, **arg.options)
        if self.description:
            parser.description = (parser.description or "") + self.description

    def schema(self):
        return [arg.schema() for arg in self.arguments]
