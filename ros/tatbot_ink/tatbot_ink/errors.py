"""Preparation refusals, before robot execution."""


class CompileError(ValueError):
    """The artwork or preparation request cannot produce a faithful robot program."""
