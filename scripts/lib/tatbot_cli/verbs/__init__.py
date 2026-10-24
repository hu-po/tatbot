"""Importing this package registers every verb (one module per noun group).

The list below fixes the order of the generated docs. Any other module in
this directory whose name does not start with ``_`` is imported after it, in
sorted name order, and declares its own noun with ``registry.declare_noun``.
A module that only some checkouts carry therefore needs no line here: where
it is absent nothing is imported, and the list's order never moves.
"""

import importlib
import pkgutil
import sys

from tatbot_cli.verbs import (  # noqa: F401
    body,
    calib,
    core,
    deploy,
    drawingbot,
    hardware,
    ink,
    live,
    research,
    rig,
    rollout,
    ros,
    teleop,
    train,
    viewer,
    vision,
    web,
    work,
)

for _name in sorted(m.name for m in pkgutil.iter_modules(__path__)):
    if not _name.startswith("_") and f"{__name__}.{_name}" not in sys.modules:
        importlib.import_module(f"{__name__}.{_name}")
