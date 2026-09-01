"""Optional bridge to the run-logging layer in scripts/lib/tatbot_runlog.py.

The plugin lives in its own uv venv; the run-log module lives in the repo and
is deliberately stdlib-only and uninstalled, because four different
interpreters have to share it. So it is loaded by path when it is there, and
absence is a normal condition — a bench run, a hand-invoked lerobot-record, a
copy of this plugin without the repo — where every call here becomes a no-op.

Nothing in this module may raise into a caller. It sits in code that moves the
arm; a logging import error must never become a robot fault.
"""

import contextlib
import importlib.util
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_RUNLOG = None          # None = not looked yet, False = unavailable
_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE or None
    _MODULE = False
    path = os.environ.get("TATBOT_RUNLOG_PY")
    if not path:
        for parent in Path(__file__).resolve().parents:
            cand = parent / "scripts" / "lib" / "tatbot_runlog.py"
            if cand.is_file():
                path = str(cand)
                break
    if not path or not Path(path).is_file():
        return None
    with contextlib.suppress(Exception):
        spec = importlib.util.spec_from_file_location("tatbot_runlog", path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODULE = module
    return _MODULE or None


def runlog():
    """The run this process belongs to, or None when there is no run."""
    global _RUNLOG
    if _RUNLOG is not None:
        return _RUNLOG or None
    _RUNLOG = False
    if os.environ.get("TATBOT_RUNLOG", "1") == "0":
        return None
    module = _load_module()
    if module is None:
        return None
    with contextlib.suppress(Exception):
        _RUNLOG = module.attach() or False
    return _RUNLOG or None


def event(kind: str, **fields) -> None:
    run = runlog()
    if run is not None:
        with contextlib.suppress(Exception):
            run.event(kind, **fields)


def update(**fields) -> None:
    run = runlog()
    if run is not None:
        with contextlib.suppress(Exception):
            run.update(**fields)


def artifact(path, **fields) -> None:
    run = runlog()
    if run is not None:
        with contextlib.suppress(Exception):
            run.artifact(path, **fields)


def run_dir() -> Path | None:
    """The current run's directory, if a parent created one."""
    if os.environ.get("TATBOT_RUNLOG", "1") == "0":
        return None
    raw = os.environ.get("TATBOT_RUN_DIR")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None
