#!/usr/bin/env python3
"""Run logging for every tatbot workflow — the writer, the index, and the CLI.

    scripts/tatbot-logs last rollout          # what happened most recently
    scripts/tatbot-logs show <run-id>         # one run, resolved to its node
    scripts/tatbot-logs tail <run-id> -f      # follow a live run

Every workflow writes one directory per run under the log root, holding the
full console output (more than the terminal showed), structured events, and
enough metadata to answer "what code, what versions, what hardware" months
later. Runs stay on the node that produced them; the CLI does the ssh.

The log root resolves TATBOT_LOG_ROOT > "log_root" in the runlog config
layers > the XDG state dir (~/.local/state/tatbot/logs). The private rig
sets "~/tatbot-logs" in its config/runlog.json; a fresh public clone with a
clean environment lands in XDG state and never touches ~/tatbot-logs.

STDLIB ONLY, ON PURPOSE. Four disjoint interpreters have to use this — the uv
plugin venv, ~/.venvs/tatbot-vision, bare system python3 with system cv2 for
scripts/vision/, and the training node's ~/il-train/.venv — and none can install into
the others. A single dependency-free file located by path is the only thing
that works in all four. Keep it that way.

Every entry point degrades to a no-op on failure: a broken logger must never
stop a run that moves the arm.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import logging.config
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# BEGIN LAYOUT — the run directory as written; scripts/tests/test_runlog.py
# fails when a run writes a file this block does not name.
# <log-root>/<workflow>/<run-id>/      (log root: see module docstring)
#   meta.json     run identity: git sha, versions, hardware, argv, exit code
#   console.log   byte-for-byte what the terminal showed
#   debug.log     Python DEBUG records (only when a shell owns console.log)
#   run.jsonl     structured events, append-only
#   KEEP          optional operator marker: this run is never pruned
#   ...           workflow artifacts: flight-*.csv, analysis.json,
#                 teleop.wxtl, poe/, rs/, session.rrd
#
# <log-root>/index.jsonl               one row per run state transition
# <log-root>/<workflow>/latest         symlink to the newest run
# <log-root>/<workflow>/.pruned/       meta.json of runs whose data was deleted
# <log-root>/.remote/<node>/           runs fetched from another node
# END LAYOUT

# run-id: <UTC>-<node>-<short>. UTC so lexicographic sort is chronological
# sort, which is what lets "newest run" be one rule instead of the three
# different ls/find hacks this replaced. The node is embedded so a run id is
# self-locating: an agent on one node can resolve a run from another.
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[a-z0-9][a-z0-9_.-]*-[0-9a-f]{4}$")
RUN_ID_FMT = "%Y%m%dT%H%M%SZ"

BANNER_TOKEN = "tatbot-run"
# AGENTS.md tells agents to grep for this; scripts/tests/test_runlog.py pins it.
BANNER_RE = re.compile(
    r"^=== tatbot-run (start|end|child) (?P<run_id>\S+)(?P<rest> .*)? ===$")

DEFAULT_CONFIG = {
    "schema_version": 1,
    # No home-dir default here (plan Phase 1): the private rig supplies
    # "~/tatbot-logs" via config/runlog.json; unset means the XDG state dir.
    "log_root": None,
    "stale_running_hours": 24,
    "prune_budget_s": 20,
    "default": {"class": "text", "keep_days": 90, "keep_runs": 500},
    "workflows": {
        "rollout": {"class": "text", "keep_days": 90, "keep_runs": 500},
        "rollout_async": {"class": "text", "keep_days": 90, "keep_runs": 500},
        "teleop": {"class": "text", "keep_days": 90, "keep_runs": 500},
        "record": {"class": "text", "keep_days": 90, "keep_runs": 200},
        "tune": {"class": "text", "keep_days": 30, "keep_runs": 100},
        "train": {"class": "text", "keep_days": 180, "keep_runs": 200},
        "vision": {"class": "media", "keep_days": 7, "keep_runs": 5},
        "calib": {"class": "media", "keep_days": 30, "keep_runs": 20},
        "selftest": {"class": "text", "keep_days": 1, "keep_runs": 5},
        # Pre-runlog vision evidence (hundreds of GB) lives under the same
        # workflow directories. `reindex --legacy` indexes it for visibility;
        # prune never deletes it (why_not_deletable refuses non-run-id names).
        "legacy": {"enabled": False},
    },
}

logger = logging.getLogger("tatbot.runlog")
_CURRENT: "RunLog | None" = None


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _utcnow()).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


_HOST_NODE: str | None = None


def _node() -> str:
    """TATBOT_NODE, else `hostname -s` mapped through a `hostname` alias in
    config/nodes.json, as tatbot_cli.nodes.this_node does (a node's hostname
    need not be its name). Resolved once per process: every index row asks."""
    global _HOST_NODE
    env = os.environ.get("TATBOT_NODE")
    if env:
        return env
    if _HOST_NODE is None:
        host = socket.gethostname().split(".")[0].lower()
        _HOST_NODE = next((name for name, rec in _fleet_map().items()
                           if isinstance(rec, dict) and str(rec.get("hostname", "")).lower() == host), host)
    return _HOST_NODE


def _run(cmd, cwd=None, timeout=5) -> str | None:
    """Best-effort command capture. Never raises, never blocks a run."""
    try:
        out = subprocess.run(cmd, cwd=cwd, timeout=timeout, check=False,
                             capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _repo_root(start: Path | None = None) -> Path | None:
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        # Session releases and the viewer overlay are git archives. Their
        # checked-in retention policy must resolve exactly as in a checkout;
        # requiring .git silently sends services to a different log index.
        archive_layout = (
            (parent / "scripts/lib/tatbot_runlog.py").is_file()
            and (parent / "scripts/tatbot").is_file()
            and (parent / "config/runlog.json").is_file()
        )
        if (parent / "AGENTS.md").is_file() and (
            (parent / ".git").exists() or archive_layout
        ):
            return parent
    return None


def load_config() -> dict:
    """Repo policy, then per-node overrides. Deep-merged, failure-tolerant."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    repo = _repo_root()
    for path in (
        (repo / "config" / "runlog.json") if repo is not None else None,
        Path("~/.config/tatbot/runlog.json").expanduser(),
    ):
        if not path or not path.is_file():
            continue
        try:
            layer = json.loads(path.read_text())
        except Exception as exc:
            print(f"runlog: ignoring unreadable {path}: {exc}", file=sys.stderr)
            continue
        for key, val in layer.items():
            if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                for k2, v2 in val.items():
                    if isinstance(v2, dict) and isinstance(cfg[key].get(k2), dict):
                        cfg[key][k2].update(v2)
                    else:
                        cfg[key][k2] = v2
            else:
                cfg[key] = val
    return cfg


def _fleet_module():
    """tatbot_cli.nodes — the `--on` hop's reading of config/nodes.json — or None.

    tatbot_cli is stdlib-only and sits next to this file, so the import costs
    nothing this file's four interpreters lack; it is deferred and guarded
    because the writer entry points must never depend on it."""
    try:
        from tatbot_cli import nodes
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from tatbot_cli import nodes
        except ImportError:
            return None
    return nodes


def _fleet_map() -> dict:
    """config/nodes.json as `tatbot --on` reads it; {} without a fleet."""
    repo = _repo_root()
    fleet = _fleet_module()
    if repo is None or fleet is None:
        return {}
    try:
        return fleet.load(repo)
    except (OSError, ValueError):
        return {}


def _sweep_nodes(cfg: dict, nmap: dict | None = None) -> tuple[list[str], list[str]]:
    """(nodes to sweep, retired nodes left out) for `--all-nodes`.

    env > runlog config > config/nodes.json keys. No hardcoded node names in
    code (plan Phase 1): the fleet is config, not a constant. A node with no
    roles is retired — nothing runs there and it is usually off — and is left
    out the way `tatbot status --fleet` leaves it out: a retired node cost
    every sweep a connect timeout and a non-zero exit until 2026-09-17.
    TATBOT_NODES is an explicit request and is swept as given."""
    env = os.environ.get("TATBOT_NODES")
    if env:
        return env.split(), []
    nmap = _fleet_map() if nmap is None else nmap
    wanted = cfg.get("nodes") or list(nmap)
    retired = [n for n in wanted if n in nmap and not nmap[n].get("roles")]
    return [n for n in wanted if n not in retired], retired


def log_root(cfg: dict | None = None) -> Path:
    root = os.environ.get("TATBOT_LOG_ROOT") or (cfg or load_config()).get("log_root")
    if root:
        return Path(root).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".local/state"
    return base / "tatbot" / "logs"


def retention_for(workflow: str, cfg: dict) -> dict:
    policy = dict(cfg["default"])
    policy.update(cfg["workflows"].get(workflow, {}))
    return policy


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

# Whole-environment dumps leak credentials (cameras.env, HF tokens), so the
# capture is an allowlist plus everything TATBOT_* except known-secret substrings.
ENV_ALLOW = ("DISPLAY", "CUDA_VISIBLE_DEVICES", "VIRTUAL_ENV", "HF_HOME",
             "CONDA_DEFAULT_ENV", "ROS_DOMAIN_ID", "LD_LIBRARY_PATH")

# Substrings that must never be persisted even if they happen to be TATBOT_*.
# Covers TATBOT_CAMERA_PASSWORD_*, *_TOKEN, *_SECRET, *_KEY, *_PASSWORD,
# *_CREDENTIALS — matches anywhere in the key, not just suffix (camera keys
# are TATBOT_CAMERA_PASSWORD_CAMERA{N}, which contains _PASSWORD_ mid-string).
_ENV_SECRET_SUBSTRINGS = ("_PASSWORD", "_TOKEN", "_SECRET", "_KEY", "_CREDENTIALS")

# Versions worth having when a run misbehaves. trossen-arm is first for a
# reason: on 2026-08-21 a stale checkout served rollouts on 1.8.8, the version
# whose own commit message records it wedging the follower controller twice in
# one minute, and nothing in the run said so.
VERSION_PKGS = ("trossen-arm", "lerobot", "lerobot_robot_trossen",
                "lerobot_robot_tatbot", "torch", "numpy", "transformers")


def _git_meta(repo: Path | None) -> dict:
    if repo is None:
        return {}
    g = ["git", "-C", str(repo)]
    meta: dict = {"repo": str(repo)}
    sha = _run(g + ["rev-parse", "HEAD"])
    if sha:
        meta["sha"] = sha
        meta["short"] = sha[:7]
    meta["branch"] = _run(g + ["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(g + ["status", "--porcelain"])
    if status is not None:
        files = [ln[3:] for ln in status.splitlines() if ln.strip()]
        meta["dirty"] = bool(files)
        meta["dirty_files"] = files[:40]
    upstream = _run(g + ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream:
        meta["upstream"] = upstream
        counts = _run(g + ["rev-list", "--left-right", "--count", f"{upstream}...HEAD"])
        if counts and len(counts.split()) == 2:
            behind, ahead = counts.split()
            meta["behind"], meta["ahead"] = int(behind), int(ahead)
    meta["committed_at"] = _run(g + ["log", "-1", "--format=%cI"])
    return {k: v for k, v in meta.items() if v is not None}


def _pkg_versions() -> dict:
    out = {}
    try:
        import importlib.metadata as md
    except Exception:
        return out
    for name in VERSION_PKGS:
        # A package simply not installed in this interpreter is the norm, not
        # an error: the four environments carry different subsets.
        with contextlib.suppress(Exception):
            out[name] = md.version(name)
    return out


def collect_meta(workflow: str, argv: list[str] | None = None,
                 extra: dict | None = None) -> dict:
    """Identity of this run. Budgeted, individually failure-tolerant, and it
    NEVER opens a hardware connection — the arm driver is exclusive (AGENTS.md),
    so driver/firmware versions are filled in later by the plugin that already
    holds the connection."""
    now = _utcnow()
    repo = _repo_root()
    meta = {
        "schema_version": SCHEMA_VERSION,
        "workflow": workflow,
        "status": "running",
        "exit_code": None,
        "started_at": _iso(now),
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "ended_at": None,
        "duration_s": None,
        "node": {
            "hostname": _node(),
            "user": os.environ.get("USER") or "",
            "pid": os.getpid(),
            "python": sys.version.split()[0],
        },
        "cmd": {
            "argv": list(argv if argv is not None else sys.argv),
            "cwd": os.getcwd(),
        },
        "env": {k: os.environ.get(k) for k in ENV_ALLOW},
        "versions": {"python_packages": _pkg_versions()},
        "artifacts": [],
        "counters": {"warn": 0, "error": 0, "estop": 0, "overforce": 0},
        "summary": None,
        "keep": False,
    }
    meta["env"].update({k: v for k, v in os.environ.items()
                        if k.startswith("TATBOT_") and not any(s in k for s in _ENV_SECRET_SUBSTRINGS)})
    with contextlib.suppress(Exception):
        meta["node"]["kernel"] = os.uname().release
    with contextlib.suppress(Exception):
        meta["git"] = _git_meta(repo)
    if extra:
        # Scrub any secret-substring keys that arrived via extra["env"].
        try:
            if isinstance(extra.get("env"), dict):
                extra = dict(extra)
                extra["env"] = {k: v for k, v in extra["env"].items()
                                if not any(s in k for s in _ENV_SECRET_SUBSTRINGS)}
        except Exception:
            pass
        _deep_update(meta, extra)
    return meta


def _deep_update(base: dict, new: dict) -> dict:
    for key, val in new.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], val)
        else:
            base[key] = val
    return base


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


@contextlib.contextmanager
def _flocked(path: Path, flags: int):
    """An fd on the file `path` names, holding its exclusive flock.

    `logs compact` replaces the index under this lock, so a caller that opened
    the old file and then waited for the lock reopens instead of writing into
    an unlinked inode. The third attempt keeps whatever it opened.
    """
    for attempt in range(3):
        fd = os.open(path, flags, 0o644)
        try:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_EX)
            if attempt < 2 and not _names(path, fd):
                continue
            yield fd
            return
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _names(path: Path, fd: int) -> bool:
    try:
        return os.path.samestat(os.fstat(fd), os.stat(path))
    except OSError:
        return False


def _append_line(path: Path, obj: dict) -> None:
    """One kernel-atomic append. Concurrent workflows share these files, so the
    line must fit in a single write and the lock is held only for the write."""
    line = json.dumps(obj, default=str)
    if len(line) > 4000:
        obj = dict(obj)
        for key, val in list(obj.items()):
            if isinstance(val, str) and len(val) > 500:
                obj[key] = val[:500]
        obj["_truncated"] = True
        line = json.dumps(obj, default=str)
    data = (line + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _flocked(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND) as fd:
        os.write(fd, data)


# --------------------------------------------------------------------------
# logging attachment
# --------------------------------------------------------------------------

# Chatty third-party loggers at DEBUG turn a useful file into an unusable one.
_NOISY = ("urllib3", "PIL", "matplotlib", "numba", "asyncio", "fsspec",
          "huggingface_hub", "botocore", "filelock", "h5py", "git")


class _RunFileHandler(logging.FileHandler):
    """A file handler that survives logging.basicConfig(force=True).

    lerobot reconfigures the root logger when imported — tune.py has carried a
    comment about that since it silently swallowed every INFO line. The worse
    variant is force=True, which removes AND CLOSES the existing root handlers;
    a closed FileHandler raises ValueError on the next record and the run's file
    log ends there. So close() detaches instead, and only finalize() really
    closes.
    """

    _tatbot_runlog = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._final = False

    def close(self) -> None:
        if self._final:
            super().close()

    def final_close(self) -> None:
        self._final = True
        with contextlib.suppress(Exception):
            self.close()


def _ensure_attached(handler: _RunFileHandler, console_level: int) -> None:
    root = logging.getLogger()
    if handler not in root.handlers:
        root.addHandler(handler)
    # DEBUG has to reach the file, so the root threshold drops — but the
    # terminal keeps its own level, otherwise every run turns into a firehose.
    root.setLevel(min(root.level or logging.WARNING, handler.level))
    for other in root.handlers:
        if other is not handler and other.level < console_level:
            other.setLevel(console_level)
    for name in _NOISY:
        logging.getLogger(name).setLevel(max(logging.INFO, handler.level))


def _patch_reconfigurers(handler: _RunFileHandler, console_level: int) -> None:
    for mod, name in ((logging, "basicConfig"),
                      (logging.config, "dictConfig"),
                      (logging.config, "fileConfig")):
        orig = getattr(mod, name, None)
        if orig is None or getattr(orig, "_tatbot_wrapped", False):
            continue

        def wrapper(*args, _orig=orig, **kwargs):
            try:
                return _orig(*args, **kwargs)
            finally:
                with contextlib.suppress(Exception):
                    _ensure_attached(handler, console_level)

        setattr(wrapper, "_tatbot_wrapped", True)  # noqa: B010
        setattr(mod, name, wrapper)


# --------------------------------------------------------------------------
# RunLog
# --------------------------------------------------------------------------

class RunLog:
    """One run's directory. Every method is failure-tolerant by contract."""

    def __init__(self, run_dir: Path, workflow: str, run_id: str,
                 owns_console: bool = True):
        self.dir = run_dir
        self.workflow = workflow
        self.run_id = run_id
        self.started_at = time.time()
        self.owns_console = owns_console
        self._seq = 0
        self._handler: _RunFileHandler | None = None
        self._final = False

    # -- paths ------------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.dir / name

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    def read_meta(self) -> dict:
        try:
            return json.loads(self.meta_path.read_text())
        except Exception:
            return {}

    # -- structured events ------------------------------------------------
    def event(self, kind: str, **fields) -> None:
        try:
            self._seq += 1
            row = {"t": _iso(), "mono": round(time.time() - self.started_at, 3),
                   "seq": self._seq, "pid": os.getpid(), "kind": kind}
            row.update(fields)
            _append_line(self.dir / "run.jsonl", row)
            if kind in ("warn", "error", "estop", "overforce"):
                self._bump(kind if kind != "warn" else "warn")
            if self._handler is not None:
                _ensure_attached(self._handler, logging.INFO)
        except Exception:
            pass

    def _bump(self, counter: str) -> None:
        with contextlib.suppress(Exception):
            meta = self.read_meta()
            counters = meta.setdefault("counters", {})
            counters[counter] = counters.get(counter, 0) + 1
            _atomic_write_json(self.meta_path, meta)

    def artifact(self, path, **fields) -> Path:
        p = Path(path)
        with contextlib.suppress(Exception):
            entry: dict[str, Any] = {"name": p.name, "path": str(p)}
            if p.is_file():
                entry["bytes"] = p.stat().st_size
            entry.update(fields)
            meta = self.read_meta()
            meta.setdefault("artifacts", []).append(entry)
            _atomic_write_json(self.meta_path, meta)
            self.event("artifact", **entry)
        return p

    def update(self, **fields) -> None:
        with contextlib.suppress(Exception):
            meta = self.read_meta()
            _deep_update(meta, fields)
            _atomic_write_json(self.meta_path, meta)

    # -- logging ----------------------------------------------------------
    def attach_logging(self, console_level=logging.INFO,
                       file_level=logging.DEBUG) -> None:
        """Send DEBUG-and-below to a file while the terminal keeps its level.

        When a shell already tees the terminal into console.log, writing there
        from Python too would duplicate every line from two independently
        buffered writers, interleaved. So the shell-owned case gets debug.log
        and console.log stays exactly what the terminal showed.
        """
        try:
            name = "console.log" if self.owns_console else "debug.log"
            handler = _RunFileHandler(self.dir / name, encoding="utf-8")
            handler.setLevel(file_level)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s [%(filename)s:%(lineno)d] %(message)s"))
            self._handler = handler
            _ensure_attached(handler, console_level)
            _patch_reconfigurers(handler, console_level)
            if self.owns_console and not any(
                    isinstance(h, logging.StreamHandler)
                    and not isinstance(h, logging.FileHandler)
                    for h in logging.getLogger().handlers):
                stream = logging.StreamHandler()
                stream.setLevel(console_level)
                stream.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                logging.getLogger().addHandler(stream)
        except Exception as exc:
            print(f"runlog: file logging unavailable: {exc}", file=sys.stderr)

    # -- lifecycle --------------------------------------------------------
    def finalize(self, exit_code: int = 0, status: str | None = None) -> None:
        if self._final:
            return
        # Shell cleanup and an SSH-disconnect watchdog may race to close the
        # same run. A terminal record is immutable: do not append another
        # run.end or overwrite the original exit status from a new process.
        persisted = self.read_meta()
        if persisted.get("status") not in (None, "running"):
            self._final = True
            return
        self._final = True
        try:
            ended = _utcnow()
            duration = round(time.time() - self.started_at, 3)
            if status is None:
                status = {0: "ok", 130: "interrupted", 124: "timeout"}.get(
                    exit_code, "fail")
            meta = self.read_meta()
            meta.update({"status": status, "exit_code": exit_code,
                         "ended_at": _iso(ended), "duration_s": duration})
            console = self.dir / "console.log"
            if console.is_file():
                counters = meta.setdefault("counters", {})
                counters["console_bytes"] = console.stat().st_size
                # Native C++ workflows cannot attach the Python logger. Their
                # deterministic transition line still makes the run-level
                # counter useful; Python workflows emit structured events and
                # therefore already have a nonzero counter here.
                if not counters.get("estop"):
                    with contextlib.suppress(Exception), console.open(errors="replace") as stream:
                        counters["estop"] = sum(line.startswith("E-STOP:") for line in stream)
            _atomic_write_json(self.meta_path, meta)
            self.event("run.end", status=status, exit_code=exit_code,
                       duration_s=duration)
            index_append(self, meta, terminal=True)
        except Exception:
            pass
        finally:
            if self._handler is not None:
                with contextlib.suppress(Exception):
                    logging.getLogger().removeHandler(self._handler)
                    self._handler.final_close()
            global _CURRENT
            if _CURRENT is self:
                _CURRENT = None

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finalize(0 if exc_type is None else 1,
                      status="ok" if exc_type is None else
                      ("interrupted" if exc_type is KeyboardInterrupt else "fail"))
        return False


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

def index_path(cfg: dict | None = None) -> Path:
    return log_root(cfg) / "index.jsonl"


def index_append(run: "RunLog", meta: dict, terminal: bool = False,
                 cfg: dict | None = None) -> None:
    """One row per state transition; readers take the last row per run_id.

    Append-only is what makes this safe for concurrent workflows and honest
    about crashed ones: a 'running' row with no terminal row is evidence, where
    today a hard kill leaves none at all.
    """
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.run_id,
        "workflow": run.workflow,
        "node": meta.get("node", {}).get("hostname", _node()),
        "dir": str(run.dir),
        "status": meta.get("status"),
        "t": _iso(),
    }
    if terminal:
        row.update({"exit_code": meta.get("exit_code"),
                    "ended_at": meta.get("ended_at"),
                    "duration_s": meta.get("duration_s"),
                    "counters": meta.get("counters", {}),
                    "summary": meta.get("summary"),
                    "artifacts": [a.get("name") for a in meta.get("artifacts", [])]})
    else:
        row.update({"started_at": meta.get("started_at"),
                    "pid": meta.get("node", {}).get("pid"),
                    "git": meta.get("git", {}).get("short"),
                    "dirty": meta.get("git", {}).get("dirty"),
                    "key": meta.get("key", {})})
    with contextlib.suppress(Exception):
        _append_line(index_path(cfg), row)


def index_rows(cfg: dict | None = None, limit: int | None = None, *, strict: bool = False) -> list[dict]:
    path = index_path(cfg)
    if not strict and not path.is_file():
        return []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []
    except Exception:
        if strict:
            raise
        return []
    if limit:
        lines = lines[-limit * 4:]
    rows = []
    for line in lines:
        line = line.strip()
        if line:
            if strict:
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("run_id"):
                    raise ValueError(f"invalid run-index row in {path}")
                rows.append(row)
                continue
            with contextlib.suppress(Exception):
                rows.append(json.loads(line))
    return rows


def index_runs(cfg: dict | None = None, workflow: str | None = None, *, strict: bool = False) -> list[dict]:
    """Collapse the transition rows into one record per run, newest last."""
    merged: dict[str, dict] = {}
    for row in index_rows(cfg, strict=strict):
        rid = row.get("run_id")
        if not rid:
            continue
        merged.setdefault(rid, {}).update(row)
    runs = list(merged.values())
    if workflow:
        runs = [r for r in runs if r.get("workflow", "").startswith(workflow)]
    runs.sort(key=lambda r: r.get("run_id", ""))
    return runs


def _pid_alive(pid, *, strict: bool = False) -> bool:
    try:
        if strict and (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0):
            raise ValueError("run pid must be a positive integer")
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        if strict:
            raise
        return False


def resolve_status(row: dict, *, node: str | None = None, strict: bool = False) -> str:
    """A 'running' row only means running if it is this host and the pid lives."""
    status = row.get("status") or "unknown"
    if status != "running":
        return status
    if row.get("data_present") is False:
        return "pruned"
    if row.get("node") != (node or _node()):
        return "running (elsewhere)"
    return "running" if _pid_alive(row.get("pid"), strict=strict) else "crashed (no finalize)"


# --------------------------------------------------------------------------
# retention
# --------------------------------------------------------------------------

def _read_meta(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / "meta.json").read_text())
    except Exception:
        return None


def _age_days(meta: dict) -> float:
    stamp = meta.get("ended_at") or meta.get("started_at")
    if not stamp:
        return 0.0
    with contextlib.suppress(Exception):
        dt = datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc)
        return (_utcnow() - dt).total_seconds() / 86400.0
    return 0.0


def why_not_deletable(run_dir: Path, root: Path, workflow: str,
                      cfg: dict) -> str | None:
    """None if this directory may be deleted, else the reason it may not.

    This function is the only thing standing between the pruner and every log
    on the box, so it refuses on anything it does not positively recognise —
    including an unreadable meta.json.
    """
    if run_dir.is_symlink():
        return "symlink"
    if not run_dir.is_dir():
        return "not a directory"
    if run_dir.parent.resolve() != (root / workflow).resolve():
        return "outside the workflow directory"
    if not RUN_ID_RE.match(run_dir.name):
        return "name is not a run id"
    if run_dir.name == os.environ.get("TATBOT_RUN_ID"):
        return "the current run"
    if (run_dir / "KEEP").exists():
        return "KEEP marker"
    meta = _read_meta(run_dir)
    if meta is None:
        return "meta.json unreadable"
    if meta.get("keep"):
        return "meta.keep"
    if meta.get("status") == "running":
        same_host = meta.get("node", {}).get("hostname") == _node()
        if same_host and _pid_alive(meta.get("node", {}).get("pid")):
            return "still running here"
        hours = (_age_days(meta) * 24)
        if hours < cfg.get("stale_running_hours", 24):
            return "recently started"
    return None


def prune(workflow: str, cfg: dict | None = None, dry_run: bool = True,
          budget_s: float | None = None, verbose: bool = True) -> list[dict]:
    """Delete the oldest runs of one workflow past its retention policy."""
    cfg = cfg or load_config()
    policy = retention_for(workflow, cfg)
    removed: list[dict] = []
    if policy.get("enabled") is False:
        return removed
    root = log_root(cfg)
    wf_dir = root / workflow
    if not wf_dir.is_dir():
        return removed
    budget = budget_s if budget_s is not None else cfg.get("prune_budget_s", 20)
    deadline = time.time() + budget

    candidates = sorted(p for p in wf_dir.iterdir()
                        if p.is_dir() and RUN_ID_RE.match(p.name))
    keep_runs = policy.get("keep_runs")
    keep_days = policy.get("keep_days")
    # Keepers are the UNION of both rules, never the intersection: a run is
    # only eligible if it is both beyond the count AND beyond the age.
    protected = set()
    if keep_runs:
        protected.update(p.name for p in candidates[-int(keep_runs):])

    for run_dir in candidates:
        if time.time() > deadline:
            if verbose:
                print(f"prune: budget reached, {len(candidates)} candidates left "
                      "for the next run", file=sys.stderr)
            break
        if run_dir.name in protected:
            continue
        meta = _read_meta(run_dir)
        if keep_days and meta is not None and _age_days(meta) < float(keep_days):
            continue
        reason = why_not_deletable(run_dir, root, workflow, cfg)
        if reason:
            continue
        size = _dir_size(run_dir)
        entry = {"run_id": run_dir.name, "workflow": workflow, "bytes": size,
                 "status": (meta or {}).get("status"),
                 "ended_at": (meta or {}).get("ended_at")}
        if dry_run:
            removed.append(entry)
            if verbose:
                print(f"prune: WOULD remove {run_dir.name} "
                      f"({workflow}, {_human(size)})", file=sys.stderr)
            continue
        # Record first, delete second: a crash mid-rmtree then leaves a
        # half-empty directory the next prune finishes, with the record already
        # durable. An empty dir with no index row would read as "no logs".
        with contextlib.suppress(Exception):
            pruned_dir = wf_dir / ".pruned"
            pruned_dir.mkdir(parents=True, exist_ok=True)
            if meta is not None:
                _atomic_write_json(pruned_dir / f"{run_dir.name}.meta.json", meta)
            _append_line(index_path(cfg), {
                "schema_version": SCHEMA_VERSION, "run_id": run_dir.name,
                "workflow": workflow, "node": _node(), "status": "pruned",
                "pruned_at": _iso(), "data_present": False, "bytes_freed": size})
        try:
            shutil.rmtree(run_dir)
        except Exception as exc:
            print(f"prune: failed to remove {run_dir}: {exc}", file=sys.stderr)
            continue
        removed.append(entry)
        if verbose:
            # stderr, never stdout: `begin` runs this and runlog.sh captures its stdout as
            # RUN_DIR (a 60-run prune once became a 6 KB "log path", exit 130).
            print(f"prune: removed {run_dir.name} ({workflow}, {_human(size)}, "
                  f"ended {entry['ended_at']}, status {entry['status']})", file=sys.stderr)
    return removed


def _dir_size(path: Path) -> int:
    total = 0
    with contextlib.suppress(Exception):
        for root_dir, _dirs, files in os.walk(path):
            for name in files:
                with contextlib.suppress(OSError):
                    total += (Path(root_dir) / name).stat().st_size
    return total


def _human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def _mint_run_id(node: str | None = None) -> str:
    return (f"{_utcnow().strftime(RUN_ID_FMT)}-{node or _node()}-"
            f"{os.urandom(2).hex()}")


def banner(kind: str, run_id: str, rest: str = "") -> str:
    return f"=== {BANNER_TOKEN} {kind} {run_id}{(' ' + rest) if rest else ''} ==="


def _emit(line: str) -> None:
    """Banners go to stderr: unbuffered, and it survives | tee and | head, so
    the run id is in whatever a human ends up pasting."""
    with contextlib.suppress(Exception):
        sys.stderr.write(line + "\n")
        sys.stderr.flush()


def init(workflow: str, *, meta: dict | None = None,
         argv: list[str] | None = None, root: Path | None = None,
         attach_logging: bool = True, console_level: int = logging.INFO,
         file_level: int = logging.DEBUG, prune_first: bool = True,
         owns_console: bool | None = None, emit_banner: bool = True) -> RunLog:
    """Create a run directory and become the current run."""
    global _CURRENT
    cfg = load_config()
    base = Path(root).expanduser() if root else log_root(cfg)
    run_id = _mint_run_id()
    run_dir = base / workflow / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if owns_console is None:
        owns_console = os.environ.get("TATBOT_RUN_CONSOLE") != "shell"
    run = RunLog(run_dir, workflow, run_id, owns_console=owns_console)

    payload = collect_meta(workflow, argv=argv, extra=meta)
    payload["run_id"] = run_id
    payload["retention"] = retention_for(workflow, cfg)
    _atomic_write_json(run.meta_path, payload)
    run.event("run.start", workflow=workflow, argv=payload["cmd"]["argv"])
    index_append(run, payload, terminal=False, cfg=cfg)
    _update_latest(base / workflow, run_dir)

    if attach_logging:
        run.attach_logging(console_level=console_level, file_level=file_level)
    if emit_banner:
        _emit(banner("start", run_id, f"pid={payload['node']['pid']} log={run_dir}"))
    if prune_first:
        # Before any hardware connection, and synchronous on purpose: the
        # deletions then land in THIS run's console.log where an agent will
        # find them, and no orphaned rm -rf outlives a killed run.
        with contextlib.suppress(Exception):
            prune(workflow, cfg=cfg, dry_run=False, verbose=True)
    _CURRENT = run
    return run


def attach(run_dir=None, *, attach_logging: bool = True,
           console_level: int = logging.INFO,
           file_level: int = logging.DEBUG) -> RunLog | None:
    """Join a run a parent process already created. None if there is none.

    Absence is a NORMAL condition — a bench test, a hand-run lerobot-record, a
    vision script run directly. Callers must keep working when this returns
    None.
    """
    global _CURRENT
    if _CURRENT is not None:
        return _CURRENT
    target = run_dir or os.environ.get("TATBOT_RUN_DIR")
    if not target or os.environ.get("TATBOT_RUNLOG", "1") == "0":
        return None
    path = Path(target).expanduser()
    if not path.is_dir():
        return None
    run_id = os.environ.get("TATBOT_RUN_ID") or path.name
    workflow = os.environ.get("TATBOT_RUN_WORKFLOW") or path.parent.name
    owns_console = os.environ.get("TATBOT_RUN_CONSOLE") != "shell"
    run = RunLog(path, workflow, run_id, owns_console=owns_console)
    if attach_logging:
        run.attach_logging(console_level=console_level, file_level=file_level)
    _CURRENT = run
    return run


def current() -> RunLog | None:
    return _CURRENT


def _update_latest(workflow_dir: Path, run_dir: Path) -> None:
    """Atomic 'latest' symlink — replaces three different ls/find hacks."""
    with contextlib.suppress(Exception):
        link = workflow_dir / "latest"
        tmp = workflow_dir / f".latest.{os.getpid()}"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(run_dir.name)
        os.replace(tmp, link)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
       "-o", "StrictHostKeyChecking=accept-new"]

HELP = """tatbot-logs — find and read the full log of any tatbot run.

  tatbot-logs last rollout              the most recent run of a workflow
  tatbot-logs show <run-id>             one run: meta, errors, console tail
  tatbot-logs tail <run-id> -f          follow a run that is still going
  tatbot-logs list [workflow]           recent runs, newest last
  tatbot-logs fetch <run-id>            copy a remote run here (skips media)
  tatbot-logs count rollout_async       how many launches the index holds (reconcile before/after)

A run id carries its node, so it is self-locating:

  20260821T164233Z-<node>-a3f1
  └ UTC start        └ node └ unique

  tatbot-logs show 20260821T164233Z-<node>-a3f1  # ssh's to that node by itself
  tatbot-logs list --all-nodes                   # sweep every known node

Runs live in <log root>/<workflow>/<run-id>/ on the node that produced them
(`root` prints the log root). console.log holds more than the terminal showed.
`list` shows which workflows this node holds runs for.

Other commands: du, prune [--yes], reindex [--legacy], compact, selftest.
`count <workflow> --expect N --before M` exits 1 unless exactly N launches were
added since the snapshot M: reconcile run COUNT before calling any launch
uncommanded (the 2026-08-24 lesson).
"""


def _node_of(run_id: str) -> str | None:
    m = re.match(r"^\d{8}T\d{6}Z-([a-z0-9][a-z0-9_.-]*)-[0-9a-f]{4}$", run_id)
    return m.group(1) if m else None


def _remote_argv(node: str, args: list[str], nmap: dict | None = None) -> list[str]:
    """The ssh command that runs this same CLI on `node`.

    The target and the checkout come from config/nodes.json through
    tatbot_cli.nodes, exactly as `tatbot --on <node>` resolves them; a node
    the map does not know is dialed by name and assumed at the public repo's
    conventional path. It used to glob `$(cd ~/tatbot* && pwd)`, which is
    empty wherever the log root or a second checkout sits beside the checkout
    (`cd` refuses two arguments), so two nodes answered
    `can't open file '/scripts/lib/tatbot_runlog.py'` (2026-09-17)."""
    nmap = _fleet_map() if nmap is None else nmap
    fleet = _fleet_module()
    if fleet is not None and node in nmap:
        target = fleet.ssh_target(nmap, node) or node
        checkout = fleet.remote_checkout(nmap, node)
    else:
        target, checkout = node, "$HOME/tatbot"
    script = f"{checkout}/scripts/lib/tatbot_runlog.py"
    return SSH + [target, f"python3 {script} " + " ".join(map(_q, args))]


def _remote(node: str, args: list[str]) -> tuple[int, str]:
    """Run this same CLI on another node; fall back to reading its index."""
    try:
        cmd = _remote_argv(node, args)
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return out.returncode, (out.stdout or out.stderr)
    except Exception as exc:
        return 255, f"{node}: unreachable ({exc})"


def _q(s: str) -> str:
    return "'" + str(s).replace("'", "'\\''") + "'"


def _fmt_row(row: dict) -> str:
    status = resolve_status(row)
    mark = {"ok": " ", "running": "*", "pruned": "-"}.get(status, "!")
    dur = row.get("duration_s")
    dur_s = f"{dur:6.1f}s" if isinstance(dur, (int, float)) else "      -"
    summary = row.get("summary") or ""
    return (f" {mark} {row.get('run_id',''):<30} {row.get('workflow',''):<14} "
            f"{status:<22} {dur_s}  {summary}")[:200]


def _list_remote(node: str, args) -> tuple[int, list[dict] | None, str]:
    """One other node's listing: (exit code, its rows under --json, its text)."""
    code, out = _remote(node, ["list"] + ([args.workflow] if args.workflow else [])
                        + ["-n", str(args.n)] + (["--status", args.status] if args.status else [])
                        + (["--json"] if args.json else []))
    if not args.json:
        return code, None, out
    try:
        rows = json.loads(out) if code == 0 else None
    except ValueError:
        rows = None
    if not isinstance(rows, list):
        return code or 1, None, out  # it answered, but not with a listing
    return code, rows, out


def _sweep(args) -> tuple[int, list[dict]]:
    """The other nodes' listings: printed as sections, or (--json) their rows
    for one array, each row naming its node."""
    rc, swept = 0, []
    for node in args.nodes:
        if node == _node():
            continue
        code, rows, out = _list_remote(node, args)
        if rows is not None:
            swept.extend(rows)
        elif not args.json:
            print(f"--- {node} ---")
            print(out.rstrip() if out.strip() else "(no runs)")
        # A node that did not answer makes this listing incomplete. Say so on
        # stderr and carry the first failing code out, so a search for a run
        # cannot read one node's silence as "that run does not exist".
        if code != 0:
            print(f"{node}: listing failed (exit {code})", file=sys.stderr)
            rc = rc or code
    note = "retired: no roles in config/nodes.json; not swept — `logs show <run-id>` still resolves its runs"
    for node in getattr(args, "retired", None) or []:
        print(f"{node}: {note}", file=sys.stderr) if args.json else print(f"--- {node} ---\n({note})")
    return rc, swept


def _cmd_list(args) -> int:
    rc, swept = _sweep(args) if args.nodes else (0, [])
    if args.nodes and not args.json:
        print(f"--- {_node()} (local) ---")
    runs = index_runs(workflow=args.workflow)
    if args.status:
        runs = [r for r in runs if resolve_status(r).startswith(args.status)]
    runs = runs[-args.n:]
    if args.json:
        print(json.dumps(swept + runs, indent=2, default=str))
        return rc
    if not runs:
        print("(no runs indexed)")
        return rc
    for row in runs:
        print(_fmt_row(row))
    return rc


def _cmd_last(args) -> int:
    runs = index_runs(workflow=args.workflow)
    if not runs:
        print(f"no runs indexed for {args.workflow or 'any workflow'} on {_node()}",
              file=sys.stderr)
        return 1
    row = runs[-1]
    if args.json:
        print(json.dumps(row, indent=2, default=str))
        return 0
    run_id = row.get("run_id")
    if not isinstance(run_id, str):
        print(f"no runs indexed for {args.workflow or 'any workflow'} on {_node()}",
              file=sys.stderr)
        return 1
    return _show_run(run_id, tail=args.tail)


def _find_local(run_id: str) -> Path | None:
    root = log_root()
    if not root.is_dir():
        return None
    for wf_dir in root.iterdir():
        if not wf_dir.is_dir():
            continue
        exact = wf_dir / run_id
        if exact.is_dir():
            return exact
        for cand in wf_dir.glob(f"{run_id}*"):
            if cand.is_dir() and RUN_ID_RE.match(cand.name):
                return cand
    remote = root / ".remote"
    if remote.is_dir():
        for cand in remote.glob(f"*/*/{run_id}*"):
            if cand.is_dir():
                return cand
    return None


def _show_run(run_id: str, tail: int = 40) -> int:
    run_dir = _find_local(run_id)
    if run_dir is None:
        node = _node_of(run_id)
        if node and node != _node():
            code, out = _remote(node, ["show", run_id, "--tail", str(tail)])
            print(out.rstrip())
            return code
        row = next((r for r in index_runs() if r.get("run_id") == run_id), None)
        if row and row.get("status") == "pruned":
            print(f"{run_id}: pruned {row.get('pruned_at')} — data deleted by "
                  f"retention, record kept")
            meta = (log_root() / row.get("workflow", "") / ".pruned"
                    / f"{run_id}.meta.json")
            if meta.is_file():
                print(meta.read_text())
            return 0
        print(f"{run_id}: not found on {_node()}", file=sys.stderr)
        return 1
    meta = _read_meta(run_dir) or {}
    print(f"run     {run_dir.name}")
    print(f"dir     {run_dir}")
    print(f"status  {meta.get('status')} exit={meta.get('exit_code')} "
          f"duration={meta.get('duration_s')}s")
    git = meta.get("git", {})
    print(f"git     {git.get('short')} {git.get('branch')} "
          f"dirty={git.get('dirty')} behind={git.get('behind')}")
    versions = meta.get("versions", {}).get("python_packages", {})
    if versions:
        print("versions " + " ".join(f"{k}={v}" for k, v in versions.items()))
    print(f"argv    {' '.join(meta.get('cmd', {}).get('argv', []))}")
    if meta.get("summary"):
        print(f"summary {meta['summary']}")
    print("files   " + ", ".join(sorted(p.name for p in run_dir.iterdir())))
    console = run_dir / "console.log"
    if console.is_file() and tail:
        print(f"\n--- console.log (last {tail}) ---")
        lines = console.read_text(errors="replace").splitlines()
        print("\n".join(lines[-tail:]))
    return 0


def _cmd_show(args) -> int:
    return _show_run(args.run_id, tail=args.tail)


def _cmd_tail(args) -> int:
    run_dir = _find_local(args.run_id)
    if run_dir is None:
        node = _node_of(args.run_id)
        if node and node != _node():
            print(f"run is on {node}; ssh {node} and tail there, or "
                  f"'tatbot-logs fetch {args.run_id}'", file=sys.stderr)
            return 1
        print(f"{args.run_id}: not found", file=sys.stderr)
        return 1
    target = run_dir / args.file
    if not target.is_file():
        print(f"{target} does not exist", file=sys.stderr)
        return 1
    cmd = ["tail", "-n", str(args.n)] + (["-f"] if args.follow else []) + [str(target)]
    return subprocess.call(cmd)


def _cmd_fetch(args) -> int:
    node = args.node or _node_of(args.run_id)
    if not node or node == _node():
        print("run is already local", file=sys.stderr)
        return 1
    dest = log_root() / ".remote" / node
    dest.mkdir(parents=True, exist_ok=True)
    # Media is excluded by default: a vision run is tens of GB and an agent
    # wants meta/console/events, not frames. --all opts in.
    excludes = [] if args.all else [
        "--exclude=poe/", "--exclude=rs/", "--exclude=*.rrd", "--exclude=*.z16",
        "--exclude=*.jpg", "--exclude=*.png", "--exclude=*.mp4"]
    # Dial the node the way `list`/`show` do (its config/nodes.json ssh target),
    # and read its runs where it keeps them, not where this node does.
    nmap, fleet = _fleet_map(), _fleet_module()
    target = (fleet.ssh_target(nmap, node) if fleet is not None else None) or node
    code, out = _remote(node, ["root"])
    root = out.strip().splitlines()[-1] if code == 0 and out.strip() else "~/tatbot-logs"
    src = f"{target}:{root}/*/{args.run_id}*/"
    cmd = (["rsync", "-a", "--partial", "--info=progress2"] + excludes
           + ["-e", " ".join(SSH[:1] + SSH[2:])]
           + [src, str(dest / args.run_id) + "/"])
    print(" ".join(cmd))
    return subprocess.call(cmd)


def count_launches(workflow: str, cfg: dict | None = None) -> int:
    """How many times `workflow` was launched: its runs with a start row in the index.

    Every launch appends exactly one `running` row carrying `started_at`, and
    `logs compact` keeps that field in the run's merged row, so this is the
    launch count even for runs that never finalized, before or after a compact.
    Exact workflow match: `rollout` and `rollout_async` are different launchers
    and must not be summed."""
    return len({row["run_id"] for row in index_rows(cfg)
                if row.get("workflow") == workflow and row.get("run_id") and row.get("started_at")})


def _cmd_count(args) -> int:
    n = count_launches(args.workflow)
    if args.expect is None:
        print(n)
        return 0
    if args.before is None:
        print("logs count: --expect needs --before <count from `logs count` taken before the launches>;"
              " without it the comparison is against the all-time total", file=sys.stderr)
        return 2
    before = args.before
    got = n - before
    ok = got == args.expect
    print(f"reconcile {args.workflow}: launched {args.expect}, index gained {got} "
          f"({before} -> {n}) — {'OK' if ok else 'MISMATCH — STOP, investigate before more motion'}")
    return 0 if ok else 1


def _cmd_du(args) -> int:
    root = log_root()
    if not root.is_dir():
        print(f"{root} does not exist")
        return 0
    total = 0
    for wf_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        size = _dir_size(wf_dir)
        total += size
        n = len([p for p in wf_dir.iterdir() if not p.name.startswith('.')])
        print(f"{wf_dir.name:<20} {_human(size):>10}  {n:>4} entries")
    print(f"{'TOTAL':<20} {_human(total):>10}")
    return 0


def _cmd_prune(args) -> int:
    cfg = load_config()
    workflows = ([args.workflow] if args.workflow
                 else sorted(p.name for p in log_root(cfg).iterdir() if p.is_dir()))
    if not args.yes:
        print("dry run — nothing will be deleted. Add --yes to execute.\n")
    freed = 0
    for wf in workflows:
        if wf.startswith("."):
            continue
        for entry in prune(wf, cfg=cfg, dry_run=not args.yes,
                           budget_s=args.budget, verbose=True):
            freed += entry["bytes"]
    print(f"\n{'freed' if args.yes else 'would free'} {_human(freed)}")
    return 0


def _cmd_reindex(args) -> int:
    """Make pre-runlog files visible to `list` without touching them."""
    cfg = load_config()
    root = log_root(cfg)
    known = {r.get("run_id") for r in index_runs(cfg)}
    added = 0
    for wf_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if wf_dir.name.startswith("."):
            continue
        for entry in sorted(wf_dir.iterdir()):
            if entry.name in known or entry.name == "latest":
                continue
            if entry.is_dir() and RUN_ID_RE.match(entry.name):
                continue
            if not args.legacy:
                continue
            with contextlib.suppress(Exception):
                stat = entry.stat()
                _append_line(index_path(cfg), {
                    "schema_version": SCHEMA_VERSION, "run_id": entry.name,
                    "workflow": wf_dir.name, "node": _node(), "schema": "legacy",
                    "status": "legacy", "dir": str(entry),
                    "bytes": stat.st_size if entry.is_file() else _dir_size(entry),
                    "ended_at": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
                added += 1
    print(f"indexed {added} legacy entries")
    return 0


def _cmd_compact(args) -> int:
    """One row per run, read and replaced under the appenders' lock: a row
    appended between the read and the replace used to be lost."""
    cfg = load_config()
    path = index_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _flocked(path, os.O_RDONLY | os.O_CREAT):
        runs = index_runs(cfg)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in runs))
        os.replace(tmp, path)
    print(f"compacted to {len(runs)} rows")
    return 0


def _cmd_selftest(args) -> int:
    run = init("selftest", meta={"key": {"selftest": True}}, prune_first=False)
    run.event("phase", name="check")
    logging.getLogger("tatbot.selftest").debug("debug line reaches the file")
    logging.getLogger("tatbot.selftest").info("info line")
    probe = run.path("probe.txt")
    probe.write_text("hello")
    run.artifact(probe)
    run.update(summary="selftest ok")
    run.finalize(0)
    ok = True
    for name in ("meta.json", "run.jsonl", "probe.txt"):
        if not (run.dir / name).is_file():
            print(f"FAIL missing {name}", file=sys.stderr)
            ok = False
    meta = _read_meta(run.dir) or {}
    if meta.get("status") != "ok":
        print(f"FAIL status {meta.get('status')}", file=sys.stderr)
        ok = False
    if not any(r.get("run_id") == run.run_id for r in index_runs()):
        print("FAIL run missing from index", file=sys.stderr)
        ok = False
    print(("PASS " if ok else "FAIL ") + str(run.dir))
    return 0 if ok else 1


# -- shell-facing subcommands (scripts/lib/runlog.sh calls these) ------------

def _cmd_begin(args) -> int:
    extra = {}
    for pair in args.set or []:
        key, _, val = pair.partition("=")
        extra.setdefault("key", {})[key] = val
    if args.parent_pid:
        # The launcher shell is the run; this process exits once it prints the dir.
        extra["node"] = {"pid": args.parent_pid}
    run = init(args.workflow, meta=extra, argv=(args.argv0 and [args.argv0]) or None,
               owns_console=True, attach_logging=False, emit_banner=True)
    print(run.dir)          # stdout is the contract: runlog.sh captures this
    return 0


def _cmd_end(args) -> int:
    path = Path(args.dir)
    run = RunLog(path, path.parent.name, path.name)
    meta = _read_meta(path) or {}
    started = meta.get("started_at")
    if started:
        with contextlib.suppress(Exception):
            dt = datetime.strptime(started[:19], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc)
            run.started_at = time.time() - (_utcnow() - dt).total_seconds()
    run.finalize(args.exit_code)
    meta = _read_meta(path) or {}
    _emit(banner("end", path.name,
                 f"exit={args.exit_code} {meta.get('duration_s', 0)}s  ->  "
                 f"scripts/tatbot-logs show {path.name}"))
    return 0


def _cmd_event(args) -> int:
    path = Path(args.dir)
    run = RunLog(path, path.parent.name, path.name)
    fields = {}
    for pair in args.fields or []:
        key, _, val = pair.partition("=")
        fields[key] = val
    run.event(args.kind, **fields)
    return 0


def _cmd_artifact(args) -> int:
    path = Path(args.dir)
    run = RunLog(path, path.parent.name, path.name)
    run.artifact(args.path)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="tatbot-logs", description=HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    def add_nodes(p):
        p.add_argument("--node", dest="nodes", action="append", default=None)
        p.add_argument("--all-nodes", dest="all_nodes", action="store_true")

    p = sub.add_parser("root", help="print the resolved log root and exit")
    p.set_defaults(func=lambda a: print(log_root()) or 0)

    p = sub.add_parser("list", help="recent runs, newest last")
    p.add_argument("workflow", nargs="?")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--status")
    p.add_argument("--json", action="store_true")
    add_nodes(p)
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("last", help="the most recent run of a workflow")
    p.add_argument("workflow", nargs="?")
    p.add_argument("--tail", type=int, default=40)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_last)

    p = sub.add_parser("show", help="one run: meta, files, console tail")
    p.add_argument("run_id")
    p.add_argument("--tail", type=int, default=40)
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("tail", help="follow a run that is still going")
    p.add_argument("run_id")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-n", type=int, default=40)
    p.add_argument("--file", default="console.log")
    p.set_defaults(func=_cmd_tail)

    p = sub.add_parser("fetch", help="copy a remote run here (skips media)")
    p.add_argument("run_id")
    p.add_argument("--node")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("count", help="launch count of a workflow (its `running` index rows)")
    p.add_argument("workflow")
    p.add_argument("--expect", type=int, default=None,
                   help="with --before: exit 1 unless exactly this many launches were added")
    p.add_argument("--before", type=int, default=None, help="the count printed before the launches")
    p.set_defaults(func=_cmd_count)

    p = sub.add_parser("du", help="disk use per workflow")
    p.set_defaults(func=_cmd_du)

    p = sub.add_parser("prune", help="apply retention (dry run unless --yes)")
    p.add_argument("workflow", nargs="?")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--budget", type=float, default=None)
    p.set_defaults(func=_cmd_prune)

    p = sub.add_parser("reindex", help="index pre-runlog files for visibility")
    p.add_argument("--legacy", action="store_true")
    p.set_defaults(func=_cmd_reindex)

    p = sub.add_parser("compact", help="collapse the index to one row per run")
    p.set_defaults(func=_cmd_compact)

    p = sub.add_parser("selftest", help="create, finalize and verify a run")
    p.set_defaults(func=_cmd_selftest)

    # Shell-facing: scripts/lib/runlog.sh drives these.
    p = sub.add_parser("begin")
    p.add_argument("--workflow", required=True)
    p.add_argument("--argv0")
    p.add_argument("--parent-pid", type=int)
    p.add_argument("--set", action="append")
    p.set_defaults(func=_cmd_begin)

    p = sub.add_parser("end")
    p.add_argument("--dir", required=True)
    p.add_argument("--exit-code", type=int, default=0)
    p.set_defaults(func=_cmd_end)

    p = sub.add_parser("event")
    p.add_argument("--dir", required=True)
    p.add_argument("--kind", required=True)
    p.add_argument("fields", nargs="*")
    p.set_defaults(func=_cmd_event)

    p = sub.add_parser("artifact")
    p.add_argument("--dir", required=True)
    p.add_argument("--path", required=True)
    p.set_defaults(func=_cmd_artifact)

    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        print(HELP)
        return 0
    if getattr(args, "all_nodes", False):
        args.nodes, args.retired = _sweep_nodes(load_config())
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
