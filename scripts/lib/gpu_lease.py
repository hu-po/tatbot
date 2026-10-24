"""A per-node GPU lease: a few slots, each an exclusive flock, with free-memory admission.

Several agents and worktrees share each node's one GPU. A process that opens a
CUDA context, or renders through the GPU's EGL vendor, holds a slot for as long
as it uses the GPU. A software render (Mesa llvmpipe) opens no GPU context and
takes no slot. The node's `gpu` record in config/nodes.json sets the slot count:

    "gpu": {"slots": 2, "reserve_mib": 2048}   a discrete card
    "gpu": {"slots": 2, "unified": true}       memory shared with the CPU

A node without a record has no lease: `acquire` hands back an empty lease and
`tatbot status` reports the row not applicable.

Admission. A slot is granted only when the node's free memory covers the
request, and the requests already granted plus this one fit in what the lease
may hand out, `total - reserve_mib`. The reserve belongs to the node's other
GPU users (a desktop session, services that never take the lease). A flock
cannot see those users; the free-memory reading can. Free memory comes from
`nvidia-smi`, or from /proc/meminfo `MemAvailable` on a unified-memory node,
whose `nvidia-smi` reports no memory figures. A request larger than the lease
could ever hand out is a ValueError at once, never a wait, even while every slot
is held.

Busy. `acquire(mib)` raises BlockingIOError at once when no slot can be
granted; the tatbot CLI maps that to exit 6, and a unit test SKIPs on it. A run
that should wait passes `wait_s` and joins a bounded first-come queue; it gets
the same error if the queue is full or the wait runs out.

Files live under $TATBOT_GPU_LEASE_DIR, else <XDG state>/tatbot/gpu-lease. They
never go in /tmp, which systemd's PrivateTmp hides from services.
  slot-<i>.lock  flocked by its holder while the lease lasts and never removed,
                 so the kernel frees the slot however the holder dies
  slot-<i>.json  the holder {pid, run_id, worktree, purpose, mib, since}, read
                 only while its slot is locked (a dead holder's file is ignored)
  queue/*.json   one flocked ticket per waiter, oldest first
  admit.lock     serialises every grant and every read of who holds what: a
                 slot's holder is found by trying its lock, and an unserialised
                 try would collide with a grant's own
The lease is per login user. Point TATBOT_GPU_LEASE_DIR at a shared directory
where a node's GPU users log in as different users.

STDLIB ONLY, like tatbot_paths: the CLI, the tests and the engine environments
all import it. `python3 scripts/lib/gpu_lease.py status` prints this node's
lease; `hold --mib N [--wait-s S] -- CMD...` runs CMD under a slot and exits 6
when the lease is busy.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import tatbot_paths

MAX_QUEUE = 4  # waiters past this are refused at once: each one may wait the full wait_s
POLL_S = 2.0
EXIT_BUSY = 6  # tatbot_cli.EXIT_BUSY
_RECORD_KEYS = {"slots", "reserve_mib", "unified"}
_NVIDIA_SMI = ("nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits")


def lease_dir() -> Path:
    """$TATBOT_GPU_LEASE_DIR, else the per-user state directory's gpu-lease."""
    env = os.environ.get("TATBOT_GPU_LEASE_DIR", "").strip()
    return Path(env).expanduser() if env else tatbot_paths.state_root() / "gpu-lease"


def _cli():
    """tatbot_cli's node map and lock probe; the package sits beside this file."""
    try:
        from tatbot_cli import locks, nodes
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tatbot_cli import locks, nodes
    return locks, nodes


def validate(gpu) -> dict | None:
    """A `gpu` record with its defaults filled in; None stays None (no lease)."""
    if gpu is None:
        return None
    if not isinstance(gpu, dict):
        raise ValueError("a gpu record is an object")
    unknown = sorted(k for k in gpu if k not in _RECORD_KEYS and not k.startswith("//"))
    if unknown:
        raise ValueError(f"unknown gpu record keys {unknown}; known: {sorted(_RECORD_KEYS)}")
    slots, reserve = gpu.get("slots"), gpu.get("reserve_mib", 0)
    for name, value, floor in (("slots", slots, 1), ("reserve_mib", reserve, 0)):
        if type(value) is not int or value < floor:
            raise ValueError(f"gpu.{name} must be an integer >= {floor}, not {value!r}")
    if type(gpu.get("unified", False)) is not bool:
        raise ValueError("gpu.unified must be true or false")
    return {"slots": slots, "reserve_mib": reserve, "unified": gpu.get("unified", False)}


def node_gpu(nmap: dict | None = None, node: str | None = None) -> dict | None:
    """This node's `gpu` record from config/nodes.json, validated; None without one."""
    _, nodes = _cli()
    if nmap is None:
        nmap = nodes.load(tatbot_paths.repo_root())
    node = node or nodes.this_node(nmap)
    return validate(nmap.get(node, {}).get("gpu"))


def parse_nvidia_smi(text: str) -> dict:
    """`memory.total,memory.free` in MiB (csv, no header, no units) for the first GPU."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("nvidia-smi listed no GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 2 or not all(field.isdigit() for field in fields):
        raise ValueError(f"nvidia-smi reports no memory figures ({lines[0].strip()!r}); "
                         "a unified-memory node's gpu record says \"unified\": true")
    total, free = (int(field) for field in fields)
    return {"total_mib": total, "free_mib": free, "source": "nvidia-smi"}


def parse_meminfo(text: str) -> dict:
    """MemTotal and MemAvailable from /proc/meminfo, in MiB."""
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in ("MemTotal", "MemAvailable"):
            values[name] = int(rest.split()[0]) // 1024
    if len(values) != 2:
        raise ValueError("/proc/meminfo lacks MemTotal or MemAvailable")
    return {"total_mib": values["MemTotal"], "free_mib": values["MemAvailable"], "source": "meminfo"}


def read_memory(gpu: dict, *, timeout_s: float = 5.0) -> dict:
    """{total_mib, free_mib, source} for this node's GPU, as its record says to read it."""
    if gpu["unified"]:
        return parse_meminfo(Path("/proc/meminfo").read_text())
    reply = subprocess.run(_NVIDIA_SMI, capture_output=True, text=True, timeout=timeout_s, check=False)
    if reply.returncode:
        raise ValueError(f"nvidia-smi exit {reply.returncode}: {reply.stderr.strip()[:200]}")
    return parse_nvidia_smi(reply.stdout)


def refusal(gpu: dict, mib: int, memory: dict, granted_mib: int) -> str | None:
    """Why `mib` cannot be granted now, or None when it can."""
    budget = memory["total_mib"] - gpu["reserve_mib"]
    if memory["free_mib"] < mib:
        return f"{memory['free_mib']} MiB free < {mib} MiB requested ({memory['source']})"
    if granted_mib + mib > budget:
        return (f"{granted_mib} MiB already granted + {mib} MiB requested > {budget} MiB the lease "
                f"hands out ({memory['total_mib']} total - {gpu['reserve_mib']} reserved)")
    return None


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict) -> None:
    """Whole-file replace, so a reader never sees half a holder."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _slots(directory: Path, count: int) -> tuple[list[dict], list[int]]:
    """(held slots with their holders, free slot numbers). Creates nothing."""
    locks, _ = _cli()
    held, free = [], []
    for slot in range(count):
        if locks.held(directory / f"slot-{slot}.lock"):
            held.append({"slot": slot, **_read_json(directory / f"slot-{slot}.json")})
        else:
            free.append(slot)
    return held, free


def _queue(directory: Path, *, prune: bool) -> list[tuple[str, dict]]:
    """Live tickets, oldest first. A dead waiter's ticket is removed when `prune`."""
    locks, _ = _cli()
    live = []
    for path in sorted((directory / "queue").glob("*.json")):
        if locks.held(path):
            live.append((path.name, _read_json(path)))
        elif prune:
            path.unlink(missing_ok=True)
    return live


class _Ticket:
    """A waiter's place in the queue: a flocked file, renamed in only once locked."""

    def __init__(self, directory: Path, holder: dict):
        queue = directory / "queue"
        queue.mkdir(exist_ok=True)
        self.name = f"{time.time_ns():020d}-{os.getpid()}-{id(self):x}.json"
        staging = queue / f".{self.name}.tmp"
        self._fd = os.open(staging, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        os.write(self._fd, json.dumps(holder, sort_keys=True).encode())
        self.path = queue / self.name
        os.rename(staging, self.path)

    def close(self) -> None:
        self.path.unlink(missing_ok=True)
        os.close(self._fd)


class Lease:
    """A granted slot. `release()`, or leaving its `with` block, frees it.

    `slot` is None on a node without a gpu record: there is nothing to share and
    nothing is held."""

    def __init__(self, slot: int | None = None, fd: int | None = None,
                 sidecar: Path | None = None, holder: dict | None = None):
        self.slot, self.holder, self._fd, self._sidecar = slot, holder, fd, sidecar

    @property
    def held(self) -> bool:
        return self._fd is not None

    def fileno(self) -> int:
        """The slot's lock; pass it to a child (pass_fds) that must keep the slot if this process dies."""
        if self._fd is None:
            raise ValueError("no slot is held")
        return self._fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            self._sidecar.unlink(missing_ok=True)  # before the unlock, so no new holder's file is lost
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> Lease:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _claim(directory: Path, slot: int, holder: dict) -> Lease | None:
    path = directory / f"slot-{slot}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sidecar = directory / f"slot-{slot}.json"
        _write_json(sidecar, holder)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    return Lease(slot, fd, sidecar, holder)


def _try_grant(directory: Path, gpu: dict, holder: dict, read, ticket: _Ticket | None):
    """(Lease, None) or (None, why not). Runs under admit.lock."""
    held, free = _slots(directory, gpu["slots"])
    queue = [name for name, _ in _queue(directory, prune=True)]
    ahead = queue.index(ticket.name) if ticket else len(queue)
    if len(free) <= ahead:
        holders = ", ".join(f"pid {h.get('pid')} {h.get('purpose') or '?'}" for h in held)
        return None, f"{len(held)}/{gpu['slots']} slots held ({holders}); {ahead} waiting ahead"
    why = refusal(gpu, holder["mib"], read(), sum(int(h.get("mib") or 0) for h in held))
    if why:
        return None, why
    for slot in free:
        lease = _claim(directory, slot, holder)
        if lease:
            return lease, None
    return None, "every free slot was taken meanwhile"


@contextlib.contextmanager
def _admission(directory: Path):
    fd = os.open(directory / "admit.lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _observe(directory: Path, count: int, timeout_s: float) -> tuple[list[dict], list[int], list[dict]]:
    """(held, free, queued) read under admit.lock, so the read never makes a grant fail.

    Creates nothing: without admit.lock no grant was ever made here. Waits at
    most `timeout_s` for an admission in progress."""
    try:
        fd = os.open(directory / "admit.lock", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return [], list(range(count)), []
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{directory / 'admit.lock'} stayed locked for {timeout_s:.1f} s") from None
                time.sleep(0.002)
        held, free = _slots(directory, count)
        return held, free, [value for _, value in _queue(directory, prune=False)]
    finally:
        os.close(fd)


def acquire(mib: int, wait_s: float = 0.0, purpose: str = "", *, run_id: str | None = None,
            gpu: dict | None = None, directory: Path | None = None, memory=None,
            poll_s: float = POLL_S, max_queue: int = MAX_QUEUE) -> Lease:
    """A slot for `mib` MiB of GPU memory, as a context manager.

    wait_s 0 raises BlockingIOError at once when the lease is busy; a positive
    wait queues (at most `max_queue` waiters) and raises it when the wait runs
    out. `gpu` defaults to this node's record and `memory` (a callable returning
    read_memory's dict) to reading the node. With no record the lease is empty."""
    if type(mib) is not int or mib < 0:
        raise ValueError(f"mib must be a non-negative integer, not {mib!r}")
    gpu = validate(gpu) if gpu is not None else node_gpu()
    if gpu is None:
        return Lease()
    read = memory or (lambda: read_memory(gpu))
    budget = read()["total_mib"] - gpu["reserve_mib"]
    if mib > budget:  # checked before the queue: no wait could make it fit
        raise ValueError(f"{mib} MiB can never be granted here: the lease hands out at most {budget} MiB")
    directory = Path(directory) if directory is not None else lease_dir()
    directory.mkdir(parents=True, exist_ok=True)
    holder = {"pid": os.getpid(), "run_id": run_id or os.environ.get("TATBOT_RUN_ID"),
              "worktree": str(tatbot_paths.repo_root()), "purpose": purpose, "mib": mib, "since": _utc()}
    deadline = time.monotonic() + max(0.0, wait_s)
    ticket = None
    try:
        while True:
            with _admission(directory):
                if ticket is None and wait_s > 0:
                    ticket = _join(directory, holder, max_queue)
                lease, why = _try_grant(directory, gpu, holder, read, ticket)
            remaining = deadline - time.monotonic()
            if lease or remaining <= 0:
                break
            time.sleep(min(poll_s, remaining))
    finally:
        if ticket:
            ticket.close()
    if lease is None:
        raise BlockingIOError(f"GPU lease busy: {why}")
    return lease


def _join(directory: Path, holder: dict, max_queue: int) -> _Ticket:
    waiting = len(_queue(directory, prune=True))
    if waiting >= max_queue:
        raise BlockingIOError(f"GPU lease busy: the queue is full ({waiting} waiting)")
    return _Ticket(directory, holder)


def status(gpu: dict, *, directory: Path | None = None, memory=None, timeout_s: float = 5.0) -> dict:
    """Holders, waiters, free memory and what could be granted now. Read-only."""
    gpu = validate(gpu)
    directory = Path(directory) if directory is not None else lease_dir()
    held, free, queued = _observe(directory, gpu["slots"], timeout_s)
    reading = (memory or (lambda: read_memory(gpu, timeout_s=timeout_s)))()
    granted = sum(int(h.get("mib") or 0) for h in held)
    grantable = min(reading["free_mib"], reading["total_mib"] - gpu["reserve_mib"] - granted)
    return {"dir": str(directory), **gpu, "free_slots": len(free), "held": held, "queued": queued,
            "memory": reading, "granted_mib": granted,
            "grantable_mib": max(0, grantable) if len(free) > len(queued) else 0}


def _hold(args) -> int:
    try:
        lease = acquire(args.mib, args.wait_s, args.purpose, run_id=args.run_id)
    except BlockingIOError as exc:
        print(f"gpu_lease: {exc}", file=sys.stderr)
        return EXIT_BUSY
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"gpu_lease: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    with lease:
        if not lease.held:
            print("gpu_lease: this node has no gpu record; running without a lease", file=sys.stderr)
        # The child keeps the slot's lock open, so a killed wrapper does not free a slot still in use.
        fds = (lease.fileno(),) if lease.held else ()
        return subprocess.run(args.command, pass_fds=fds, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpu_lease.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="this node's lease as JSON")
    hold = sub.add_parser("hold", help="run a command under a slot; exit 6 when busy")
    hold.add_argument("--mib", type=int, required=True)
    hold.add_argument("--wait-s", type=float, default=0.0)
    hold.add_argument("--purpose", default="")
    hold.add_argument("--run-id")
    hold.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.cmd == "hold":
        args.command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not args.command:
            parser.error("hold needs a command after --")
        return _hold(args)
    try:
        gpu = node_gpu()
        report = status(gpu) if gpu else {"applicable": False, "reason": "no gpu record for this node"}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"gpu_lease: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
