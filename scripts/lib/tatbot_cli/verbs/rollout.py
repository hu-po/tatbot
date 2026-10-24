"""rollout · serve — the policy server and its offline checks."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

from tatbot_cli import EXIT_BUSY, EXIT_OK, EXIT_TOOL_FAILED
from tatbot_cli.registry import OFFLINE, REMOTE, verb
from tatbot_cli.verbs._common import lerobot_py, py, sh


def _bench_args(p):
    p.add_argument("what", choices=("plausibility",))
    p.add_argument("args", nargs="*")


@verb(effects=('read_files', 'write_files', 'network', 'start_process'), visibility="public", noun="rollout", verb="bench", tier=OFFLINE, summary="no-robot plausibility check of a policy's action chunks",
      wraps=("scripts/eval/trajectory_plausibility.py",), args=_bench_args, passthrough="trajectory_plausibility.py",
      example=("plausibility", "--", "--help"), doc="docs/imitation_learning.md")
def rollout_bench(ctx, ns, rest):
    return lerobot_py(ctx, "scripts/eval/trajectory_plausibility.py", *ns.args, *rest)


def _contract_arg(p):
    p.add_argument("source", help="checkpoint dir, config JSON, or - for stdin")


@verb(effects=('read_files',), visibility="public", noun="rollout", verb="contract", tier=OFFLINE, summary="the input/action contract stored with a checkpoint",
      wraps=("scripts/eval/checkpoint_contract.py",), passthrough="checkpoint_contract.py", args=_contract_arg,
      example=("~/il-serve/models/flagship",))
def rollout_contract(ctx, ns, rest):
    return py(ctx, "scripts/eval/checkpoint_contract.py", ns.source, *rest)


# --- serve ---------------------------------------------------------------------


def _serve_root() -> Path:
    return Path(os.environ.get("TATBOT_SERVE_ROOT", "~/il-serve")).expanduser()


def _serve_state() -> tuple[Path, dict | None]:
    state = _serve_root() / "current-server.json"
    if not state.is_file():
        return state, None
    try:
        return state, json.loads(state.read_text())
    except Exception:
        return state, {}


def _serve_start_args(p):
    p.add_argument("--policy", required=True, help="checkpoint dir with config.json")


@verb(effects=('read_files', 'network', 'start_process', 'gpu'), noun="serve", verb="start", tier=REMOTE, summary="one foreground async policy server with an explicit checkpoint contract",
      role="serve", wraps=("scripts/eval/serve.sh",), passthrough="serve.sh", args=_serve_start_args,
      example=("--policy", "~/il-serve/models/flagship"), doc="docs/imitation_learning.md",
      invariants=("A stale server on :8080 silently serves the wrong model to the next session — stop it when the session ends.",
                  "Training-only nodes are always rejected."))
def serve_start(ctx, ns, rest):
    return sh(ctx, "scripts/eval/serve.sh", "--policy", ns.policy, *rest)


@verb(effects=('read_files', 'stop_process', 'delete_files'), native=True, noun="serve", verb="stop", tier=REMOTE, summary="SIGTERM the server named in the state file", role="serve", example=(),
      invariants=("Only the server the state file names; never a broad pkill.",))
def serve_stop(ctx, ns, rest):
    state, payload = _serve_state()
    if payload is None:
        print(f"serve: no live state at {state}; nothing to stop", file=sys.stderr)
        return EXIT_OK
    if not payload.get("pid"):
        # Unreadable is not "nothing running": a server it names may still be
        # serving the wrong model on :8080.
        print(f"serve: {state} is unreadable or names no pid; nothing was stopped -- "
              "find the policy server yourself before the next session", file=sys.stderr)
        return EXIT_TOOL_FAILED
    pid = int(payload["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"serve: pid {pid} already gone; removing stale {state}")
        state.unlink(missing_ok=True)
        return EXIT_OK
    except PermissionError:
        print(f"serve: pid {pid} belongs to another user", file=sys.stderr)
        return EXIT_BUSY
    print(f"serve: sent SIGTERM to {pid}")
    return EXIT_OK
