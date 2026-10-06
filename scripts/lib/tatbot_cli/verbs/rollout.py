"""rollout · serve — trained policies on the arm, and the server that feeds them."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

from tatbot_cli import EXIT_BUSY, EXIT_OK, EXIT_TOOL_FAILED
from tatbot_cli.registry import MOTION_AUTO, OFFLINE, REMOTE, verb
from tatbot_cli.verbs._common import (
    ink_argv,
    ink_flags,
    lerobot_py,
    py,
    sh,
    tag_arg,
    tool_flag,
)

GATE_INV = (
    "Policies use the physical left arm and its local wrist views; checkpoint camera keys must match exactly.",
    "Never loosen force, workspace-floor or retreat limits to make it work; re-zeroing the floor (--robot.z_floor_m) is legitimate.",
    "One launch at a time; the launcher refuses while another client holds the arm.",
    "Reconcile run COUNT before calling any launch uncommanded: `tatbot logs count rollout_async` before, then "
    "`--expect N --before M` after.",
    "Every launch carries an automatic launch id, ledgered and audited by arm_gate (pid chain, SSH origin); "
    "--tag adds a label.",
)


def _run_args(p):
    p.add_argument("policy", nargs="?", help="server-side checkpoint dir (server default when omitted)")
    p.add_argument("--duration", type=int, default=60, help="seconds (default 60)")
    p.add_argument("--type", dest="policy_type", help="act | multi_task_dit | evo1")
    tag_arg(p)
    ink_flags(p)


@verb(effects=('read_files', 'write_files', 'network', 'autonomous_motion', 'sensor_read'), noun="rollout", verb="run", tier=MOTION_AUTO, summary="flagship async rollout: policy server on the serve node, robot client here",
      role="arm", wraps=("scripts/il_rollout_async.sh",), passthrough="robot_client", args=_run_args, needs_tool=True,
      launch_id=True, ink_hook=True, example=("--duration", "60", "--tag", "a1",), doc="docs/imitation_learning.md",
      tool_arm="left", invariants=GATE_INV)
def rollout_run(ctx, ns, rest):
    pos = [str(ns.duration)]
    if ns.policy:
        pos.append(ns.policy)
        if ns.policy_type:
            pos.append(ns.policy_type)
    elif ns.policy_type:
        print("rollout run: --type needs a policy path (positional order of il_rollout_async.sh)", file=sys.stderr)
        return 2
    return sh(ctx, "scripts/il_rollout_async.sh", *pos, *tool_flag(ctx), *ink_argv(ns), *rest)


def _analyze_args(p):
    p.add_argument("target", nargs="*", help="run dir, run id, flight CSV, or analysis.json with --compare")
    p.add_argument('--arm', choices=('left', 'right'), help='physical arm for historical logs without arm metadata')
    p.add_argument("--settle", type=float, help="seconds to ignore at the start (default: il_analyze_rollout.SETTLE_S)")
    p.add_argument("--lift-mm", type=float, help="clearance above the floor that counts as a lift (default 6)")


@verb(effects=('read_files', 'write_files'), visibility="public", noun="rollout", verb="analyze", tier=OFFLINE, summary="did the pen draw the shape, did the loop keep time",
      wraps=("scripts/il_analyze_rollout.py",), passthrough="il_analyze_rollout.py", args=_analyze_args,
      example=("~/tatbot-logs/rollout_async/last",), doc="docs/imitation_learning.md",
      invariants=("FK 'contact' is proximity to a calibrated plane, NOT a touch measurement; the operator's eyes outrank it.",
                  "Tip height, sustained lifts, descent and xy footprint (hull area) come from the touched-off tip on "
                  "the recorded physical arm's tool_mount; --settle / --lift-mm tune the window and the lift threshold.",
                  "A/B two policies by running `rollout run` once per rep (--tag labels each: a1 b1 a2 b2 ...), then "
                  "`rollout analyze -- --compare` over the analysis.json files."))
def rollout_analyze(ctx, ns, rest):
    flags: list[str] = []
    if ns.arm is not None:
        flags += ['--arm', ns.arm]
    if ns.settle is not None:
        flags += ["--settle", str(ns.settle)]
    if ns.lift_mm is not None:
        flags += ["--lift-mm", str(ns.lift_mm)]
    return lerobot_py(ctx, "scripts/il_analyze_rollout.py", *ns.target, *flags, *rest)


def _bench_args(p):
    p.add_argument("what", choices=("wire", "plausibility"))
    p.add_argument("args", nargs="*")


@verb(effects=('read_files', 'write_files', 'network', 'start_process'), visibility="public", noun="rollout", verb="bench", tier=OFFLINE, summary="no-robot checks of the serving path (wire bench / plausibility)",
      wraps=("scripts/eval/wire_bench.py", "scripts/eval/trajectory_plausibility.py"), args=_bench_args, passthrough="wire_bench.py / trajectory_plausibility.py",
      example=("wire", "--", "--help"), doc="docs/imitation_learning.md",
      invariants=("Run the wire bench whenever a model, feature set or server is new, before committing the arm.",))
def rollout_bench(ctx, ns, rest):
    # Neither bench takes a tool: the wire bench builds the follower's features
    # without one (wire_client.build_feature_robot), and plausibility is offline.
    script = "scripts/eval/wire_bench.py" if ns.what == "wire" else "scripts/eval/trajectory_plausibility.py"
    return lerobot_py(ctx, script, *ns.args, *rest)


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
