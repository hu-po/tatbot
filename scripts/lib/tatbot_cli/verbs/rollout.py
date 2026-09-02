"""rollout · serve — trained policies on the arm, and the server that feeds them."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

from tatbot_cli import EXIT_BUSY, EXIT_OK
from tatbot_cli.registry import MOTION_AUTO, OFFLINE, REMOTE, SENSOR, Plan, verb
from tatbot_cli.verbs._common import (
    ink_argv,
    ink_flags,
    lerobot_py,
    nonce_arg,
    py,
    sh,
    tool_flag,
)

GATE_INV = (
    "A human with the E-stop is physically present; the launcher fails closed without /dev/tatbot-estop.",
    "Never loosen force, workspace-floor or retreat limits to make it work; re-zeroing the floor (--robot.z_floor_m) is legitimate.",
    "One launch at a time; the launcher refuses while another client holds the arm.",
    "Reconcile run COUNT before calling any launch uncommanded (tatbot rollout reconcile).",
    "Every launch consumes one single-use --nonce; a replayed command is refused and audited.",
)


def _run_args(p):
    p.add_argument("policy", nargs="?", help="server-side checkpoint dir (server default when omitted)")
    p.add_argument("--duration", type=int, default=60, help="seconds (default 60)")
    p.add_argument("--type", dest="policy_type", help="groot (default) | act | multi_task_dit | evo1")
    nonce_arg(p)
    ink_flags(p)


@verb(noun="rollout", verb="run", tier=MOTION_AUTO, summary="flagship async rollout: policy server on the serve node, robot client here",
      role="arm", wraps=("scripts/il_rollout_async.sh",), passthrough="robot_client", args=_run_args, needs_tool=True,
      nonce=True, dip_hook=True, example=("--duration", "60", "--nonce", "sleepy-sat-0829-a",), doc="docs/imitation_learning.md",
      invariants=GATE_INV)
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


def _sync_args(p):
    p.add_argument("policy", help="local checkpoint dir (config.json / train_config.json)")
    p.add_argument("--duration", type=int, default=60)
    nonce_arg(p)
    ink_flags(p)


@verb(noun="rollout", verb="sync", tier=MOTION_AUTO, summary="legacy synchronous absolute-action rollout (explicit checkpoint)",
      role="arm", wraps=("scripts/il_rollout.sh",), passthrough="lerobot-rollout", args=_sync_args, needs_tool=True,
      nonce=True, dip_hook=True, example=("~/il-serve/models/act/checkpoints/last/pretrained_model", "--nonce", "sleepy-sat-0829-b"),
      doc="docs/imitation_learning.md", invariants=GATE_INV + ("Relative-action GR00T is refused here; use rollout run.",))
def rollout_sync(ctx, ns, rest):
    return sh(ctx, "scripts/il_rollout.sh", ns.policy, str(ns.duration), *tool_flag(ctx), *ink_argv(ns), *rest)


def _compare_args(p):
    p.add_argument("policies", nargs="+", help="name=dir … [reps] [duration] — passed as-is to il_compare_policies.sh")
    nonce_arg(p)


@verb(noun="rollout", verb="compare", tier=MOTION_AUTO, summary="interleaved on-robot A/B of N policies (A B A B …)",
      role="arm", wraps=("scripts/il_compare_policies.sh",), args=_compare_args, needs_tool=True, nonce=True,
      example=("h64=/x/h64", "h64flow=/x/h64flow", "3", "30", "--nonce", "sleepy-sat-0829-c"),
      doc="docs/imitation_learning.md",
      invariants=GATE_INV + ("Start from a FRESH SHEET; the script cannot check that.",
                             "Each interleaved launch goes through arm_gate; the first nonce covers only the first launch."))
def rollout_compare(ctx, ns, rest):
    return sh(ctx, "scripts/il_compare_policies.sh", *ns.policies, *tool_flag(ctx), *rest)


def _target_arg(p):
    p.add_argument("target", nargs="*", help="run dir, run id, flight CSV, or analysis.json with --compare")


@verb(noun="rollout", verb="analyze", tier=OFFLINE, summary="did the pen draw the shape, did the loop keep time",
      wraps=("scripts/il_analyze_rollout.py",), passthrough="il_analyze_rollout.py", args=_target_arg,
      example=("~/tatbot-logs/rollout_async/last",), doc="docs/imitation_learning.md",
      invariants=("FK 'contact' is proximity to a calibrated plane, NOT a touch measurement; the operator's eyes outrank it.",))
def rollout_analyze(ctx, ns, rest):
    return py(ctx, "scripts/il_analyze_rollout.py", *ns.target, *rest)


@verb(noun="rollout", verb="fk", tier=OFFLINE, summary="FK read of a rollout flight log (lift/descent/footprint)",
      wraps=("scripts/eval/flight_fk.py",), passthrough="flight_fk.py", args=_target_arg,
      example=("~/tatbot-logs/rollout_async/last",))
def rollout_fk(ctx, ns, rest):
    return py(ctx, "scripts/eval/flight_fk.py", *ns.target, *rest)


def _reconcile_args(p):
    p.add_argument("what", choices=("snapshot", "check"))
    p.add_argument("args", nargs="*", help="check: <N> <before>")


@verb(noun="rollout", verb="reconcile", tier=OFFLINE, summary="rollout run COUNT vs the index — count before calling anything uncommanded",
      wraps=("scripts/eval/reconcile.py",), args=_reconcile_args, example=("snapshot",),
      doc="docs/imitation_learning.md")
def rollout_reconcile(ctx, ns, rest):
    return py(ctx, "scripts/eval/reconcile.py", ns.what, *ns.args, *rest)


def _bench_args(p):
    p.add_argument("what", choices=("wire", "plausibility"))
    p.add_argument("args", nargs="*")


@verb(noun="rollout", verb="bench", tier=OFFLINE, summary="no-robot checks of the serving path (wire bench / plausibility)",
      wraps=("scripts/eval/wire_bench.py", "scripts/eval/trajectory_plausibility.py"), args=_bench_args,
      example=("wire", "--", "--help"), doc="docs/imitation_learning.md",
      invariants=("Run the wire bench whenever a model, feature set or server is new, before committing the arm.",))
def rollout_bench(ctx, ns, rest):
    script = "scripts/eval/wire_bench.py" if ns.what == "wire" else "scripts/eval/trajectory_plausibility.py"
    args = [*ns.args, *tool_flag(ctx), *rest] if ctx.ee_tool else [*ns.args, *rest]
    return lerobot_py(ctx, script, *args)


def _contract_arg(p):
    p.add_argument("source", help="checkpoint dir, config JSON, or - for stdin")


@verb(noun="rollout", verb="contract", tier=OFFLINE, summary="the input/action contract stored with a checkpoint",
      wraps=("scripts/eval/checkpoint_contract.py",), passthrough="checkpoint_contract.py", args=_contract_arg,
      example=("~/il-serve/models/flagship",))
def rollout_contract(ctx, ns, rest):
    return py(ctx, "scripts/eval/checkpoint_contract.py", ns.source, *rest)


@verb(noun="rollout", verb="replay-safety", tier=OFFLINE, summary="replay flight CSVs through the measured-motion watchdog (hardware-free)",
      wraps=("scripts/eval/replay_motion_safety.py",), passthrough="replay_motion_safety.py", example=("--", "--help"),
      invariants=("Incident regression and threshold audits only — never a substitute for operator-observed acceptance.",))
def rollout_replay_safety(ctx, ns, rest):
    return py(ctx, "scripts/eval/replay_motion_safety.py", *rest)


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


@verb(noun="serve", verb="start", tier=REMOTE, summary="one foreground async policy server with an explicit checkpoint contract",
      role="serve", wraps=("scripts/eval/serve.sh",), passthrough="serve.sh", args=_serve_start_args,
      example=("--policy", "~/il-serve/models/flagship"), doc="docs/imitation_learning.md",
      invariants=("A stale server on :8080 silently serves the wrong model to the next session — stop it when the session ends.",
                  "Training-only nodes are always rejected."))
def serve_start(ctx, ns, rest):
    return sh(ctx, "scripts/eval/serve.sh", "--policy", ns.policy, *rest)


@verb(noun="serve", verb="status", tier=SENSOR, summary="what the policy server on this node is serving", role="serve", example=())
def serve_status(ctx, ns, rest):
    if ctx.dry_run:
        return Plan(argv=["<native>", "read", str(_serve_root() / "current-server.json")])
    state, payload = _serve_state()
    if payload is None:
        print(f"serve: no state file at {state} — nothing is serving (or it exited cleanly)")
        return EXIT_OK
    pid = payload.get("pid")
    alive = False
    if pid is not None:
        try:
            os.kill(int(pid), 0)
            alive = True
        except Exception:
            pass
    payload["alive"] = alive
    print(json.dumps(payload, indent=2) if ctx.json else
          f"serve: pid {pid} {'alive' if alive else 'DEAD (stale state file)'}  policy={payload.get('policy') or payload.get('policy_path')}  port={payload.get('port')}")
    return EXIT_OK


@verb(noun="serve", verb="stop", tier=REMOTE, summary="SIGTERM the server named in the state file", role="serve", example=(),
      invariants=("Only the server the state file names; never a broad pkill.",))
def serve_stop(ctx, ns, rest):
    state, payload = _serve_state()
    if not payload or not payload.get("pid"):
        print(f"serve: no live state at {state}; nothing to stop", file=sys.stderr)
        return EXIT_OK
    pid = int(payload["pid"])
    if ctx.dry_run:
        return Plan(argv=["kill", "-TERM", str(pid)], notes=[f"from {state}"])
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
