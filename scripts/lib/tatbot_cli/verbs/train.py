"""train · data — GPU nodes and datasets."""

from __future__ import annotations

import os
import sys

from tatbot_cli import EXIT_OK, EXIT_USAGE, gates, interp
from tatbot_cli.registry import OFFLINE, REMOTE, Plan, verb
from tatbot_cli.verbs._common import lerobot_py, py, sh, tool_flag

TRAIN_INV = (
    "One GPU training job per node; run.sh holds ~/il-train/.tatbot-training.lock.",
    "Honor SWEEP_PAUSE: run.sh refuses to start while ~/il-train/SWEEP_PAUSE exists.",
    "Training nodes never serve a rollout policy.",
)


# --- train ---------------------------------------------------------------------


@verb(effects=('read_files', 'write_files', 'network', 'start_process', 'gpu'), noun="train", verb="run", tier=OFFLINE, summary="exactly one LeRobot training job, foreground, with the shared invariants",
      role="train", wraps=("scripts/train/run.sh",), passthrough="lerobot-train",
      example=("--", "--help"), doc="docs/imitation_learning.md", invariants=TRAIN_INV)
def train_run(ctx, ns, rest):
    return sh(ctx, "scripts/train/run.sh", *rest)


def _manifest_args(p):
    p.add_argument("manifest")
    p.add_argument("job")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--render", action="store_true")
    g.add_argument("--execute", action="store_true")


def _manifest_effects(effects, ns, rest):
    return effects if ns.execute else effects - {"start_process", "gpu", "write_files", "network"}


@verb(refine_effects=_manifest_effects, effects=('read_files', 'write_files', 'network', 'start_process', 'gpu'), noun="train", verb="manifest", tier=OFFLINE, summary="render or execute one job from an experiment manifest",
      role="train", wraps=("scripts/train/manifest_job.py",), args=_manifest_args,
      example=("<manifest.json>", "<job>", "--render"), invariants=TRAIN_INV)
def train_manifest(ctx, ns, rest):
    # manifest_job.py renders by default and only defines --execute. Keep the
    # CLI's explicit --render spelling as a user-facing no-op instead of
    # forwarding an option the wrapped tool rejects.
    flag = ["--execute"] if ns.execute else []
    return py(ctx, "scripts/train/manifest_job.py", ns.manifest, ns.job, *flag, *rest)


def _train_python(ctx, rel: str, *args: str) -> Plan:
    """Run LeRobot-dependent tooling in the node's pinned training venv."""

    python = gates.train_root() / ".venv" / "bin" / "python"
    return Plan(
        argv=[str(python), ctx.path(rel), *args],
        notes=[f"interpreter: pinned training environment {python}"],
    )


@verb(effects=('read_files', 'write_files', 'gpu'), noun="train", verb="offline-eval", tier=OFFLINE, summary="score saved checkpoints on a held-out dataset",
      role="train", wraps=("scripts/train/offline_eval.py",), passthrough="offline_eval.py", example=("--", "--help"))
def train_offline_eval(ctx, ns, rest):
    return _train_python(ctx, "scripts/train/offline_eval.py", *rest)


@verb(effects=('read_files',), noun="train", verb="profile", tier=OFFLINE, summary="this node's training profile (paths, tuning family)",
      wraps=("scripts/train/node_profile.py",), passthrough="node_profile.py", example=())
def train_profile(ctx, ns, rest):
    return py(ctx, "scripts/train/node_profile.py", *rest)


@verb(effects=('read_files', 'write_files'), native=True, output="text", noun="train", verb="pause", tier=OFFLINE, summary="create ~/il-train/SWEEP_PAUSE (a rollout owns the GPU)",
      role="train", example=(), invariants=("Do not remove the marker until the rollout owner releases it.",))
def train_pause(ctx, ns, rest):
    marker = gates.train_root() / "SWEEP_PAUSE"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    print(f"train: paused — {marker}")
    return EXIT_OK


@verb(effects=('read_files', 'delete_files'), native=True, output="text", noun="train", verb="resume", tier=OFFLINE, summary="remove ~/il-train/SWEEP_PAUSE", role="train", example=())
def train_resume(ctx, ns, rest):
    marker = gates.train_root() / "SWEEP_PAUSE"
    if marker.exists():
        marker.unlink()
        print(f"train: resumed — removed {marker}")
    else:
        print("train: not paused")
    return EXIT_OK


# --- data ----------------------------------------------------------------------
#
# Two tools, two environments, two verbs. `data hub` is scripts/dataset_hub.py
# (huggingface_hub only, so any environment that imports it will do). `data ds`
# is scripts/train/dataset.py, which imports torch, pyarrow and lerobot at the
# top: system python3 has none of them on any node, so it runs under the pinned
# LeRobot environment the way `train offline-eval` and `rollout bench` do.

HUB_SUBS = ("push", "pull", "list", "info", "whoami", "set-record", "set-list", "set-pull")
DS_SUBS = ("split", "aggregate", "recompute-stats", "canonicalize", "normalize-task", "feature-view", "depth-view",
           "validate", "digest", "compare")


def _hub_args(p):
    p.add_argument("sub", choices=HUB_SUBS, metavar="<sub>", help="|".join(HUB_SUBS))
    p.add_argument("args", nargs="*", help="dataset_hub.py arguments (put flags after `--`)")


def _hub_effects(effects, ns, rest):
    if ns.sub in ("list", "info", "whoami", "set-list"):
        return effects - {"write_files", "remote_write"}
    if ns.sub in ("pull", "set-pull"):
        return effects - {"remote_write"}
    return effects


@verb(refine_effects=_hub_effects, effects=('read_files', 'write_files', 'network', 'remote_write'), noun="data", verb="hub", tier=REMOTE, summary="the Hugging Face dataset archive: push / pull / list / info / whoami / set-*",
      wraps=("scripts/dataset_hub.sh", "scripts/dataset_hub.py"), passthrough="dataset_hub.py <sub>", args=_hub_args,
      example=("push", "--", "--help"), doc="docs/imitation_learning.md",
      invariants=("Runs dataset_hub.py under the first environment that imports huggingface_hub "
                  "(plugin venv, then ~/il-train), else a throwaway `uv run --with` one.",
                  "Everything it creates is private under hu-po/; real recordings always, sim batches only when a "
                  "manifest cites them, derived views never.",
                  "Flags go after `--` (`data hub push -- --root DIR`): bare flags before it are re-ordered by argparse."))
def data_hub(ctx, ns, rest):
    python, why = interp.hub_python(ctx.repo, probe=not ctx.dry_run)
    if python is None:
        print(f"data hub: {why}", file=sys.stderr)
        return EXIT_USAGE
    return Plan(argv=[*python, ctx.path("scripts/dataset_hub.py"), ns.sub, *ns.args, *rest], notes=[f"interpreter: {why}"])


def _ds_args(p):
    p.add_argument("sub", choices=DS_SUBS, metavar="<sub>", help="|".join(DS_SUBS))
    p.add_argument("args", nargs="*", help="dataset.py arguments (put flags after `--`)")


def _ds_effects(effects, ns, rest):
    return effects - {"write_files"} if ns.sub in ("validate", "digest", "compare") else effects


@verb(refine_effects=_ds_effects, effects=('read_files', 'write_files'), noun="data", verb="ds", tier=OFFLINE, summary="LeRobot dataset tools: split / aggregate / stats / views / validate / digest / compare",
      wraps=("scripts/train/dataset.py",), passthrough="train/dataset.py <sub>", args=_ds_args,
      example=("digest", "--", "--help"), doc="docs/imitation_learning.md",
      invariants=("Runs under the pinned LeRobot environment (plugin venv, ~/il-serve, ~/il-train, else `uv run`); "
                  "dataset.py imports torch/pyarrow/lerobot and cannot run on system python3.",
                  "Derived views (split / feature-view / depth-view) are hardlinked and never pushed to the hub.",
                  "`compare A B` diffs two feature dicts before an aggregate; aggregate refuses on any difference."))
def data_ds(ctx, ns, rest):
    # On a training node the pinned training venv wins (train run consumes what it
    # aggregates); elsewhere whatever LeRobot environment this node carries.
    if os.access(gates.train_root() / ".venv" / "bin" / "python", os.X_OK):
        return _train_python(ctx, "scripts/train/dataset.py", ns.sub, *ns.args, *rest)
    return lerobot_py(ctx, "scripts/train/dataset.py", ns.sub, *ns.args, *rest)


@verb(effects=('read_files', 'write_files'), noun="data", verb="tool-meta", tier=OFFLINE, summary="stamp a recorded dataset with meta/tool.json",
      wraps=("scripts/il_tool_meta.py",), passthrough="il_tool_meta.py", needs_tool=True, example=("--", "--help"),
      doc="docs/tools.md")
def data_tool_meta(ctx, ns, rest):
    return py(ctx, "scripts/il_tool_meta.py", *tool_flag(ctx), *rest)
