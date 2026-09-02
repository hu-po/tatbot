"""train · data · sim — GPU nodes, datasets, and the x86-only sim factory."""

from __future__ import annotations

import sys

from tatbot_cli import EXIT_OK, EXIT_USAGE, gates, interp
from tatbot_cli.registry import OFFLINE, REMOTE, Plan, verb
from tatbot_cli.verbs._common import SIM_PROJECT, py, sh, tool_flag, uvmod, uvpy

TRAIN_INV = (
    "One GPU training job per node; run.sh holds ~/il-train/.tatbot-training.lock.",
    "Honor SWEEP_PAUSE: run.sh refuses to start while ~/il-train/SWEEP_PAUSE exists.",
    "Training nodes never serve a rollout policy.",
)

# --- train ---------------------------------------------------------------------


@verb(noun="train", verb="run", tier=OFFLINE, summary="exactly one LeRobot training job, foreground, with the shared invariants",
      role="train", wraps=("scripts/train/run.sh",), passthrough="lerobot-train",
      example=("--", "--help"), doc=".claude/skills/policy-training/SKILL.md", invariants=TRAIN_INV)
def train_run(ctx, ns, rest):
    return sh(ctx, "scripts/train/run.sh", *rest)


def _manifest_args(p):
    p.add_argument("manifest")
    p.add_argument("job")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--render", action="store_true")
    g.add_argument("--execute", action="store_true")


@verb(noun="train", verb="manifest", tier=OFFLINE, summary="render or execute one job from an experiment manifest",
      role="train", wraps=("scripts/train/manifest_job.py",), args=_manifest_args,
      example=("config/training/flagship.json", "n17", "--render"), invariants=TRAIN_INV)
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


@verb(noun="train", verb="offline-eval", tier=OFFLINE, summary="score saved checkpoints on a held-out dataset",
      role="train", wraps=("scripts/train/offline_eval.py",), passthrough="offline_eval.py", example=("--", "--help"))
def train_offline_eval(ctx, ns, rest):
    return _train_python(ctx, "scripts/train/offline_eval.py", *rest)


@verb(noun="train", verb="profile", tier=OFFLINE, summary="this node's training profile (paths, tuning family)",
      wraps=("scripts/train/node_profile.py",), passthrough="node_profile.py", example=())
def train_profile(ctx, ns, rest):
    return py(ctx, "scripts/train/node_profile.py", *rest)


@verb(noun="train", verb="tb-sync", tier=REMOTE, summary="mirror this node's TensorBoard runs to the aggregation node",
      role="train", wraps=("scripts/train/tb_sync.sh",), example=())
def train_tb_sync(ctx, ns, rest):
    return sh(ctx, "scripts/train/tb_sync.sh", *rest)


@verb(noun="train", verb="pause", tier=OFFLINE, summary="create ~/il-train/SWEEP_PAUSE (a rollout owns the GPU)",
      role="train", example=(), invariants=("Do not remove the marker until the rollout owner releases it.",))
def train_pause(ctx, ns, rest):
    marker = gates.train_root() / "SWEEP_PAUSE"
    if ctx.dry_run:
        return Plan(argv=["touch", str(marker)])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    print(f"train: paused — {marker}")
    return EXIT_OK


@verb(noun="train", verb="resume", tier=OFFLINE, summary="remove ~/il-train/SWEEP_PAUSE", role="train", example=())
def train_resume(ctx, ns, rest):
    marker = gates.train_root() / "SWEEP_PAUSE"
    if ctx.dry_run:
        return Plan(argv=["rm", "-f", str(marker)])
    if marker.exists():
        marker.unlink()
        print(f"train: resumed — removed {marker}")
    else:
        print("train: not paused")
    return EXIT_OK


# --- data ----------------------------------------------------------------------

HUB_VERBS = ("push", "pull", "list", "info", "whoami", "set-record", "set-list", "set-pull")
DS_VERBS = ("split", "aggregate", "recompute-stats", "canonicalize", "normalize-task", "feature-view", "validate", "digest")


def _hub_args(p):
    p.add_argument("args", nargs="*")


for _v in HUB_VERBS:
    def _mk(name):
        @verb(noun="data", verb=name, tier=REMOTE if name in ("push", "pull", "set-pull") else OFFLINE,
              summary=f"dataset hub: {name}", wraps=("scripts/dataset_hub.sh", "scripts/dataset_hub.py"),
              passthrough="dataset_hub.py", args=_hub_args, example=("--", "--help"), doc="docs/imitation_learning.md",
              invariants=("Runs dataset_hub.py under the first environment that imports huggingface_hub "
                          "(plugin venv, then ~/il-train), else a throwaway `uv run --with` one.",))
        def _hub(ctx, ns, rest):
            python, why = interp.hub_python(ctx.repo)
            if python is None:
                print(f"data {name}: {why}", file=sys.stderr)
                return EXIT_USAGE
            return Plan(argv=[*python, ctx.path("scripts/dataset_hub.py"), name, *ns.args, *rest], notes=[f"interpreter: {why}"])
        _hub.__name__ = f"data_{name.replace('-', '_')}"
        return _hub
    _mk(_v)

for _v in DS_VERBS:
    def _mk2(name):
        @verb(noun="data", verb=name, tier=OFFLINE, summary=f"dataset tooling: {name}",
              wraps=("scripts/train/dataset.py",), passthrough="train/dataset.py", args=_hub_args,
              example=("--", "--help"), doc="docs/imitation_learning.md")
        def _ds(ctx, ns, rest):
            return py(ctx, "scripts/train/dataset.py", name, *ns.args, *rest)
        _ds.__name__ = f"data_{name.replace('-', '_')}"
        return _ds
    _mk2(_v)


@verb(noun="data", verb="tool-meta", tier=OFFLINE, summary="stamp a recorded dataset with meta/tool.json",
      wraps=("scripts/il_tool_meta.py",), passthrough="il_tool_meta.py", needs_tool=True, example=("--", "--help"),
      doc="docs/tools.md")
def data_tool_meta(ctx, ns, rest):
    return py(ctx, "scripts/il_tool_meta.py", *tool_flag(ctx), *rest)


def _compare_args(p):
    p.add_argument("a")
    p.add_argument("b")


@verb(noun="data", verb="compare", tier=OFFLINE, summary="diff two datasets' feature dicts before aggregating",
      wraps=("scripts/train/compare_features.py",), args=_compare_args, example=("~/ds/a", "~/ds/b"))
def data_compare(ctx, ns, rest):
    return py(ctx, "scripts/train/compare_features.py", ns.a, ns.b, *rest)


# --- sim -----------------------------------------------------------------------

SIM_INV = ("SAPIEN/ManiSkill publish x86_64 wheels only; sim verbs need role `sim`.",
           "Datasets go under ~/tatbot-sim, never inside the repo tree.")


@verb(noun="sim", verb="list", tier=OFFLINE, summary="the named distributions the factory can generate",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/factory.py",), example=(), doc="docs/imitation_learning.md")
def sim_list(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.factory", "--list")


def _gen_args(p):
    p.add_argument("distribution", help="paper-draw | skin-erase | skin-tattoo | body-tattoo (see `sim list`)")


@verb(noun="sim", verb="generate", tier=OFFLINE, summary="generate one named distribution into a LeRobot v3 dataset",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/factory.py", "python/tatbot_sim/src/tatbot_sim/generate.py"),
      passthrough="tatbot_sim.generate (tyro)", args=_gen_args, example=("paper-draw", "--", "--out-dir", "~/tatbot-sim/x", "--num-episodes", "8"),
      invariants=SIM_INV)
def sim_generate(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.factory", ns.distribution, *rest)


def _compile_scenario_args(p):
    p.add_argument("placement", help="placement v4 JSON")


@verb(noun="sim", verb="compile", tier=OFFLINE, summary="compile one Inkmap placement into a replayable posed-body scenario",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/cli.py",),
      passthrough="tatbot_sim.inkmap.cli compile", args=_compile_scenario_args,
      example=("config/inkmap/examples/forearm-placement-v4.json", "--", "--pose", "reclined-left-arm-supported", "--output", "/tmp/forearm.json"),
      invariants=("Offline materialization only; the output is not robot motion authorization.",))
def sim_compile(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.cli", "compile", ns.placement, *rest)


@verb(noun="sim", verb="sample", tier=OFFLINE, summary="materialize a bounded procedural posed-body scenario suite",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/cli.py",),
      passthrough="tatbot_sim.inkmap.cli sample", example=("--", "--output-dir", "~/tatbot-sim/scenarios", "--count", "64"),
      invariants=("Generated data must stay outside the repository.",
                  "Every attempt is recorded; bounded retries fail rather than silently dropping a sample.",
                  "Offline materialization only; scenarios are not robot motion authorization."))
def sim_sample(ctx, ns, rest):
    return uvmod(
        ctx, SIM_PROJECT, "tatbot_sim.inkmap.cli", "sample", *rest,
        env={"TATBOT_TOOL_ID": "lutin-3rl-bugpin"},
    )


def _simscript(name, rel, summary, example=("--", "--help")):
    @verb(noun="sim", verb=name, tier=OFFLINE, summary=summary, role="sim", wraps=(rel,), passthrough=f"{rel.split('/')[-1]} (tyro)",
          example=example, invariants=SIM_INV)
    def _fn(ctx, ns, rest):
        return uvpy(ctx, SIM_PROJECT, rel, *rest)
    _fn.__name__ = f"sim_{name}"
    return _fn


_simscript("preview", "scripts/sim_preview.py", "preview what the factory would generate, no dataset written")
_simscript("rerender", "scripts/sim_rerender.py", "re-render a recorded sim dataset under fresh visual draws")
_simscript("cinematic", "scripts/sim_cinematic.py", "path-traced takes of a distribution for showing outside the lab")
_simscript("audit", "scripts/sim_dataset_audit.py", "what a generated dataset / a night of shards actually is")
_simscript("samples", "scripts/sim_dataset_samples.py", "pull frames and short clips out of an existing dataset")


@verb(noun="sim", verb="reach", tier=OFFLINE, summary="can the fitted tool reach everywhere the randomization sends it",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/audit_reach.py",), passthrough="tatbot_sim.audit_reach", example=("--", "--help"))
def sim_reach(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.audit_reach", *rest)


@verb(noun="sim", verb="viewer", tier=OFFLINE, summary="build a local viewer for a directory of cinematic renders",
      wraps=("scripts/render_viewer.py",), passthrough="render_viewer.py", example=("--", "--help"))
def sim_viewer(ctx, ns, rest):
    return py(ctx, "scripts/render_viewer.py", *rest)
