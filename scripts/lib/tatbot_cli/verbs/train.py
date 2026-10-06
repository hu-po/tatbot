"""train · data · sim — GPU nodes, datasets, and the x86-only sim factory."""

from __future__ import annotations

import os
import sys

from tatbot_cli import EXIT_OK, EXIT_USAGE, gates, interp
from tatbot_cli.registry import MOTION_AUTO, OFFLINE, REMOTE, Plan, verb
from tatbot_cli.verbs._common import SIM_PROJECT, lerobot_py, py, sh, tool_flag, uvmod, uvpy

TRAIN_INV = (
    "One GPU training job per node; run.sh holds ~/il-train/.tatbot-training.lock.",
    "Honor SWEEP_PAUSE: run.sh refuses to start while ~/il-train/SWEEP_PAUSE exists.",
    "Training nodes never serve a rollout policy.",
)


def _travel_args(parser):
    parser.add_argument("--python", help="existing travel/LeRobot interpreter; otherwise use the travel uv project")


def _travel_plan(ctx, ns, command, rest):
    if ns.python:
        return Plan(argv=[ns.python, "-m", "tatbot_travel.cli", command, *rest],
                    env={"PYTHONPATH": ctx.path("python/tatbot_travel/src")})
    return uvmod(ctx, "python/tatbot_travel", "tatbot_travel.cli", command, *rest)


@verb(effects=('read_files', 'write_files', 'start_process', 'gpu'), visibility="public", noun="travel", verb="preview", tier=OFFLINE,
      summary="render a travel episode and optional camera/hand inspection views without a dataset",
      args=_travel_args, passthrough="travel preview", example=("--", "--help"), doc="python/tatbot_travel/README.md")
def travel_preview(ctx, ns, rest):
    return _travel_plan(ctx, ns, "preview", rest)


@verb(effects=('read_files', 'write_files', 'start_process', 'gpu'), visibility="public", noun="travel", verb="generate", tier=OFFLINE,
      summary="generate travel LeRobot episodes with canonical SOMA surface-address labels",
      args=_travel_args, passthrough="travel generate", example=("--", "--help"), doc="python/tatbot_travel/README.md")
def travel_generate(ctx, ns, rest):
    return _travel_plan(ctx, ns, "generate", rest)

def _trace_args(parser):
    parser.add_argument("--python", help="interpreter with trossen_arm, MuJoCo and the travel dependencies "
                                         "(default: the arm node's ~/travel/.venv-jal)")


@verb(effects=('read_files', 'write_files', 'start_process', 'autonomous_motion', 'sensor_read'), noun="travel",
      verb="trace", tier=MOTION_AUTO, role="arm", visibility="public",
      summary="scan the practice forearm with the blue wrist camera and follow a stroke of ink at the laser's hover",
      args=_trace_args, passthrough="trace", example=("--", "scan"), doc="python/tatbot_travel/README.md",
      tool_arm="left")
def travel_trace(ctx, ns, rest):
    python = ns.python or os.path.expanduser("~/travel/.venv-jal/bin/python")
    return Plan(argv=[python, "-m", "tatbot_travel.trace", *rest],
                env={"PYTHONPATH": ctx.path("python/tatbot_travel/src")})


# --- train ---------------------------------------------------------------------


@verb(effects=('read_files', 'write_files', 'network', 'start_process', 'gpu'), noun="train", verb="run", tier=OFFLINE, summary="exactly one LeRobot training job, foreground, with the shared invariants",
      role="train", wraps=("scripts/train/run.sh",), passthrough="lerobot-train",
      example=("--", "--help"), doc="docs/cli.md", invariants=TRAIN_INV)
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


# --- sim -----------------------------------------------------------------------
#
# Each sim verb wraps one tatbot_sim module or script in the sim environment and
# answers one question; `tatbot sim --help-all` lists them.

SIM_INV = ("SAPIEN/ManiSkill publish x86_64 wheels only; sim verbs need role `sim`.",
           "Datasets go under ~/tatbot-sim, never inside the repo tree.")
BODY_TATTOO_TOOL = "lutin-3rl-bugpin"


def _workspace_tool_id(ctx) -> str | None:
    """The datasheet config/workspace.yaml says is fitted (what tatbot_sim.tools.active_tool falls back to)."""
    try:
        import tool_spec
        return tool_spec.active_tool_id(ctx.repo)
    except Exception:  # noqa: BLE001 — a note, never a gate
        return None


def _sim_tool_env(ctx) -> tuple[dict[str, str], list[str]]:
    """TATBOT_TOOL_ID for a sim run that asks about the tool: the stated --ee-tool,
    else nothing — tatbot_sim then reads the fitted tool from config/workspace.yaml,
    and the plan says so instead of hard-coding a tool id (the old `sim sample`
    pinned lutin-3rl-bugpin regardless of what was stated)."""
    if ctx.ee_tool:
        return {"TATBOT_TOOL_ID": ctx.ee_tool}, []
    fitted = _workspace_tool_id(ctx)
    return {}, [f"no --ee-tool stated: tatbot_sim uses the fitted tool from config/workspace.yaml "
                f"({fitted or 'none recorded'}); state --ee-tool <id> to run for another tool"]


def _sim_input_owns_tool(path: str) -> bool:
    """Recognize the strict local bundle without resolving caller-owned paths."""
    try:
        import json
        from pathlib import Path

        source = Path(path)
        if not source.is_file() or source.stat().st_size > 20_000_000:
            return False
        value = json.loads(source.read_text())
        return isinstance(value, dict) and value.get("schema") == "tatbot.inkmap-sim-bundle/1"
    except (OSError, UnicodeError, ValueError):
        return False


def _body_tattoo_tool_env(ctx, command: str) -> tuple[dict[str, str], list[str]] | None:
    """Declare the body distribution's nominal tool without consulting calibration."""
    if ctx.ee_tool and ctx.ee_tool != BODY_TATTOO_TOOL:
        print(
            f"{command} builds the body-tattoo distribution with "
            f"{BODY_TATTOO_TOOL}; got --ee-tool {ctx.ee_tool}",
            file=sys.stderr,
        )
        return None
    return (
        {"TATBOT_TOOL_ID": BODY_TATTOO_TOOL},
        [
            f"body-tattoo distribution declares tool {BODY_TATTOO_TOOL}; "
            "nominal geometry is advisory in offline sim"
        ],
    )


@verb(effects=('read_files',), visibility="public", noun="sim", verb="list", tier=OFFLINE, summary="the named distributions the factory can generate",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/factory.py",), example=(), doc="docs/imitation_learning.md")
def sim_list(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.factory", "--list")


def _gen_args(p):
    p.add_argument("distribution", help="paper-draw | skin-erase | skin-tattoo | body-tattoo (see `sim list`)")


@verb(effects=('read_files', 'write_files', 'network', 'start_process', 'gpu'), visibility="public", noun="sim", verb="generate", tier=OFFLINE, summary="generate one named distribution into a LeRobot v3 dataset",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/factory.py", "python/tatbot_sim/src/tatbot_sim/generate.py"),
      passthrough="tatbot_sim.generate (tyro)", args=_gen_args, example=("paper-draw", "--", "--out-dir", "~/tatbot-sim/x", "--num-episodes", "8"),
      invariants=SIM_INV)
def sim_generate(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.factory", ns.distribution, *rest, extra="maniskill")


def _compile_args(p):
    p.add_argument("placement", help="placement v5 JSON or immutable Inkmap simulation bundle")


def _sample_args(p):
    p.add_argument("--count", type=int, required=True, metavar="N",
                   help="bounded artwork scenario count; backend options follow --")


@verb(effects=('read_files', 'write_files'), visibility="public", noun="sim", verb="sample", tier=OFFLINE,
      summary="materialize a bounded artwork suite of posed-body scenarios",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/cli.py",),
      passthrough="tatbot_sim.inkmap.cli sample", args=_sample_args,
      example=("--count", "3", "--", "--output-dir", "/tmp/body-suite"), doc="docs/simulation.md",
      invariants=("Offline materialization only; the output is not robot motion authorization.",
                  "Generated suites stay outside the repository; every attempt is recorded, and bounded retries fail rather than silently dropping a sample.",
                  f"Declares the body-tattoo tool {BODY_TATTOO_TOOL}; nominal geometry is advisory, not calibration-gated."))
def sim_sample(ctx, ns, rest):
    tool_run = _body_tattoo_tool_env(ctx, "sim sample")
    if tool_run is None:
        return EXIT_USAGE
    env, notes = tool_run
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.cli", "sample", "--count", str(ns.count), *rest, env=env, notes=notes)


@verb(effects=('read_files', 'write_files'), visibility="public", noun="sim", verb="compile", tier=OFFLINE,
      summary="compile one Inkmap placement into a replayable posed-body scenario",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/cli.py",),
      passthrough="tatbot_sim.inkmap.cli compile", args=_compile_args,
      example=("config/inkmap/examples/forearm-placement-v6.json", "--", "--pose", "reclined-left-arm-supported", "--output", "/tmp/forearm.json"),
      invariants=("Offline materialization only; the output is not robot motion authorization.",
                  "Placement files use the stated --ee-tool, otherwise the fitted tool in config/workspace.yaml; immutable bundles use their declared tool unless --ee-tool is explicitly stated."))
def sim_compile(ctx, ns, rest):
    env, notes = _sim_tool_env(ctx)
    tool: list[str] = []
    if "--tool" not in rest and (ctx.ee_tool or not _sim_input_owns_tool(ns.placement)):
        # inkmap/cli.py defaults --tool to a hard-coded id and never reads TATBOT_TOOL_ID,
        # so the fitted tool has to travel explicitly when none was stated.
        chosen = ctx.ee_tool or _workspace_tool_id(ctx)
        if chosen:
            tool = ["--tool", chosen]
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.cli", "compile", ns.placement, *tool, *rest, env=env, notes=notes)


def _perception_args(p):
    p.add_argument("scenario", help="compiled typed Inkmap v3 scenario JSON")


@verb(effects=('read_files', 'write_files'), visibility="public", noun="sim", verb="perception", tier=OFFLINE,
      summary="render audited privileged Inkmap perception labels on CPU",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/perception_dataset.py",),
      passthrough="tatbot_sim.inkmap.cli perception", args=_perception_args,
      example=("/tmp/scenario-v3.json", "--", "--output-dir", "/tmp/perception"),
      invariants=("Only typed v3 scenarios and visually admitted identities are accepted.",
                  "Dense labels are privileged sidecars, never deployed-policy observation fields.",
                  "Production scale remains blocked until throughput, storage, and visual-review gates pass."))
def sim_perception(ctx, ns, rest):
    return uvmod(
        ctx,
        SIM_PROJECT,
        "tatbot_sim.inkmap.cli",
        "perception",
        ns.scenario,
        *rest,
    )


def _perception_audit_args(p):
    p.add_argument("path", help="perception corpus directory to audit")


@verb(effects=('read_files',), visibility="public", noun="sim", verb="perception-audit", tier=OFFLINE,
      summary="audit an existing privileged Inkmap perception corpus",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/perception_audit.py",),
      passthrough="tatbot_sim.inkmap.perception_audit", args=_perception_audit_args,
      example=("/tmp/inkmap-perception",),
      invariants=("Missing frames, hashes, provenance, identity approval, split integrity, and rejection labels fail closed.",))
def sim_perception_audit(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.perception_audit", ns.path, *rest)


@verb(effects=('read_files', 'write_files'), visibility="public", noun="sim", verb="pilot-plan", tier=OFFLINE,
      summary="write and audit the full Inkmap synthetic-pilot ledger",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/pilot.py", "config/inkmap/synthetic-pilot.json"),
      passthrough="tatbot_sim.inkmap.cli pilot-plan", example=("--", "--output-dir", "/tmp/inkmap-pilot", "--seed", "42"),
      invariants=("Planning is offline and does not launch a renderer, GPU job, robot, or generation service.",
                  "Unreviewed identities, unassigned compute, reach/clearance, stencil, GPU, and human-review gates remain explicit ledger entries.",
                  "No unrun episode receives a success label."))
def sim_pilot_plan(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.cli", "pilot-plan", *rest)


def _pilot_audit_args(p):
    p.add_argument("plan", help="pilot-plan.json to audit")


@verb(effects=('read_files',), visibility="public", noun="sim", verb="pilot-audit", tier=OFFLINE,
      summary="audit an existing Inkmap synthetic-pilot ledger",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/pilot.py",),
      passthrough="tatbot_sim.inkmap.cli pilot-audit", args=_pilot_audit_args,
      example=("/tmp/inkmap-pilot/pilot-plan.json",),
      invariants=("Read-only offline audit; no renderer, GPU job, robot, or generation service is started.",
                  "Digest drift, unreviewed identity admission, missing cells/artworks, and premature success labels fail closed."))
def sim_pilot_audit(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.cli", "pilot-audit", ns.plan, *rest)


@verb(effects=('read_files', 'write_files', 'start_process'), visibility="public", noun="sim", verb="parity", tier=OFFLINE,
      summary="produce browser/simulator Inkmap mapping and mask parity evidence",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/parity_evidence.py", "web/inkmap/tools/surface_parity.ts"),
      passthrough="tatbot_sim.inkmap.parity_evidence", example=("--", "--output", "/tmp/inkmap-parity"),
      invariants=("Automated mapping/mask thresholds do not fill the human or GPU rendered-scene review cells.",))
def sim_parity(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.parity_evidence", *rest)


@verb(effects=('read_files', 'write_files', 'network'), visibility="public", noun="sim", verb="materialize", tier=OFFLINE,
      summary="generate and trace immutable Inkgen SVG artifacts before scenario compilation",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/cli.py", "web/inkgen/app.py"),
      passthrough="tatbot_sim.inkmap.cli materialize-designs", example=("--", "--help"),
      invariants=("This is the only sim stage allowed to call Inkgen; reset, step, compile, and generate consume local SVG bytes only.",
                  "Artifacts stay outside the repository and include exact SVG/PNG hashes and generation provenance."))
def sim_materialize(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.cli", "materialize-designs", *rest)


@verb(effects=('read_files', 'write_files'), visibility="public", noun="sim", verb="recipes", tier=OFFLINE,
      output="json", summary="expand a frozen artwork library into reproducible scenario recipes",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/recipes.py",),
      passthrough="tatbot_sim.inkmap.cli recipes", example=("--", "--help"),
      doc="docs/simulation.md",
      invariants=("Offline: no generator, no compiler, no renderer and no network is involved.",
                  "Recipes are pure functions of the frozen plan and their index, so sharding, resuming or reordering a run produces identical files.",
                  "Artwork family and identity splits are assigned before augmentation; a rotated or rescaled variant cannot change a split.",
                  "A recipe is not a render and a render is not a drawing: compiled/rendered/executed stay zero here."))
def sim_recipes(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.cli", "recipes", *rest)


@verb(effects=('read_files',), visibility="public", noun="sim", verb="recipes-status", tier=OFFLINE,
      output="json", summary="read a recipe ledger and audit its splits without any network",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/recipes.py",),
      passthrough="tatbot_sim.inkmap.cli recipes-status", example=("--", "--help"),
      doc="docs/simulation.md",
      invariants=("Read-only and offline.",
                  "Counts distinguish requested, admitted, rejected, compiled, rendered and executed."))
def sim_recipes_status(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.cli", "recipes-status", *rest)


def _resolve_args(p):
    p.add_argument("prompt", help='typed InkLang sentence, e.g. "a heron on the left forearm"')


@verb(effects=('read_files', 'write_files'), visibility="public", noun="sim", verb="resolve", tier=OFFLINE,
      summary="resolve a typed InkLang request into one replayable posed-body scenario",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/cli.py", "web/inkmap/src/core/lang.ts"),
      passthrough="tatbot_sim.inkmap.cli resolve", args=_resolve_args,
      example=("a dbv3-orbit on the left forearm", "--", "--design-id", "dbv3-orbit", "--size-mm", "30", "30", "--seed", "42", "--output-dir", "/tmp/tatbot-resolved"),
      invariants=("The canonical deterministic InkLang parser produces a typed request; free text never writes scenario JSON.",
                  "Normal designs come from the frozen shared collection or complete materializations; spiral-v1 is the calibration control.",
                  f"The body-tattoo distribution declares {BODY_TATTOO_TOOL}; nominal geometry warns but never opens a calibration gate.",
                  "Resolution is offline, bounded, and produces a complete rejection ledger."))
def sim_resolve(ctx, ns, rest):
    tool_run = _body_tattoo_tool_env(ctx, "sim resolve")
    if tool_run is None:
        return EXIT_USAGE
    env, notes = tool_run
    return uvmod(
        ctx, SIM_PROJECT, "tatbot_sim.inkmap.cli", "resolve", ns.prompt,
        *rest, env=env, notes=notes,
    )


def _no_args(p):
    """A structured adapter whose backend options require an explicit --."""


@verb(effects=('read_files', 'write_files', 'start_process'), visibility="public", noun="sim", verb="qualify-artwork", tier=OFFLINE,
      summary="qualify shared artwork paint, duration, and simulated deposition", role="sim",
      wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/artwork_qualification.py",),
      passthrough="tatbot_sim.inkmap.artwork_qualification", args=_no_args,
      example=("--", "--output-dir", "/tmp/artwork-qualification"), doc="docs/simulation.md",
      invariants=("Every artwork and rejected size remains in the report; evaluation families are disjoint.",))
def sim_qualify_artwork(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.artwork_qualification", *rest)


def _sim_dataset_args(p):
    p.add_argument("dataset", nargs="+", help="sim dataset directories generated with --judge")


EVAL_INV = (
    "Input datasets must be generated with --judge and retain intended/drawn/overlay evidence.",
    "Policy evaluation keeps ManiSkill, the LeRobot client, and the policy server in separate environments; local IPC is versioned JSON plus typed arrays, never pickle.",
    "Policy observations come from the Tatbot follower's live feature declaration; nominal sim tool geometry warns but never opens a calibration gate.",
    "Training-seed overlap, dirty source, or fewer than three episodes blocks comparison.",
)


@verb(effects=('read_files', 'write_files'), visibility="public", noun="sim", verb="eval dataset", tier=OFFLINE,
      summary="score judged expert datasets and write a screen report", role="sim",
      wraps=("python/tatbot_sim/src/tatbot_sim/eval_cli.py",),
      passthrough="tatbot_sim.eval_cli", args=_sim_dataset_args,
      example=("/tmp/tatbot-eval-dataset", "--", "--output-dir", "/tmp/tatbot-eval-report"),
      doc="docs/simulation.md", invariants=EVAL_INV)
def sim_eval_dataset(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.eval_cli", *ns.dataset, *rest)


@verb(effects=('read_files', 'write_files', 'network', 'start_process', 'gpu'), visibility="public", noun="sim", verb="eval policy", tier=OFFLINE,
      summary="run a checkpoint through a local ManiSkill worker and the async wire", role="sim",
      wraps=("scripts/eval/sim_policy_eval.py",), passthrough="sim_policy_eval.py", args=_no_args,
      example=("--", "--client-mode", "hold-control", "--output-dir", "/tmp/hold-report"),
      doc="docs/simulation.md", invariants=EVAL_INV)
def sim_eval_policy(ctx, ns, rest):
    mode = "policy"
    for index, token in enumerate(rest):
        if token.startswith("--client-mode="):
            mode = token.partition("=")[2]
        elif token == "--client-mode" and index + 1 < len(rest):
            mode = rest[index + 1]
    if mode == "hold-control":
        return uvpy(ctx, SIM_PROJECT, "scripts/eval/sim_policy_eval.py", *rest, extra="maniskill")
    return lerobot_py(ctx, "scripts/eval/sim_policy_eval.py", *rest)


@verb(effects=('read_files', 'write_files', 'network', 'start_process', 'gpu'), visibility="public", noun="sim", verb="preview", tier=OFFLINE,
      summary="preview what the factory would generate, no dataset written",
      role="sim", wraps=("scripts/sim_preview.py",),
      passthrough="sim_preview.py (tyro)", args=_no_args, example=("--", "--help"), invariants=SIM_INV)
def sim_preview(ctx, ns, rest):
    return uvpy(ctx, SIM_PROJECT, "scripts/sim_preview.py", *rest, extra="maniskill")


@verb(effects=('read_files', 'write_files', 'network', 'start_process', 'gpu'), visibility="public", noun="sim", verb="reach", tier=OFFLINE,
      summary="audit tool IK reach over the domain-randomization distribution, headless",
      role="sim", wraps=("python/tatbot_sim/src/tatbot_sim/audit_reach.py",),
      passthrough="tatbot_sim.audit_reach (tyro)", args=_no_args, example=("--", "--help"), invariants=SIM_INV)
def sim_reach(ctx, ns, rest):
    env, notes = _sim_tool_env(ctx)
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.audit_reach", *rest, env=env, notes=notes)


@verb(effects=('read_files', 'write_files', 'network', 'start_process', 'gpu'), visibility="public", noun="sim", verb="cinematic", tier=OFFLINE,
      summary="path-traced takes of a distribution for showing outside the lab",
      role="sim", wraps=("scripts/sim_cinematic.py",),
      passthrough="sim_cinematic.py (tyro)", args=_no_args, example=("--", "--help"), invariants=SIM_INV)
def sim_cinematic(ctx, ns, rest):
    return uvpy(ctx, SIM_PROJECT, "scripts/sim_cinematic.py", *rest, extra="maniskill")


def _viewer_args(p):
    p.add_argument("directory", nargs="+", metavar="DIR", help="render directories to include in the HTML viewer")


@verb(effects=('read_files', 'write_files'), visibility="public", noun="sim", verb="viewer", tier=OFFLINE,
      summary="build an HTML viewer over render directories with system Python",
      wraps=("scripts/render_viewer.py",), passthrough="render_viewer.py", args=_viewer_args,
      example=("/tmp/render-one", "/tmp/render-two"), doc="docs/simulation.md",
      invariants=("Builds local HTML only; requires no simulator role, GPU or simulator installation.",))
def sim_viewer(ctx, ns, rest):
    return py(ctx, "scripts/render_viewer.py", *ns.directory, *rest)


@verb(effects=('read_files', 'write_files'), visibility="public", noun="sim", verb="audit", tier=OFFLINE,
      summary="audit a generated dataset or a night of shards",
      role="sim", wraps=("scripts/sim_dataset_audit.py",),
      passthrough="sim_dataset_audit.py (tyro)", args=_no_args, example=("--", "--help"), invariants=SIM_INV)
def sim_audit(ctx, ns, rest):
    return uvpy(ctx, SIM_PROJECT, "scripts/sim_dataset_audit.py", *rest)


def _samples_args(p):
    p.add_argument("dataset", help="dataset or shard directory to extract stills and clips from")


@verb(effects=('read_files', 'write_files'), visibility="public", noun="sim", verb="samples", tier=OFFLINE,
      summary="extract stills and short clips from a dataset or shards",
      role="sim", wraps=("scripts/sim_dataset_samples.py",),
      passthrough="sim_dataset_samples.py (tyro)", args=_samples_args,
      example=("/tmp/dataset", "--", "--out", "/tmp/samples"), invariants=SIM_INV)
def sim_samples(ctx, ns, rest):
    return uvpy(ctx, SIM_PROJECT, "scripts/sim_dataset_samples.py", "--path", ns.dataset, *rest)


@verb(effects=('read_files', 'write_files', 'start_process'), visibility="public", noun="sim", verb="showcase-artwork", tier=OFFLINE,
      summary="materialize five typed artwork previews for the Inkmap pose gallery", role="sim",
      wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/showcase.py",),
      passthrough="tatbot_sim.inkmap.showcase", args=_no_args,
      example=("--", "--output-dir", "/tmp/artwork-showcase"), doc="docs/simulation.md",
      invariants=("Shared artwork bytes and named body poses are immutable inputs; output is outside the checkout.",
                  "Preview generation does not claim robot reach or human review."))
def sim_showcase_artwork(ctx, ns, rest):
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.showcase", *rest)
