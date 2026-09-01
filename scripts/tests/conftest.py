"""Collection guard: the suite must run from a bare clone on any node.

Nine modules in here import a heavy optional dependency (directly, or through
the script they exercise). Without a guard they raise at *collection* time, and
one missing wheel takes down the whole run — `pytest -q scripts/tests/` reported
"9 errors during collection" and skipped every other test in the directory,
including ones that need nothing but the standard library.

Collection errors are indistinguishable from a red suite at a glance, which is
how the command printed in AGENTS.md stayed broken without anyone noticing. So
a module whose dependency is absent is *ignored and named* instead: the run goes
green on what it could actually check, and the header says what it could not.

Install the full set to run everything (see `scripts/tests/requirements.txt`):

    uvx --with-requirements scripts/tests/requirements.txt pytest -q scripts/tests/

or just `scripts/check tests`, which does that for you.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path

# test module stem -> every import that must resolve for it to be collectable.
# Lists, not single names: a module can clear its first missing import and hit
# another one behind it. Mapping only the first is how this table was wrong on
# its first draft -- installing opencv/pyarrow/torch simply moved the collection
# error to `lerobot` and `tensorboard`.
#
# Keep in sync with scripts/tests/requirements.txt. `lerobot` is deliberately
# NOT in that file: it is an entire robotics stack pulled from a git fork with
# a pinned trossen override, and dragging it into this suite's throwaway
# environment is not worth it. The three modules that need it are collected
# when the suite is run from the plugin's own venv, and skipped here.
OPTIONAL_DEPS: dict[str, tuple[str, ...]] = {
    "test_calibrate_board_session": ("cv2",),
    "test_ee_fiducial": ("cv2",),
    "test_ee_tracker_io": ("cv2",),
    "test_fiducial_inventory": ("cv2",),
    "test_il_analyze_rollout": ("numpy",),
    "test_ink_hook": ("numpy",),
    "test_palette_cal": ("numpy",),
    "test_train_dataset": ("pyarrow", "lerobot"),
    "test_train_depth_encoding": ("pyarrow", "lerobot"),
    "test_train_offline_eval": ("torch", "lerobot"),
    "test_train_sampler": ("torch",),
    "test_train_tb_bridge": ("torch", "tensorboard"),
}


def _absent(mods: tuple[str, ...]) -> list[str]:
    return [m for m in mods if find_spec(m) is None]


_missing = {stem: absent for stem, mods in OPTIONAL_DEPS.items() if (absent := _absent(mods))}

collect_ignore = [f"{stem}.py" for stem in _missing]

# --- deployment-configured tests --------------------------------------------
# Some modules exercise behaviour that only exists once a deployment has been
# described: its fleet (config/nodes.json), its hardware profile, its measured
# arm goldens, its ink/palette inventory. A public checkout has none of those,
# and a test that cannot run is a SKIP, not a failure — the same contract
# scripts/check keeps for a node that lacks a toolchain.
DEPLOYMENT_FILES: dict[str, tuple[str, ...]] = {
    "test_audio_route": ("config/audio/ee-input1.asoundrc",),
    "test_cli": ("config/nodes.json",),
    "test_ee_fiducial": ("config/fiducials.json",),
    "test_estop_launchers": ("config/nodes.json",),
    "test_eval_checkpoint_contract": ("config/nodes.json",),
    "test_fiducial_inventory": ("config/fiducials.json",),
    "test_fuse_session": ("config/fiducials.json",),
    "test_il_dip": ("config/inks.yaml", "config/palette.yaml"),
    "test_ink_hook": ("config/inks.yaml", "config/palette.yaml"),
    "test_ink_spec": ("config/inks.yaml", "config/palette.yaml"),
    "test_profile": ("config/profiles/tatbot.json",),
    "test_runlog": ("config/nodes.json",),
    "test_staged_pose_single_source": ("config/trossen/tatbot.yaml",),
    "test_tool_spec": ("config/workspace.yaml",),
}

_repo_root = Path(__file__).resolve().parents[2]
_unconfigured = {
    stem: absent
    for stem, files in DEPLOYMENT_FILES.items()
    if (absent := [f for f in files if not (_repo_root / f).is_file()])
}
collect_ignore += [f"{stem}.py" for stem in _unconfigured]


# Fiducial tests parse an inventory; the deployment's own (config/fiducials.json)
# is not in a public checkout, so fall back to the synthetic example. This is a
# TEST-ONLY fallback: runtime never substitutes an example for measured data.
_repo = Path(__file__).resolve().parents[2]
if not (_repo / "config" / "fiducials.json").is_file():
    os.environ.setdefault(
        "TATBOT_FIDUCIAL_CONFIG", str(_repo / "config" / "examples" / "fiducials.json"))



def pytest_report_header(config) -> str | None:  # noqa: ANN001 - pytest hook signature
    """Name what was not checked, so a green run is never mistaken for a full one."""
    lines = []
    if _unconfigured:
        files = sorted({f for absent in _unconfigured.values() for f in absent})
        lines.append(
            f"tatbot: skipping {len(_unconfigured)} module(s) that need a configured "
            f"deployment ({', '.join(files)}) -- see config/examples/")
    if not _missing:
        return "\n".join(lines) or None
    by_dep: dict[str, list[str]] = {}
    for stem, mods in sorted(_missing.items()):
        for mod in mods:
            by_dep.setdefault(mod, []).append(stem)
    parts = [f"{mod} ({len(stems)})" for mod, stems in sorted(by_dep.items())]
    hint = "see scripts/tests/requirements.txt"
    if all("lerobot" in mods for mods in _missing.values()):
        # requirements.txt deliberately omits lerobot; pointing at it would be
        # advice that cannot be followed.
        hint = "run from python/lerobot_robot_tatbot/.venv to include these"
    lines.append(
        f"tatbot: skipping {len(_missing)} module(s), missing {', '.join(parts)} -- {hint}"
    )
    return "\n".join(lines)
