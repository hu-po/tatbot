"""The three distributions, and the triples that are not one of them.

The factory generates three separate distributions from one core: a ballpoint
drawing on the ruled paper pad, a laser removing ink from the silicone skin,
and a 3RL depositing on that same skin. What keeps them separate is not three
codebases but three preset recipes plus one rule about which (task, tool,
substrate) triples mean anything.

Both halves are silent when wrong, which is why they are tested here:

  * an invalid triple does not crash, it produces a full dataset whose prompts
    describe something other than what the arm did -- a laser "drawing", a
    squiggle tracing grid lines on a blank skin;
  * a preset that drifts from the run it was recovered from changes the
    distribution a dataset was generated under while still calling it by the
    same name.

Pure config logic, no render device:

    cd python/tatbot_sim && uv run python tests/test_distributions.py
"""

from __future__ import annotations

import dataclasses
import os

from tatbot_sim import tasks, tools

BALLPOINT = "lutin-ballpoint-dot"
LASER = "picosecond-laser-pen"
LINER = "lutin-3rl-bugpin"


def _tool(tool_id):
    return tools.registry().load_tool(tool_id, tools.REPO)


def _substrate(tool_id):
    return tools.registry().substrate_for(_tool(tool_id), tools.REPO)


def _refuses(task, tool_id):
    try:
        tasks.validate_task(task, _tool(tool_id), _substrate(tool_id))
    except ValueError as exc:
        return str(exc)
    raise AssertionError(f"{task!r} + {tool_id!r} was accepted")


# ---------------------------------------------------------------- valid triples


def test_the_three_distributions_are_each_a_valid_triple():
    tasks.validate_task("erase", _tool(LASER), _substrate(LASER))
    tasks.validate_task("language", _tool(LINER), _substrate(LINER))
    for task in ("maze", "language", "shapes"):
        tasks.validate_task(task, _tool(BALLPOINT), _substrate(BALLPOINT))


# ------------------------------------------------------------- the field op


def test_erase_with_a_pen_fitted_is_refused():
    for pen in (BALLPOINT, LINER):
        msg = _refuses("erase", pen)
        assert "remove pigment" in msg, msg
        # the hint names a tool that can actually do it
        assert LASER in msg, msg


def test_drawing_with_the_laser_fitted_is_refused():
    """The inverse guard, which did not exist before the validator: a draw
    task with a removal tool stripped a blank sheet for a whole run and only
    said so in a warning printed after the last episode."""
    for task in ("language", "shapes", "maze"):
        msg = _refuses(task, LASER)
        assert "lay pigment down" in msg or "printed ruling" in msg, msg


# ---------------------------------------------------------------- the ruling


def test_squiggles_need_a_ruled_substrate():
    for task in ("maze", "shapes"):
        msg = _refuses(task, LINER)
        assert "printed ruling" in msg, msg


def test_language_scenes_do_not_need_a_ruling():
    tasks.validate_task("language", _tool(LINER), _substrate(LINER))


def test_an_unknown_task_lists_the_known_ones():
    try:
        tasks.validate_task("stipple", _tool(BALLPOINT), _substrate(BALLPOINT))
    except ValueError as exc:
        assert "language" in str(exc) and "erase" in str(exc), exc
    else:
        raise AssertionError("accepted an unknown task")


# ------------------------------------------------------------------ mix shares


def test_only_mix_components_with_a_share_are_validated():
    """A pen-fitted mix run that never samples an erase episode is fine, and
    refusing it because a pen cannot remove would refuse today's default."""
    assert tasks.active_tasks("mix", erase_frac=0.0, squiggle_frac=0.25) == ["maze", "language"]
    assert tasks.active_tasks("mix", erase_frac=0.3, squiggle_frac=0.2) == [
        "erase", "maze", "language",
    ]
    assert tasks.active_tasks("mix", erase_frac=1.0, squiggle_frac=0.0) == ["erase"]
    assert tasks.active_tasks("erase") == ["erase"]


# ------------------------------------------------------- substrate resolution


def test_resolving_for_the_paper_pad_reproduces_the_shipped_paper_ranges():
    """Phase 2 moved these ranges out of the dataclass and into the substrate
    registry. The numbers the paper pad resolves to must be the ones every
    already-generated paper dataset was drawn from, or the move silently
    changed the distribution while keeping its name."""
    from tatbot_sim.config import DRConfig

    dr = DRConfig().resolve_for(_substrate(BALLPOINT))
    assert dr.surface.amplitude_m == (0.0004, 0.0018), dr.surface.amplitude_m
    assert dr.surface.max_slope_rad == (0.005, 0.04), dr.surface.max_slope_rad
    assert dr.surface.feature_m == (0.05, 0.12), dr.surface.feature_m
    assert dr.pad.z_range == (0.0, 0.055), dr.pad.z_range
    assert dr.surface.peak_scale == (0.85, 1.10), dr.surface.peak_scale


def test_the_skin_resolves_to_its_own_rest_height_and_mound():
    """What actually shapes a draped substrate is its mound, not the ripple
    ranges: _sample_skin_shape sends `draped` through drape_height_field, which
    reads peak_scale and never touches amplitude/slope/feature. So the skin
    owns the fields that reach it, and the ones that do not stand at the paper
    pad's values rather than at invented skin ones."""
    from tatbot_sim.config import DRConfig

    skin = _substrate(LASER)
    dr = DRConfig().resolve_for(skin)
    assert dr.pad.z_range == tuple(skin.rest_z_m), dr.pad.z_range
    assert dr.surface.peak_scale == (0.95, 1.00), dr.surface.peak_scale
    # the laser's proven overnight run resolved to exactly this envelope
    assert dr.pad.z_range == (0.0, 0.002), dr.pad.z_range


def test_sheet_wear_follows_the_ruling():
    """Wear variants are ghost strokes drawn over a printed ruling, and the
    code only ever asks for them on the ruled branch (env._load_scene). This
    resolution changes no pixels; it stops run_meta.json from RECORDING wear
    variants for skin runs that never had any -- the laser overnight batch's
    meta claims sheet.enabled=true over a substrate whose sheets are built by
    skin_sheets, which takes no wear argument at all."""
    from tatbot_sim.config import DRConfig

    assert DRConfig().resolve_for(_substrate(BALLPOINT)).sheet.enabled is True
    assert DRConfig().resolve_for(_substrate(LASER)).sheet.enabled is False


def test_an_explicit_override_survives_resolution():
    from tatbot_sim.config import DRConfig, SheetWearDR, SurfaceDR

    dr = DRConfig(
        surface=SurfaceDR(amplitude_m=(0.01, 0.02), max_slope_rad=(0.1, 0.2)),
        sheet=SheetWearDR(enabled=True),
    )
    dr.resolve_for(_substrate(LASER))
    # fiducial_benchmark sets sheet.enabled=False by hand; an override in
    # either direction has to survive, or resolution would silently undo it
    assert dr.surface.amplitude_m == (0.01, 0.02)
    assert dr.surface.max_slope_rad == (0.1, 0.2)
    assert dr.sheet.enabled is True
    # and the leaf that was left alone still came from the substrate
    assert dr.surface.feature_m is not None


def test_resolution_is_idempotent():
    from tatbot_sim.config import DRConfig

    skin = _substrate(LASER)
    once = dataclasses.asdict(DRConfig().resolve_for(skin))
    twice = dataclasses.asdict(DRConfig().resolve_for(skin).resolve_for(skin))
    assert once == twice


# ------------------------------------------------------------------- presets


def test_every_preset_is_a_valid_triple():
    """Including the blocked one: its recipe is written now precisely so its
    numbers can be read before they generate anything, and a recipe that could
    not pass the validator would be a plan to write a mislabelled dataset."""
    from tatbot_sim import distributions

    for dist in distributions.DISTRIBUTIONS.values():
        args = dist.build_args()
        tool, substrate = _tool(dist.tool_id), _substrate(dist.tool_id)
        for task in tasks.active_tasks(args.task, args.erase_frac, args.squiggle_frac):
            tasks.validate_task(task, tool, substrate)


def test_a_blocked_distribution_refuses_to_run_and_says_why():
    """The mechanism, tested against a synthetic entry rather than a real one.

    No distribution is blocked today — skin-tattoo's were waived by the
    operator on 2026-08-27 (the 3RL is the ballpoint's form factor, so it runs
    the ballpoint's settings). The gate stays: the next tool added here will
    arrive without measurements too, and a preset that generates plausible
    data under unvalidated assumptions is worse than one that refuses.
    """
    from tatbot_sim import distributions, factory

    live = distributions.DISTRIBUTIONS["paper-draw"]
    blocked = distributions.Distribution(
        name="paper-draw",  # the name main() will look up
        tool_id=live.tool_id,
        summary=live.summary,
        build_args=live.build_args,
        blockers=("a synthetic blocker, for the test",),
    )
    saved = distributions.DISTRIBUTIONS["paper-draw"]
    distributions.DISTRIBUTIONS["paper-draw"] = blocked
    factory.DISTRIBUTIONS["paper-draw"] = blocked
    try:
        factory.main(["paper-draw", "--out-dir", "/tmp/should-not-be-written"])
    except SystemExit as exc:
        assert "a synthetic blocker" in str(exc), exc
    else:
        raise AssertionError("a blocked distribution ran anyway")
    finally:
        distributions.DISTRIBUTIONS["paper-draw"] = saved
        factory.DISTRIBUTIONS["paper-draw"] = saved


def test_the_tattoo_recipe_is_the_paper_one_on_the_skin():
    """The operator's call, pinned: same settings, different substrate. If
    someone later tunes tattoo ink or pacing for real 3RL data, this test is
    where they should notice they are changing the borrowed-settings claim."""
    from tatbot_sim import distributions

    paper = distributions.DISTRIBUTIONS["paper-draw"].build_args()
    tattoo = distributions.DISTRIBUTIONS["skin-tattoo"].build_args()
    assert tattoo.horizon == paper.horizon
    assert tattoo.dr.ink == paper.dr.ink
    # the one deliberate difference: no squiggles on a blank skin
    assert tattoo.task == "language" and paper.task == "mix"


def test_the_erase_recipe_can_hold_its_longest_episode():
    """erase_seconds samples up to 60 s at 30 Hz = 1800 control steps. A
    horizon under that does not fail, it silently truncates the long erases
    and ships a dataset of shorter ones -- which is the exact failure
    erase_seconds was introduced to fix."""
    from tatbot_sim import distributions

    args = distributions.DISTRIBUTIONS["skin-erase"].build_args()
    assert args.horizon >= args.erase_seconds[1] * 30, (args.horizon, args.erase_seconds)


def test_the_paper_preset_only_departs_from_the_defaults_deliberately():
    """Args grows over time. A preset that silently inherits a new default it
    was never checked against is how a distribution drifts without a commit
    that says so -- so the paper recipe, which IS today's default incantation,
    is pinned to exactly the fields it means to set."""
    from tatbot_sim import distributions, generate

    preset = distributions.DISTRIBUTIONS["paper-draw"].build_args()
    base = generate.Args(out_dir=preset.out_dir)
    differs = {
        f.name for f in dataclasses.fields(generate.Args)
        if getattr(preset, f.name) != getattr(base, f.name)
    }
    assert differs == {"horizon", "distribution"}, differs


def test_a_distribution_names_itself_in_its_dataset():
    from tatbot_sim import distributions

    for name, dist in distributions.DISTRIBUTIONS.items():
        if dist.blockers:
            continue
        assert dist.build_args().distribution == name


def test_the_tool_cannot_be_chosen_after_import_which_is_why_factory_reexecs():
    """The load-bearing fact behind factory.main's os.execv.

    The fitted tool is resolved while the package is being imported -- the
    agent class body and the URDF build both ask for it as they are defined --
    and cached for the process. Importing `tatbot_sim` is enough to fix it, and
    `python -m tatbot_sim.factory` cannot run a line of its own code before
    that import happens. A paper-draw run on a laser-fitted bench therefore
    built the laser's geometry and wrote the laser's prompts (measured
    2026-08-27) until the launcher started setting the variable and
    re-executing.

    If this test ever fails, active_tool() has become late-bound and the
    re-exec can be simplified away. Until then it cannot.
    """
    fitted = tools.active_tool().tool_id
    other = BALLPOINT if fitted != BALLPOINT else LASER
    prior = os.environ.get("TATBOT_TOOL_ID")
    os.environ["TATBOT_TOOL_ID"] = other
    try:
        assert tools.active_tool().tool_id == fitted, (
            "active_tool() now responds to a late TATBOT_TOOL_ID — see the "
            "docstring: factory's re-exec may be removable"
        )
    finally:
        if prior is None:
            os.environ.pop("TATBOT_TOOL_ID", None)
        else:
            os.environ["TATBOT_TOOL_ID"] = prior


def test_the_launcher_refuses_to_fight_an_existing_tool_override():
    """TATBOT_TOOL_ID and --distribution both decide what is in the gripper.
    Whichever quietly won, the other's user would be reading a run that is not
    the one they asked for."""
    from tatbot_sim import factory

    prior = os.environ.get("TATBOT_TOOL_ID")
    os.environ["TATBOT_TOOL_ID"] = LASER
    try:
        try:
            factory.select_tool(factory.DISTRIBUTIONS["paper-draw"])
        except SystemExit as exc:
            assert "TATBOT_TOOL_ID" in str(exc), exc
        else:
            raise AssertionError("a conflicting TATBOT_TOOL_ID was accepted")
        # setting it to the SAME tool the distribution asks for is not a fight
        os.environ["TATBOT_TOOL_ID"] = LASER
        factory.select_tool(factory.DISTRIBUTIONS["skin-erase"])
    finally:
        if prior is None:
            os.environ.pop("TATBOT_TOOL_ID", None)
        else:
            os.environ["TATBOT_TOOL_ID"] = prior


# ------------------------------------------------------------ scene style


def test_sacred_geometry_is_a_construction_not_a_doodle():
    """The flower of life is defined by its lattice: every circle passes
    through its neighbours' centres, which is a triangular lattice of spacing
    equal to the circle radius. Get that wrong and it is a daisy."""
    import numpy as np
    from tatbot_sim.language import MOTIFS

    grid = {"pitch_m": 0.006, "xs": [], "ys": []}
    seed = MOTIFS["seed_of_life"].compile({"r": 0.030, "rot": 0.0}, grid)
    flower = MOTIFS["flower_of_life"].compile({"r": 0.030, "rot": 0.0}, grid)
    assert len(seed) == 7, len(seed)
    assert len(flower) == 19 + 2, len(flower)  # 19 circles + two containing rings

    for strokes in (seed, flower):
        pts = np.concatenate(strokes)
        # `r` is the figure's overall radius for every other motif, so it has
        # to be here too or the sampler cannot reason about placement
        assert abs(np.abs(pts).max() - 0.030) < 1e-3, np.abs(pts).max()

    # neighbouring circle centres sit exactly one circle-radius apart
    centres = np.array([s.mean(axis=0) for s in seed])
    r_circle = np.linalg.norm(seed[0] - centres[0], axis=1).mean()
    d = np.linalg.norm(centres[1:] - centres[0], axis=1)
    assert abs(d.min() - r_circle) < 1e-4, (d.min(), r_circle)


def test_the_clean_style_draws_no_scribble():
    from tatbot_sim.language import CLEAN_STYLE, SCRIBBLY

    assert CLEAN_STYLE.style_prob == 0.0 and CLEAN_STYLE.nest_prob == 0.0
    for key in SCRIBBLY:
        assert key not in CLEAN_STYLE.motifs, key


def test_the_default_style_is_still_the_training_draw():
    """A render's preferences must not leak into the dataset: these are the
    values sample_scene used before SceneStyle existed."""
    from tatbot_sim.language import DEFAULT_STYLE

    assert DEFAULT_STYLE.style_prob == 0.30
    assert DEFAULT_STYLE.nest_prob == 0.18
    assert DEFAULT_STYLE.motifs is None
    assert DEFAULT_STYLE.max_motifs == 3


def test_a_forced_motif_scene_contains_only_that_motif():
    import numpy as np
    from tatbot_sim.language import SceneStyle, sample_scene

    grid = {"pitch_m": 0.006,
            "xs": list(np.arange(-0.06, 0.061, 0.006)),
            "ys": list(np.arange(-0.06, 0.061, 0.006))}
    style = SceneStyle(style_prob=0.0, nest_prob=0.0, max_motifs=1,
                       motifs=("seed_of_life",))
    _, program = sample_scene(np.random.default_rng(5), grid, budget_s=90.0,
                              verb="draw", style=style)
    assert [m["key"] for m in program["motifs"]] == ["seed_of_life"]
    assert program["motifs"][0]["mods"] == []
    assert "seed of life" in program["prompt"] or "rosette" in program["prompt"]


# ---------------------------------------------------------------- tool tips


def test_every_tool_models_its_own_tip():
    """A profile of revolution ends in a cone, and none of the three tools
    does: a ballpoint rides on a ball, a 3RL leaves three needles, a laser has
    an emitter window. The tip is what a macro camera looks at."""
    kinds = {BALLPOINT: "ball", LINER: "needles", LASER: "emitter"}
    for tool_id, kind in kinds.items():
        spec = _tool(tool_id)
        assert spec.tip_detail.get("kind") == kind, (tool_id, spec.tip_detail)
        assert spec.tip_detail_parts(), tool_id


def test_the_ball_is_tangent_to_the_contact_point():
    """The TCP is where ink lands and where the floor clamp stops the arm.
    A ball centred ON it would put half the ball through the paper."""
    spec = _tool(BALLPOINT)
    (ball,) = spec.tip_detail_parts()
    assert ball["kind"] == "sphere"
    assert abs((ball["z"] + ball["radius"]) - spec.protrusion_m) < 1e-9


def test_the_liner_leaves_three_needles_in_a_touching_cluster():
    spec = _tool(LINER)
    parts = spec.tip_detail_parts()
    assert len(parts) == 3
    import math

    offsets = [(p["x"], p["y"]) for p in parts]
    radius = parts[0]["radius"]
    for x, y in offsets:
        assert abs(math.hypot(x, y) - 2 * radius / math.sqrt(3)) < 1e-9
    # every needle ends exactly at the contact point
    for p in parts:
        assert abs((p["z"] + p["length"] / 2) - spec.protrusion_m) < 1e-9


def test_only_the_laser_declares_an_emitter():
    """The pulse is driven off this flag, and a pen that claimed one would
    flash blue while drawing."""
    for tool_id in (BALLPOINT, LINER):
        assert _tool(tool_id).tip_detail.get("kind") != "emitter", tool_id


def test_tip_parts_survive_the_urdf_writers():
    """Both writers consume geometry_parts: the sim's and the real rig's.
    Spheres and x/y offsets are new to both, and a writer that silently
    dropped them would model the tip as nothing at all.

    Deliberately not indexed by position — rings are appended after the tip
    detail, so "the last N parts" is not the tip and asserting it was wrong.
    """
    from tatbot_sim.urdf import tool_visuals

    for tool_id, want in ((BALLPOINT, "sphere"), (LINER, "cylinder"), (LASER, "sphere")):
        spec = _tool(tool_id)
        visuals = tool_visuals(spec)
        assert len(visuals) == len(spec.geometry_parts()), tool_id
        tags = [geom.tag for _xyz, _rpy, geom, _c in visuals]
        assert want in tags, (tool_id, set(tags))
        # a sphere appears ONLY as tip detail, so counting them counts the tip
        if want == "sphere":
            assert tags.count("sphere") == len(spec.tip_detail_parts()), tool_id
        # the liner's needles are the only parts that sit off the axis
        off_axis = [v for v in visuals if abs(v[0][0]) > 1e-9 or abs(v[0][1]) > 1e-9]
        assert len(off_axis) == (3 if tool_id == LINER else 0), tool_id


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
