"""The three datasets this factory generates, as named recipes.

One robot, one bench, one set of cameras — and three distributions of data
that share almost everything and differ in ways that cannot be mixed: the
ballpoint draws on the ruled paper pad, the laser removes ink from the
silicone skin, the 3RL deposits on that same skin. The laser never touches
paper and the ballpoint never touches skin, so a tool swap is a scene swap
(see tatbot_sim.tools.active_substrate) and a task swap on top of it.

Before this module those three recipes existed only as incantations — an
environment variable, a task flag, an episode-length flag and a couple of DR
overrides that had to be remembered in the right combination, and were
recoverable only by finding the run that worked and reading its run_meta.json
back. A dataset could not say which distribution it belonged to, because
nothing had named them.

A recipe is a preset ``generate.Args``, not a config file: config/*.yaml
records physical facts (what tools and substrates ARE), while the DR tree is
seventeen nested dataclasses whose leaves tyro already exposes. Writing a
YAML overlay for them would be a second config system that has to be kept in
step with the first. ``tyro.cli(Args, default=preset)`` keeps every leaf
overridable from the command line, so a preset is a starting point rather
than a cage.

Numbers here are RECOVERED FROM THE RUNS THAT WORKED, not chosen: each
dataset dumps its fully resolved config into meta/run_meta.json, so the
provenance of a preset is a dataset on disk. Where a value came from is
recorded next to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

BALLPOINT = "lutin-ballpoint-dot"
LASER = "picosecond-laser-pen"
LINER = "lutin-3rl-bugpin"


@dataclass(frozen=True)
class Distribution:
    """One named dataset recipe: what is in the gripper, and what it does."""

    name: str
    """The id a dataset records and a training mix filters on."""

    tool_id: str
    """Sets TATBOT_TOOL_ID. The substrate follows from the tool's datasheet —
    naming it separately would let the two disagree."""

    summary: str
    """One line, for --list."""

    build_args: Callable[[], "object"]
    """Returns the preset generate.Args. A callable because importing generate
    pulls in the whole render stack, and the launcher has to set the tool
    before that import happens."""

    blockers: tuple[str, ...] = ()
    """Why this distribution cannot be generated yet. A preset that produces
    plausible-looking data under unvalidated assumptions is worse than one
    that refuses: the data looks like the other two and trains like neither."""

    tip_calibration_jitter: bool = False
    """Sample one session-persistent TCP offset before the factory re-execs.
    Only a distribution whose fitted tool has a qualified pivot calibration
    can enable this; the measured uncertainty sets the radius."""


def _paper_draw():
    from tatbot_sim.generate import Args

    return Args(
        out_dir="",
        distribution="paper-draw",
        tool_calibration_jitter=True,
        # the sim node's gen/{final,prod,trim,notrim} runs: every shipped paper batch ran
        # mix at horizon 900 with the default squiggle share.
        task="mix",
        # 1800, not 900: pacing is sampled from the real fm2 teleop band
        # (draw p10-90 = 0.9-20.7 mm/s, ~3x slower than the old sim), so the
        # same motifs take up to ~60 s. 900 would cut the slow half of the
        # distribution mid-stroke and ship a dataset of fast episodes.
        horizon=1800,
    )
    # NOTE 2026-08-31 pm: an earlier same-day revision pinned the pad to a
    # shrunken (0.22, 0.04) x z (0, 0.010) envelope. That was an artifact of
    # planning the TIP axis vertical with a crooked measured seat; the real
    # fm2 recording showed the operator holds the BORE vertical, expert.py now
    # plans that orientation, and the stock placement solves at 0.0 mm over
    # the full height range again — so the overrides are gone, not tuned.


def _skin_erase():
    # No pad.z_range override, though the hand-run version of this recipe
    # carried one: the laser stands ~130 mm proud and needs the skin low, and
    # since 2026-08-26 the substrate itself says so (silicon_skin rest_z_m
    # [0.000, 0.002]). The overnight run's reach check printed "over pad z
    # (0.0, 0.002)" at a 0.00 mm residual, so what used to be a flag someone
    # had to remember is now a property of the object.
    from tatbot_sim.generate import Args

    return Args(
        out_dir="",
        distribution="skin-erase",
        task="erase",
        # 2000, not the 900 the other distributions use. erase_seconds samples
        # 28-60 s of episode and the control rate is 30 Hz, so a 60 s episode
        # is 1800 steps: a 900 horizon would cut the longest erases in half
        # and the dataset would look like a distribution of short ones.
        # Recovered from the overnight run driver, the 128-episode laser-rgbd-overnight
        # batch behind config/training/gr00t-n17-laser-rgb-vs-rgbd-20260827.json.
        horizon=2000,
    )


def _skin_tattoo():
    """Deposit on the skin, running the ballpoint's settings.

    OPERATOR CALL, 2026-08-27: the 3RL is basically the ballpoint's form
    factor -- same Lutin body, same grip, tips within a millimetre (+59 mm
    against +60 mm) -- so rather than hold this distribution back for
    measurements it does not have, it runs the paper recipe's settings on the
    skin. Every leaf below is either the shared default (which IS the
    ballpoint's, since that is the tool they were tuned on) or forced by the
    substrate.

    What that borrows, and what it therefore owes:

    - **Ink appearance is the ballpoint's.** Blue-black on white paper, 2.2-4
      mm wide. Black liner ink worked into pink silicone is darker, finer, and
      builds over repeated passes rather than landing at once. Until real 3RL
      frames exist to calibrate against -- the way LightingDR was calibrated
      against the bench -- these episodes look like a biro drawing on a skin.
    - **Pacing is the paper recipe's**, not measured from real 3RL work. The
      erase recipe exists because sampled pacing came out 3x too fast against
      real laser recordings; the same correction has not been made here.
    - **The tip is the datasheet's**, not touched off. Fine for sim, which
      welds at the datasheet protrusion by design -- but nothing here has
      earned the right to steer the arm.

    None of that stops the sim from producing episodes; it is what a reader of
    the resulting dataset needs to know before treating it as tattoo data.
    """
    from tatbot_sim.generate import Args

    return Args(
        out_dir="",
        distribution="skin-tattoo",
        # Scenes, never squiggles: a squiggle traces the printed 6 mm ruling
        # and the skin is blank. tatbot_sim.tasks refuses the combination, so
        # this is the paper recipe's language half with its maze half dropped.
        task="language",
        # follows the paper recipe (the borrow this whole preset is): pacing
        # is sampled from the real fm2 band, so the same motifs take longer.
        horizon=1800,
    )


def _body_tattoo():
    from tatbot_sim.generate import Args

    return Args(
        out_dir="",
        distribution="body-tattoo",
        task="language",
        horizon=1800,
        # A compiled scenario is immutable, so parallel slots are repeated
        # visual/ink draws rather than pretending to be different bodies.
        num_envs=8,
    )


DISTRIBUTIONS: dict[str, Distribution] = {
    "paper-draw": Distribution(
        name="paper-draw",
        tool_id=BALLPOINT,
        summary="ballpoint drawing scenes and squiggles on the ruled paper pad",
        build_args=_paper_draw,
        tip_calibration_jitter=True,
    ),
    "skin-erase": Distribution(
        name="skin-erase",
        tool_id=LASER,
        summary="laser removing ink from the draped silicone skin",
        build_args=_skin_erase,
    ),
    "skin-tattoo": Distribution(
        name="skin-tattoo",
        tool_id=LINER,
        summary="3RL liner depositing ink on the draped silicone skin "
                "(ballpoint settings — see _skin_tattoo for what that borrows)",
        build_args=_skin_tattoo,
    ),
    "body-tattoo": Distribution(
        name="body-tattoo",
        tool_id=LINER,
        summary="3RL liner following a compiled SVG trace on a posed rigged body",
        build_args=_body_tattoo,
    ),
}
