"""The lab: studies of what the rig can tattoo, as trials in cells read with the rig's cameras; offline."""
from __future__ import annotations

from tatbot_cli.registry import MOTION_AUTO, OFFLINE, declare_noun, verb
from tatbot_cli.stack import drawing_arm

from ._common import sh

declare_noun("lab", "development", "studies of what the rig can tattoo: trials, cells, plans and camera readings")

SCRIPT = "scripts/lab.sh"
LIB = "python/tatbot_lab/src/tatbot_lab"
WRAPS = (SCRIPT, "scripts/lab.py", "scripts/lib/drawing_python.sh", "scripts/lib/drawing_python.py", f"{LIB}/cli.py",
         f"{LIB}/store.py", "python/tatbot_contracts/src/tatbot_contracts/lab.py")
EFFECTS = ("read_files", "write_files", "start_process", "network")


def _new_args(p):
    p.add_argument("--id", required=True, help="the study's id, a directory under the lab root")
    p.add_argument("--question", required=True)
    p.add_argument("--stencil", required=True, help="the generated print its cells sit on (its directory)")
    p.add_argument("--arm", help="the drawing arm, whose caps the palette names (default: the first of stack.yaml arms)")
    p.add_argument("--cell-mm", default="16", help="a cell's side; one cell is centred on each grid marker")
    p.add_argument("--cartridges", required=True, help="tool ids, in the order to fit them")
    p.add_argument("--inks", required=True, help="catalogue ink ids")
    p.add_argument("--palette", required=True, help="ink=slot,...,water=slot: the caps that hold them")
    p.add_argument("--techniques", help="line,carry,solid,whip,corners by default")
    p.add_argument("--bound", action="append", default=[], help="setting=low:high, repeated; trim_mm and "
                   "speed_mm_s are always needed (dose_per_mm, pokes per mm, may stand for the speed), other "
                   "settings unbounded are held at their default")
    p.add_argument("--constants", help="override the held machine constants: battery_v=..,machine_hz=..,stroke_mm=..")


@verb(noun="lab", verb="new", tier=OFFLINE, visibility="public", audience="primary",
      disposition="starts a study: the question, the print and the space its trials may explore",
      effects=EFFECTS, summary="a study: its question, print, cartridges, inks, caps and the bounds of each setting",
      args=_new_args, wraps=(*WRAPS, f"{LIB}/tile.py"), doc="docs/lab.md",
      example=("--id", "card-3rl", "--question", "What does the 3RL lay at each depth?", "--stencil",
               "/tmp/stencil-bef888", "--cartridges", "lutin-3rl-bugpin", "--inks",
               "true_black,bright_red", "--palette", "true_black=inkcap_large_2,bright_red=inkcap_medium_1,"
               "water=inkcap_large_1", "--bound", "trim_mm=-2.5:-1.0", "--bound", "speed_mm_s=2.5:6"),
      invariants=("A study is written once; its machine constants default to 6.0 V, 51 Hz and a 4.0 mm stroke.",
                  "Records live under TATBOT_LAB_ROOT or ~/tatbot-lab, never in the repo."))
def lab_new(ctx, ns, rest):
    args = ["new", "--id", ns.id, "--question", ns.question, "--stencil", ns.stencil, "--arm", ns.arm or drawing_arm(),
            "--cell-mm", ns.cell_mm, "--cartridges", ns.cartridges, "--inks", ns.inks, "--palette", ns.palette]
    for name in ("techniques", "constants"):
        if value := getattr(ns, name):
            args += [f"--{name}", value]
    for bound in ns.bound:
        args += ["--bound", bound]
    return sh(ctx, SCRIPT, *args, *rest)


def _propose_args(p):
    p.add_argument("study")
    p.add_argument("--file", help="proposals, one JSON object a line: technique, cartridge, inks, settings")
    p.add_argument("--card", metavar="CARTRIDGE", help="the standard characterization card for this cartridge")
    p.add_argument("--dip-card", dest="dip_card", metavar="CARTRIDGE",
                   help="the dip card for this cartridge: its mark, how far a dip carries, and a dip's flood over the "
                        "dwell, depth in the cap and drain the study bounds")
    p.add_argument("--inks", help="the card's inks, its main first (default the study's); the dip card's is the first")
    p.add_argument("--search", metavar="N", help="N trials aimed at what the study does not yet know: bins no trial "
                                                 "reached, weak elites' neighbours, repeats of surprising reads")
    p.add_argument("--technique", help="with --search: the technique whose marks to search")
    p.add_argument("--cartridge", help="with --search: the cartridge (default the study's first)")
    p.add_argument("--ink", help="with --search: the ink (default the study's first)")
    p.add_argument("--seed", help="with --search: the random seed (default 0)")
    p.add_argument("--max-runs", dest="max_runs", help="with --search: distinct depth and speed pairs, each a run "
                                                       "(default 3)")


@verb(noun="lab", verb="propose", tier=OFFLINE, visibility="public", audience="primary",
      disposition="adds trials to a study, each checked against its bounds",
      effects=EFFECTS, summary="trials from a JSON-lines file, a cartridge's characterization card, or the search",
      args=_propose_args, wraps=(*WRAPS, f"{LIB}/propose.py", f"{LIB}/space.py", f"{LIB}/search.py"), doc="docs/lab.md",
      example=("card-3rl", "--card", "lutin-3rl-bugpin"),
      invariants=("Every proposal is checked before any is written: a technique, cartridge, ink or setting outside "
                  "the study is refused.", "A trial always states its depth trim and its speed."))
def lab_propose(ctx, ns, rest):
    args = ["propose", ns.study]
    for name in ("file", "card", "dip_card", "inks", "search", "technique", "cartridge", "ink", "seed", "max_runs"):
        if value := getattr(ns, name):
            args += [f"--{name.replace('_', '-')}", value]
    return sh(ctx, SCRIPT, *args, *rest)


def _map_args(p):
    p.add_argument("study")


@verb(noun="lab", verb="map", tier=OFFLINE, visibility="public", audience="primary",
      disposition="shows which marks a study has reached, so the search can aim at the rest",
      effects=EFFECTS, summary="each archive's bins of darkness and its second descriptor, and the elite trial of each",
      args=_map_args, wraps=(*WRAPS, f"{LIB}/search.py"), doc="docs/lab.md", example=("card-3rl",),
      invariants=("Only read trials place: a bin's elite is its least-speckled trial, of two as clean the faster.",))
def lab_map(ctx, ns, rest):
    return sh(ctx, SCRIPT, "map", ns.study, *(["--json"] if ctx.json else []), *rest)


def _plan_args(p):
    p.add_argument("study")
    p.add_argument("--stencil", required=True, help="the study's print (its directory)")
    p.add_argument("--sheet", required=True, help="the physical sheet's id: a fresh one starts with every cell free")


@verb(noun="lab", verb="plan", tier=OFFLINE, visibility="public", audience="primary",
      disposition="turns proposed trials into the runs `tatbot ros draw` draws, one cell each",
      effects=EFFECTS, summary="proposed trials into runs on a sheet: programs, the draw commands and cell manifests",
      args=_plan_args, wraps=(*WRAPS, f"{LIB}/plan.py", f"{LIB}/patterns.py", f"{LIB}/tile.py",
                              "ros/tatbot_ink/tatbot_ink/program.py"), doc="docs/lab.md",
      example=("card-3rl", "--stencil", "/tmp/stencil-bef888", "--sheet", "bef888-a"),
      invariants=("A run is one cartridge, one depth trim and one speed; runs go lightest ink to darkest, black last.",
                  "One cell is centred on each grid marker, never on an anchor; a plan that fails leaves nothing."))
def lab_plan(ctx, ns, rest):
    return sh(ctx, SCRIPT, "plan", ns.study, "--stencil", ns.stencil, "--sheet", ns.sheet, *rest)


def _status_args(p):
    p.add_argument("study", nargs="?", help="one study (default every study under the lab root)")


@verb(noun="lab", verb="status", tier=OFFLINE, visibility="public", audience="primary",
      disposition="what each study has proposed, planned, drawn and read, and which cells it has spent",
      effects=EFFECTS, summary="each study's trials by status, its plans and the cells used on each sheet",
      args=_status_args, wraps=WRAPS, doc="docs/lab.md", example=("card-3rl",))
def lab_status(ctx, ns, rest):
    return sh(ctx, SCRIPT, "status", *([ns.study] if ns.study else []), *(["--json"] if ctx.json else []), *rest)


def _capture_args(p):
    p.add_argument("study")
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--side", required=True, choices=("before", "after", "dry"),
                   help="before the run draws, after it, or dry: the same views once the ink has dried")
    p.add_argument("--rest-view", action="store_true", help="the one view where the arm stands, no motion")
    p.add_argument("--no-overhead", dest="overhead", action="store_false", help="no overhead D555 view")
    p.add_argument("--overhead-only", action="store_true", help="the overhead's view alone: the arm is not touched")
    p.add_argument("--sheet-at", dest="sheet_at", metavar="X,Y",
                   help="aim at the sheet nearest this point, mm in the arm's base, as the overhead reads it (no "
                        "set-up; `tatbot ros page` lists them); after and dry take the before side's")


@verb(noun="lab", verb="capture", tier=MOTION_AUTO, visibility="public", audience="primary",
      disposition="photographs a run's cells on the rig before it draws and after, for `lab read`",
      effects=(*EFFECTS, "remote_exec", "autonomous_motion"),
      summary="a run's cells photographed by the overhead and every wrist camera from a hold over each (tatbot ros views)",
      args=_capture_args, wraps=(*WRAPS, "scripts/lib/tatbot_cli/tatbot_ros.py",
                                 "ros/tatbot_session/tatbot_session/views.py"), doc="docs/lab.md",
      example=("card-3rl", "--plan", "p0001", "--run", "run-01", "--side", "before"),
      invariants=("The after side aims by the page the before side used, so the views stand at the same poses.",
                  "Before leaves the arm holding for the draw; after lands it. Frames go to <run>/views/<side>/."))
def lab_capture(ctx, ns, rest):
    flags = [flag for flag, on in (("--rest-view", ns.rest_view), ("--no-overhead", not ns.overhead),
                                   ("--overhead-only", ns.overhead_only)) if on]
    flags += [f"--sheet-at={ns.sheet_at}"] if ns.sheet_at else []
    return sh(ctx, SCRIPT, "capture", ns.study, "--plan", ns.plan, "--run", ns.run, "--side", ns.side, *flags, *rest)


def _run_args(p):
    p.add_argument("study")
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--stencil", required=True, help="the study's print (its directory), for the reads")
    p.add_argument("--sheet-at", dest="sheet_at", metavar="X,Y",
                   help="the sheet to draw on, mm in the arm's base (`tatbot ros page` lists them)")
    p.add_argument("--ends", choices=("stop", "fly"),
                   help="draw the run's stroke ends stopping or moving (`tatbot ros draw --ends`), over the stack's "
                        "motion.yaml draw.ends; lab-run.json records it")
    p.add_argument("--from", dest="start", choices=("before", "draw", "after", "read"),
                   help="resume at this step (default before)")


@verb(noun="lab", verb="run", tier=MOTION_AUTO, visibility="public", audience="primary",
      disposition="draws one planned run end to end: views before, the draw, views after, the reads",
      effects=(*EFFECTS, "remote_exec", "autonomous_motion"),
      summary="one planned run on the rig: its cells photographed before, drawn, photographed after, and read",
      args=_run_args, wraps=(*WRAPS, "scripts/lib/tatbot_cli/tatbot_ros.py",
                             "ros/tatbot_session/tatbot_session/views.py"), doc="docs/lab.md",
      example=("card-3rl", "--plan", "p0001", "--run", "run-01", "--stencil", "/tmp/stencil-981a", "--sheet-at=400,-100"),
      invariants=("The draw states the run's cartridge, so a cartridge the mount does not hold is refused, not drawn.",
                  "Each step's outcome is in <run>/lab-run.json; a failed step stops the run and --from resumes it."))
def lab_run(ctx, ns, rest):
    args = ["run", ns.study, "--plan", ns.plan, "--run", ns.run, "--stencil", ns.stencil]
    args += [f"--sheet-at={ns.sheet_at}"] if ns.sheet_at else []
    args += [f"--ends={ns.ends}"] if ns.ends else []
    args += ["--from", ns.start] if ns.start else []
    return sh(ctx, SCRIPT, *args, *rest)


def _read_args(p):
    p.add_argument("study")
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--stencil", required=True)
    p.add_argument("--before", help="frames taken before the run, one a view (default its `lab capture` views)")
    p.add_argument("--after", help="the same views after it, by the same file names (default its captured views)")
    p.add_argument("--instrument", default="wrist", choices=("wrist", "overhead"))
    p.add_argument("--side", default="after", choices=("after", "dry"), help="the captured views to compare with before")
    p.add_argument("--run-id", help="the ros-draw run that drew it")
    p.add_argument("--roi", help="x0,y0,x1,y1 image px: read markers only there (two sheets of one print in view)")


@verb(noun="lab", verb="read", tier=OFFLINE, visibility="public", audience="primary",
      disposition="measures a drawn run's cells from the rig's own frames, no person's photo",
      effects=EFFECTS, summary="a drawn run's cells from before and after frames: darkening, carry, fade, bands, speckle",
      args=_read_args, wraps=(*WRAPS, f"{LIB}/read/__init__.py", f"{LIB}/read/frames.py", f"{LIB}/read/diff.py",
                              f"{LIB}/read/measure.py", "scripts/vision/fiducials/stencil.py"), doc="docs/lab.md",
      example=("card-3rl", "--plan", "p0001", "--run", "run-01", "--stencil", "/tmp/stencil-bef888"),
      invariants=("Each view is a before and an after frame from one pose; the after frame is aligned to the before "
                  "on the skin no cell of the run changed.",
                  "The overhead's readings keep a cell's mean change only: it cannot see a line."))
def lab_read(ctx, ns, rest):
    args = ["read", ns.study, "--plan", ns.plan, "--run", ns.run, "--stencil", ns.stencil, "--instrument", ns.instrument,
            "--side", ns.side]
    for name in ("before", "after", "run_id", "roi"):
        if value := getattr(ns, name):
            args += [f"--{name.replace('_', '-')}", value]
    return sh(ctx, SCRIPT, *args, *rest)
