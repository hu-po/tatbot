"""Who each command is for, and why — the whole-CLI disposition, in one place.

199 commands is an inventory, not a menu. Ordinary help now shows the commands
an operator or developer reaches for; everything else stays one flag away
(`--help-all`), fully documented in the machine schema and the generated
reference. Nothing was removed to make a shorter list: hiding a command from a
help page is not the same as deleting it, and only the second needs evidence
that nobody calls it.

Five audiences:

  primary       everyday work. Shown in ordinary help.
  advanced      a real command that is simply needed less often — a specialist
                mode, an administration task, a diagnostic.
  compatibility an older spelling retained during a migration. It reaches the
                same handler as the canonical name it points at.
  internal      a protocol another program drives: an operator client, an
                owner-side receiver. Typing it by hand is rarely the point.
  experimental  a benchmark or development probe. Its output is evidence for a
                decision, not an operating instruction.

Every command must appear below with a reason. `scripts/check cli` fails on one
that does not, so a new command cannot quietly inherit a placement, and the
generated disposition table cannot go stale.
"""

from __future__ import annotations

PRIMARY = "primary"
ADVANCED = "advanced"
COMPATIBILITY = "compatibility"
INTERNAL = "internal"
EXPERIMENTAL = "experimental"

AUDIENCES = (PRIMARY, ADVANCED, COMPATIBILITY, INTERNAL, EXPERIMENTAL)
# What ordinary `--help` shows. Everything else needs `--help-all`, and every
# command stays in `schema --json` and the full generated reference.
SHOWN_BY_DEFAULT = (PRIMARY,)

# name -> (audience, why). The reason is the audit: it says what the command is
# for, not what it does — the summary already says that.
DISPOSITION: dict[str, tuple[str, str]] = {
    # --- the rig, day to day ---------------------------------------------------
    "status": (PRIMARY, "the first thing to run on any node"),
    "check": (PRIMARY, "the repository's own checks, before pushing"),
    "logs": (PRIMARY, "every workflow writes one; debugging starts here"),
    "estop check": (PRIMARY, "bench-check before an attended session"),
    "arm recover": (PRIMARY, "the routine way out of a controller fault"),
    "arm set-ip": (ADVANCED, "commissioning a controller, not operating one"),
    "tool list": (PRIMARY, "what may be stated as --ee-tool"),
    "tool show": (PRIMARY, "the datasheet the constants come from"),
    "tool sync": (ADVANCED, "a maintenance check on datasheet/code agreement"),
    "tool urdf": (ADVANCED, "regenerates a tracked file after a tool change"),
    "profile show": (PRIMARY, "which hardware profile this node resolved"),
    "profile list": (ADVANCED, "inventory of a checkout's profiles"),
    "profile check": (ADVANCED, "the motion gate's own validation, run early"),
    "deploy": (PRIMARY, "how a pushed commit reaches the fleet's services"),
    # --- rig -------------------------------------------------------------------
    "rig sleep": (PRIMARY, "the end of the day: cameras and hosts idle until wake"),
    "rig wake": (PRIMARY, "the start of the day: hosts up, services running, screen on"),
    "rig status": (PRIMARY, "is the rig asleep, and what did sleep actually switch off"),
    # --- work ------------------------------------------------------------------
    "work start": (PRIMARY, "the first thing a session does: its own worktree, its own pushed branch"),
    "work status": (PRIMARY, "what is unlanded on this node, and how old it is"),
    "work land": (PRIMARY, "the way work reaches main: rebase, fast tier, push, retry"),
    "work park": (PRIMARY, "leave work unlanded on purpose, pushed, with a reason"),
    "work sweep": (ADVANCED, "what the hourly timer runs; by hand when a node was offline"),
    "work init": (ADVANCED, "once per dev machine: mark the mirror, arm the hooks, install the timer"),
    "work hook": (INTERNAL, "invoked by Claude Code and git hooks, never by hand"),
    # --- release ---------------------------------------------------------------
    "release preview": (PRIMARY, "what the next public release would change, before anything is pushed"),
    "release publish": (PRIMARY, "the way work reaches the public repository: one export commit and tag"),
    "node list": (PRIMARY, "the node/role map an operator reads"),
    "node run": (ADVANCED, "arbitrary remote execution; deliberate, not routine"),
    "schema": (ADVANCED, "machine discovery of the command tree"),
    "completion bash": (ADVANCED, "one-time shell setup"),

    # --- teleop -----------------------------------------------------------------
    "teleop start": (PRIMARY, "the session other workflows attach to"),
    "teleop plan": (PRIMARY, "inspect physical arm assignments and executor support without hardware"),
    "teleop check": (PRIMARY, "why a start would refuse, before trying it"),
    "teleop analyze": (ADVANCED, "offline reading of a flight log"),
    "teleop poses": (ADVANCED, "measured follower FK for offline camera correlation"),
    "teleop trace-network": (ADVANCED, "bounded passive evidence when controller feedback drops"),

    # --- ROS 2 drawing stack -----------------------------------------------------
    "ros deploy": (PRIMARY, "how a commit reaches the ROS workspace on the ros node"),
    "ros up": (PRIMARY, "start the drawing stack, choosing hardware and e-stop source"),
    "ros down": (PRIMARY, "stop the drawing stack"),
    "ros status": (PRIMARY, "is the stack up, and is an arm holding"),
    "ros compile": (PRIMARY, "a design becomes the program the robot runs"),
    "ros draw": (PRIMARY, "the drawing path: a compiled program on the running stack"),
    "ros touch": (PRIMARY, "measure the page plane by touch before drawing"),
    "ros station": (PRIMARY, "where the palette is now, from its roof tag; calibration never uses a stored pose"),
    "ros calib": (PRIMARY, "the automatic tool-tip calibration on the station probe, between two fresh station fixes"),
    "ros palette": (PRIMARY, "what each palette cap holds and its measured ink level, which a dip reads"),
    "ros register": (PRIMARY, "where an arm stands in the overhead camera's world; the stencil page's pose needs it"),
    "ros chain": (ADVANCED, "judging a registration's wrist-tag seat and joint offsets before anyone adopts them"),
    "ros jog": (PRIMARY, "the first real motion of a bench morning: one joint, a few millimetres"),
    "ros inspect": (PRIMARY, "the wrist camera's look at a finished drawing: coverage, placement on the print, then rest"),
    "ros decide": (PRIMARY, "the operator's continue or land after an e-stop"),
    "ros cancel": (PRIMARY, "stop a running draw or touch from any terminal; the arm holds where it is"),
    "ros logs": (PRIMARY, "debugging a ROS run starts here"),
    "ros relay-install": (ADVANCED, "one-time setup of the palette Pi's e-stop relay"),

    # --- calibration -----------------------------------------------------------
    "calib pose": (ADVANCED, "position a parked wrist to inspect tool visibility before calibration"),
    "calib inspect": (ADVANCED, "measure the current pose before choosing a calibration or task target"),
    "calib register-fit": (ADVANCED, "review original native holds after moving an arm base or fixed camera"),
    "calib joint-measure": (ADVANCED, "an attended diagnostic before proposing joint or payload settings"),

    # --- vision ----------------------------------------------------------------
    "vision track": (PRIMARY, "the vision-only tracking shadow"),
    "vision tags scan": (PRIMARY, "what the cameras can currently see"),
    "vision deploy": (PRIMARY, "build and deploy the vision/teleop stack"),
    "vision handeye d405": (ADVANCED, "a specialist wrist-camera solve"),
    "vision capture": (ADVANCED, "the wrist cameras without an executor"),
    "vision touchoff": (ADVANCED, "re-solve a tip offset from retained samples"),
    "vision tags print": (ADVANCED, "produces a print sheet, occasionally"),
    "vision stencil generate": (ADVANCED, "coded flower-of-life frames (or the legacy floral one) and their print sheets"),
    "vision stencil reference": (ADVANCED, "binds existing artwork to reproducible tracking coordinates"),
    "vision stencil print": (ADVANCED, "print sheets with a millimetre ruler and a seed and size label, for paper and stencil apps"),
    "vision stencil replay": (EXPERIMENTAL, "measures stencil correspondence and recovery against retained frames"),
    "vision stencil observe": (EXPERIMENTAL, "advisory image tracking from an existing owner, without a surface pose"),
    "vision stencil bench": (EXPERIMENTAL, "scores a stencil design and tracker on synthetic transfer scenes, in mm"),
    "vision tags export": (ADVANCED, "publishes the wrist layout to the repo"),
    "vision d": (ADVANCED, "explicit access to the Rust vision binary"),
    "vision track-target": (ADVANCED, "publishes a measured target for an experiment"),
    "vision surface replay": (EXPERIMENTAL, "evidence about surface tracking, not an operation"),
    "vision depth compare": (EXPERIMENTAL, "compares retained depth evidence"),
    "vision board witness": (EXPERIMENTAL, "qualifies retained size and depth evidence before calibration"),
    "vision surface rgbd": (EXPERIMENTAL, "an offline registration/depth benchmark"),
    "vision surface attachment-benchmark": (EXPERIMENTAL, "a controlled comparison, for a decision"),

    # --- viewer and panel ------------------------------------------------------
    "live cockpit": (PRIMARY, "every live sensor in one view"),
    "viewer start": (PRIMARY, "bring up the one fleet viewer"),
    "viewer open": (PRIMARY, "attach a window on this node"),
    "viewer status": (PRIMARY, "is the fleet viewer up and reachable"),
    "viewer view": (PRIMARY, "play a recording into the fleet viewer"),
    "viewer stop": (ADVANCED, "buffered data is lost; deliberate"),
    "viewer install": (ADVANCED, "one-time system-unit installation"),

    # --- ink -------------------------------------------------------------------
    "ink status": (PRIMARY, "palette load, stock and the ledger tail"),
    "ink mise-en-place": (PRIMARY, "the human setup checklist"),
    "ink session start": (PRIMARY, "open the ink session"),
    "ink session end": (PRIMARY, "close it with its totals"),
    "ink edit": (ADVANCED, "explicit ledger and inventory edits"),
    "ink sync": (ADVANCED, "collects other nodes' ledgers"),
    "ink": (ADVANCED, "explicit access to the rest of ink.py"),

    # --- policies and datasets -------------------------------------------------
    "rollout run": (PRIMARY, "run a trained policy on the arm"),
    "rollout analyze": (PRIMARY, "read what a rollout actually did"),
    "rollout bench": (ADVANCED, "no-robot checks of the serving path"),
    "rollout contract": (ADVANCED, "reads a checkpoint's input/action contract"),
    "serve start": (ADVANCED, "runs the async policy server explicitly"),
    "serve stop": (ADVANCED, "stops the named server"),
    "train run": (PRIMARY, "one training job"),
    "train manifest": (ADVANCED, "renders or runs a job from a manifest"),
    "train offline-eval": (ADVANCED, "scores saved checkpoints"),
    "train profile": (ADVANCED, "this node's training profile"),
    "train pause": (ADVANCED, "yields the GPU to a rollout"),
    "train resume": (ADVANCED, "takes it back"),
    "data hub": (PRIMARY, "the dataset archive"),
    "data ds": (ADVANCED, "LeRobot dataset tools"),
    "data tool-meta": (ADVANCED, "stamps a dataset with its tool identity"),
    "body bootstrap": (ADVANCED, "one-time body-model cache setup"),
    "body audit": (ADVANCED, "hash audit of that cache"),
    "body export": (ADVANCED, "maintained offline preparation of shared articulated body poses"),
    "travel preview": (PRIMARY, "inspect travel camera views and articulated geometry before generation"),
    "travel generate": (ADVANCED, "generate approved travel datasets with canonical surface labels"),
    "travel trace": (PRIMARY, "scan the practice forearm and trace its ink with the blue arm for the travel demo"),

    # --- design, inkmap, inkgen ------------------------------------------------
    "design generate": (PRIMARY, "ask the generator for artwork"),
    "design trace": (PRIMARY, "turn an image or SVG into an artwork record"),
    "design place": (PRIMARY, "place artwork as a portable design"),
    "design check": (ADVANCED, "validates a design and its material footprint"),
    "research init": (EXPERIMENTAL, "DBV3 study init through the production drawing workflow"),
    "research candidate": (EXPERIMENTAL, "DBV3 study candidate through the production drawing workflow"),
    "research review": (EXPERIMENTAL, "shared Inkmap review of acquired artwork and production costs"),
    "research feedback": (EXPERIMENTAL, "records human preferences against exact reviewed artifacts"),
    "research prepare": (EXPERIMENTAL, "DBV3 study prepare through the production drawing workflow"),
    "research page": (EXPERIMENTAL, "DBV3 study page through the production drawing workflow"),
    "research run": (EXPERIMENTAL, "DBV3 study run through the production drawing workflow"),
    "research score": (EXPERIMENTAL, "DBV3 study score through the production drawing workflow"),
    "research decide": (EXPERIMENTAL, "DBV3 study decide through the production drawing workflow"),
    "ros ready": (EXPERIMENTAL, "complete normal startup before freezing a research runtime"),
    "ros evidence": (EXPERIMENTAL, "reads durable research slot claims and their ROS execution evidence"),
    "drawingbot generate": (PRIMARY, "acquires one drawing at its intended physical size with a recorded recipe"),
    "drawingbot export": (EXPERIMENTAL, "replays native acquisition independently of compilation"),
    "drawingbot container": (EXPERIMENTAL, "packages and verifies the mounted-app acquisition runtime"),
    "inkmap dev": (PRIMARY, "the preview app on this machine"),
    "inkmap resolve": (ADVANCED, "resolves placement text to an anchor"),
    "inkmap release-prepare": (ADVANCED, "builds a gated release packet"),
    "inkmap release-audit": (ADVANCED, "verifies one"),
    "inkmap deploy": (ADVANCED, "uploads the built bundle"),
    "inkgen serve": (PRIMARY, "run the generator in the foreground"),
    "inkgen status": (PRIMARY, "is a generator answering"),
    "inkgen batch": (PRIMARY, "run or resume a generation job"),
    "inkgen batch-status": (ADVANCED, "read a job's ledger"),
    "inkgen ctl": (ADVANCED, "explicit service control"),
    "inkgen deploy": (ADVANCED, "prepares and uploads the Space"),

    # --- simulator -------------------------------------------------------------
    "sim list": (PRIMARY, "the distributions the factory can generate"),
    "sim generate": (PRIMARY, "generate a dataset"),
    "sim preview": (PRIMARY, "see what it would generate, writing nothing"),
    "sim eval dataset": (PRIMARY, "score judged datasets"),
    "sim eval policy": (PRIMARY, "run a checkpoint through the async wire"),
    "sim sample": (ADVANCED, "a bounded artwork suite"),
    "sim compile": (ADVANCED, "compile one placement into a scenario"),
    "sim resolve": (ADVANCED, "resolve a typed request into a scenario"),
    "sim materialize": (ADVANCED, "materialize artwork before compiling"),
    "sim recipes": (ADVANCED, "expand a frozen library into recipes"),
    "sim recipes-status": (ADVANCED, "audit a recipe ledger"),
    "sim showcase-artwork": (ADVANCED, "previews for the pose gallery"),
    "sim qualify-artwork": (ADVANCED, "qualify paint, duration and deposition"),
    "sim perception": (ADVANCED, "render privileged perception labels"),
    "sim perception-audit": (ADVANCED, "audit such a corpus"),
    "sim pilot-plan": (ADVANCED, "write and audit the pilot ledger"),
    "sim pilot-audit": (ADVANCED, "audit an existing one"),
    "sim parity": (ADVANCED, "browser/simulator parity evidence"),
    "sim reach": (ADVANCED, "audit IK reach over the distribution"),
    "sim audit": (ADVANCED, "audit a generated dataset"),
    "sim samples": (ADVANCED, "stills and clips from a dataset"),
    "sim viewer": (ADVANCED, "an HTML viewer over render directories"),
    "sim cinematic": (ADVANCED, "path-traced takes for showing outside the lab"),
}


def disposition(name: str) -> tuple[str, str]:
    """(audience, why) for a command. Unlisted defaults to advanced with no
    reason, which `scripts/check cli` reports rather than accepting."""
    return DISPOSITION.get(name, (ADVANCED, ""))


def visible(v, *, show_all: bool = False) -> bool:
    return show_all or v.audience in SHOWN_BY_DEFAULT
