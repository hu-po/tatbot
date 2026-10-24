#!/usr/bin/env python3
"""Backend of `tatbot ros <verb>` (scripts/lib/tatbot_cli/verbs/ros.py). Stdlib only.

It runs on any node. A verb that acts on the stack resolves the node owning the `ros` role in
config/nodes.json and runs there over ssh inside the workspace root (~/tatbot-ros; never that node's
own git checkout), or locally on the owner. `--dry-run` prints every command and runs nothing.

Overrides for a parallel workspace (a lane, the integration run), all environment:
  TATBOT_ROS_ROOT    the workspace root on the ros node (default ~/tatbot-ros)
  TATBOT_ROS_DOMAIN  ROS_DOMAIN_ID written into its env.sh (default 0)
  TATBOT_ROS_PORT    its rmw_zenohd port on 127.0.0.1 (default 7450)
  TATBOT_ROS_NODE    another host for the stack than the `ros` role's owner; every verb, `deploy` included,
                     acts on it
  TATBOT_ROS_LANE    a sim lane, 0-39 (`up --hardware sim` requires it): the stack runs on the `ros-sim`
                     role's owner, never the `ros` node, in a container (containers/ros-sim/run.sh) rooted at
                     ~/tatbot-ros-sim/<lane>, ROS_DOMAIN_ID 60 + lane, router port 7460 + lane, sim server
                     7650 + 2 lane (machine relay + 1); TATBOT_ROS_NODE may name only that owner
Every verb acts on that root. The installed units run ~/tatbot-ros; `up`/`down`/`status` on any other
root use transient units named after it (tatbot-ros-lane-<dir>[-router], systemd-run, same limits), and a
sim lane's its container: every remote command runs inside it (a one-off one while the lane is down).

Exit codes: 0 ok, 2 usage, 4 no ros node in config/nodes.json, 5 the node is unreachable, 6 a sim lane is held
or CI runs on its node, else the remote command's own code. deploy, up, down and relay-install write a `ros-cli` run log here.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "lib"))
from tatbot_cli import nodes as fleet  # noqa: E402

EXIT_USAGE, EXIT_GATE, EXIT_WRONG_NODE, EXIT_UNREACHABLE, EXIT_BUSY = 2, 3, 4, 5, 6
# AGENTS.md marks repo/ as a checkout for scripts/lib/tatbot_runlog.py, which then reads config/runlog.json
# (log_root ~/tatbot-logs); without it every run on the ros node lands in ~/.local/state/tatbot/logs.
# rust/visiond/config: the D405 registry, which the calibration reads to find its arm's wrist camera
DEPLOY_DIRS = ("ros", "config", "urdf", "scripts", "python/tatbot_contracts", "rust/visiond/config", "AGENTS.md")
UNITS = ("tatbot-ros-router.service", "tatbot-ros.service")
RELAY_UNIT = "tatbot-estop-relay.service"
PROBE_UNIT = "tatbot-probe-relay.service"   # the same relay script as its own instance: never the e-stop's
MACHINE_UNIT = "tatbot-machine-relay.service"   # and the tattoo machine's switch, its own instance again
KEYS_UNIT = "tatbot-keys-relay.service"   # the station keyboard's drawing keys, on the station node
REGISTRATIONS = Path("~/tatbot-logs/vision").expanduser()
UDP_PORT = 7640  # stack.yaml estop.udp_port
PROBE_PORT = 7641  # stack.yaml probe.udp_port
UDEV_RULE = "config/udev/99-tatbot-estop.rules"  # /dev/tatbot-estop, wherever a Pico is read
PRODUCTION_ROOT = "~/tatbot-ros"
STACK_YAML = "ros/tatbot_bringup/config/stack.yaml"
# Sim lanes (ros-sim role): roots, ROS domains, router and sim-server ports per lane, and the container runtime.
SIM_ROLE, SIM_ROOT, SIM_LANES = "ros-sim", "~/tatbot-ros-sim", 40
SIM_DOMAIN, SIM_ROUTER_PORT, SIM_ENDPOINT_PORT = 60, 7460, 7650
SIM_RUN = "repo/containers/ros-sim/run.sh"   # in the lane root, as deploy syncs it
SIM_DEPLOY_DIRS = ("containers/ros-sim",)


class VerbError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class Target:
    """A node and a workspace root there, and how to run things in it."""

    def __init__(self, node: str, *, root: str = "~/tatbot-ros", domain: str = "0", port: str = "7450",
                 lane: int | None = None):
        nmap = fleet.load(REPO)
        self.node, self.domain, self.port, self.lane = node, domain, port, lane
        self.container = lane is not None   # a sim lane: its stack runs in containers/ros-sim, not systemd
        self.local = fleet.this_node(nmap) == node
        self.ssh = None if self.local else fleet.ssh_target(nmap, node)
        if not self.local and not self.ssh:
            raise VerbError(EXIT_WRONG_NODE, f"{node} has no ssh target in config/nodes.json")
        if not (root.startswith("~/") or root.startswith("/")):
            raise VerbError(EXIT_USAGE, f"workspace root must be ~/... or absolute, not {root!r}")
        self.root_arg = root
        self.production = root.rstrip("/") == PRODUCTION_ROOT and not self.container
        slug = re.sub(r"[^A-Za-z0-9_-]", "-", root.rstrip("/").rsplit("/", 1)[-1])
        if self.production:
            self.units = UNITS
        elif self.container:   # the lane's one container: its router and stack (run.sh names it so)
            self.units = (f"tatbot-ros-sim-{slug}",)
        else:  # transient units for a lane's root, named after its directory
            self.units = (f"tatbot-ros-lane-{slug}-router.service", f"tatbot-ros-lane-{slug}.service")
        # The root as a remote shell expands it, and as rsync/scp address it.
        self.root = "$HOME/" + root[2:] if root.startswith("~/") else root
        self.root_path = os.path.expanduser(root) if self.local else (root[2:] if root.startswith("~/") else root)
        self.run_sh = f"{self.root}/{SIM_RUN}"

    def dest(self, sub: str = "") -> str:
        path = f"{self.root_path}/{sub}" if sub else self.root_path
        return path if self.local else f"{self.ssh}:{path}"

    def shell(self, script: str, *, tty: bool = False) -> list[str]:
        """`script` where the stack runs: the node's shell, or a sim lane's container (run.sh exec)."""
        if self.container:
            script = f"{self.run_sh} exec {'--tty ' if tty else ''}{self.root} {shlex.quote(script)}"
        return self.host(script, tty=tty)

    def host(self, script: str, *, tty: bool = False) -> list[str]:
        """`script` in the node's own shell, outside any lane container."""
        if self.local:
            return ["bash", "-lc", script]
        # ssh hands the script to the login user's shell (bash on the fleet); env.sh brings ROS.
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", *(["-t"] if tty else []), self.ssh, script]

    @property
    def sim_endpoint(self) -> str:
        return f"tcp://127.0.0.1:{SIM_ENDPOINT_PORT + 2 * self.lane}" if self.container else ""

    def in_env(self, command: str) -> str:
        return f"source {self.root}/env.sh && {command}"

    def reachable(self, r: "Runner") -> None:
        """Before an exec'd ssh: an unreachable node is exit 5, not ssh's 255 or the client's own code."""
        if self.local or r.dry:
            return
        proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.ssh, "true"],
                              capture_output=True, text=True)
        if proc.returncode:
            raise VerbError(EXIT_UNREACHABLE, f"{self.node} ({self.ssh}) is unreachable: {proc.stderr.strip()[-300:]}")


def owner(role: str) -> str:
    try:
        return fleet.require_role(fleet.load(REPO), role)
    except ValueError as error:
        raise VerbError(EXIT_WRONG_NODE, f"config/nodes.json: {error}") from None


def home_relative(path: str) -> str:
    """A path under this node's HOME as `~/...` (the shell expanded the ~ in TATBOT_ROS_ROOT=~/x)."""
    home = os.path.expanduser("~")
    return "~/" + path[len(home) + 1:] if path.startswith(home + "/") else path


def sim_lane(hardware: str | None = None) -> int | None:
    """TATBOT_ROS_LANE; None for the `ros` node's stack. `--hardware sim` requires it: every later verb (draw,
    status, down) follows the variable, and without it would act on the ros node's live stack."""
    text = os.environ.get("TATBOT_ROS_LANE", "")
    if not text:
        if hardware == "sim":
            raise VerbError(EXIT_USAGE, f"--hardware sim needs TATBOT_ROS_LANE=<0-{SIM_LANES - 1}> exported: draw, "
                                        "status and down follow it, and without it act on the ros node's live stack")
        return None
    if not text.isdigit() or int(text) >= SIM_LANES:
        raise VerbError(EXIT_USAGE, f"TATBOT_ROS_LANE={text!r}: a sim lane is 0-{SIM_LANES - 1}")
    if hardware not in (None, "sim"):
        raise VerbError(EXIT_USAGE, f"TATBOT_ROS_LANE selects a sim lane, which runs hardware sim only, not {hardware}")
    return int(text)


def sim_target(lane: int) -> Target:
    """A sim lane on the `ros-sim` role's owner, which never falls back to the `ros` node."""
    node = owner(SIM_ROLE)
    if node == owner("ros"):   # owner, 2026-10-18: no sim lanes beside the live stack, whatever nodes.json says
        raise VerbError(EXIT_WRONG_NODE, f"{node} holds both {SIM_ROLE} and ros: a sim lane never runs on the ros node")
    named = os.environ.get("TATBOT_ROS_NODE")
    if named and named != node:
        raise VerbError(EXIT_WRONG_NODE, f"a sim lane runs on the {SIM_ROLE} owner ({node}), not {named}")
    return Target(node, root=home_relative(os.environ.get("TATBOT_ROS_ROOT", f"{SIM_ROOT}/{lane}")),
                  domain=str(SIM_DOMAIN + lane), port=str(SIM_ROUTER_PORT + lane), lane=lane)


def ros_target() -> Target:
    """The stack's node and root: the `ros` role's owner (or TATBOT_ROS_NODE), or TATBOT_ROS_LANE's sim lane."""
    lane = sim_lane()
    if lane is not None:
        return sim_target(lane)
    return Target(os.environ.get("TATBOT_ROS_NODE") or owner("ros"),
                  root=home_relative(os.environ.get("TATBOT_ROS_ROOT", "~/tatbot-ros")),
                  domain=os.environ.get("TATBOT_ROS_DOMAIN", "0"), port=os.environ.get("TATBOT_ROS_PORT", "7450"))


# --- running ------------------------------------------------------------------------------------
class Runner:
    """Prints each step; runs it unless dry; tees output into the run log's console.log."""

    def __init__(self, dry: bool, run=None):
        self.dry, self.run = dry, run

    def say(self, text: str) -> None:
        print(text, flush=True)
        if self.run is not None:
            with open(self.run.dir / "console.log", "a") as fh:
                fh.write(text + "\n")

    def step(self, label: str, argv: list[str], *, capture: bool = False, stdin: str | None = None) -> tuple[int, str]:
        self.say(f"[{label}] {shlex.join(argv)}")
        if self.dry:
            return 0, ""
        started = time.monotonic()
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL)
        if stdin is not None:
            proc.stdin.write(stdin)
            proc.stdin.close()
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if not capture:
                self.say(line.rstrip("\n"))
        code = proc.wait()
        if self.run is not None:
            self.run.event("step", label=label, exit_code=code, duration_s=round(time.monotonic() - started, 2))
        if code == 255 and argv[0] in ("ssh", "rsync", "scp"):
            raise VerbError(EXIT_UNREACHABLE, f"{label}: {argv[0]} could not reach the node (exit 255)")
        if code:
            raise VerbError(code, f"{label} failed (exit {code})")
        return code, "".join(lines)


def with_runlog(verb: str, dry: bool, body) -> int:
    """Run `body(runner)` inside a `ros-cli` run (not for a dry run)."""
    run = None
    if not dry:
        try:
            import tatbot_runlog
            run = tatbot_runlog.init("ros-cli", attach_logging=False, meta={"verb": verb, "backend_argv": sys.argv[1:]})
        except Exception as error:  # noqa: BLE001  (a log problem never stops the verb)
            print(f"tatbot ros: run log unavailable: {error}", file=sys.stderr)
    runner = Runner(dry, run)
    code = 1  # an unexpected exception is a failed run
    try:
        code = body(runner) or 0   # a verb's own code: `calib` exits 3 when its candidate is not adopted
    except VerbError as error:
        runner.say(f"tatbot ros {verb}: {error}")
        code = error.code
    except KeyboardInterrupt:
        code = 130
    finally:
        if run is not None:
            run.finalize(code)
            print(tatbot_runlog.banner("end", run.run_id, f"exit={code} log={run.dir}"), file=sys.stderr, flush=True)
    return code


# --- deploy -------------------------------------------------------------------------------------
def tracked_files(dirs: tuple = DEPLOY_DIRS) -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--", *dirs], check=True,
                         capture_output=True).stdout.decode()
    return sorted(f for f in out.split("\0") if f and (REPO / f).is_file())


def rsync_filter(files: list[str]) -> str:
    """Include exactly `files` (and their directories), exclude everything else."""
    dirs: set[str] = set()
    for f in files:
        parts = f.split("/")[:-1]
        dirs.update("/".join(parts[:i]) for i in range(1, len(parts) + 1))
    return "".join(f"+ /{d}/\n" for d in sorted(dirs)) + "".join(f"+ /{f}\n" for f in files) + "- *\n"


def revision(dirs: tuple = DEPLOY_DIRS) -> str:
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *dirs],
                           capture_output=True, text=True).stdout.strip()
    return f"{sha or 'unknown'}\ndirty={int(bool(dirty))}\n"


def env_sh(t: Target) -> str:
    template = (REPO / "ros/tatbot_bringup/scripts/env.sh.in").read_text()
    return template.replace("@ROOT@", t.root).replace("@DOMAIN@", t.domain).replace("@PORT@", t.port)


def changed_packages(itemized: str) -> set[str]:
    """Packages under ros/ whose files rsync changed or deleted (`--itemize-changes` lines)."""
    packages = set()
    for line in itemized.splitlines():
        path = line.split(" ", 1)[1].strip() if " " in line else ""
        if "__pycache__" in path or ".pytest_cache" in path:
            continue   # a test run's byproduct, deleted by --delete-excluded: nothing to build
        if path.startswith("ros/") and path.count("/") >= 2 and not line.startswith(".d"):
            packages.add(path.split("/")[1])
    return packages


def synced_sources(itemized: str) -> list[str]:
    """The ros/ files rsync wrote (`<f` lines sent to the node, `>f` to a local root), which the deploy marks new
    there: rsync -a keeps the source's mtimes, and a file older than the last build's outputs is not rebuilt. A deploy
    after a rollback left Touch.action's MODE_VIEW out of the generated Python, every calibration hold refused
    (2026-10-22)."""
    paths = (line.split(" ", 1)[1].strip() for line in itemized.splitlines() if line[:2] in ("<f", ">f") and " " in line)
    return [path for path in paths if path.startswith("ros/")]


def buildable(packages: set[str], r: Runner) -> set[str]:
    """The changed packages colcon can build: a COLCON_IGNOREd one (the relay) builds nothing, and a
    removed one leaves its install behind until `deploy --clean`."""
    out = set()
    for name in packages:
        path = REPO / "ros" / name
        if (path / "COLCON_IGNORE").exists():
            continue
        if (path / "package.xml").is_file():
            out.add(name)
        else:
            r.say(f"# {name} is gone from ros/: its build and install stay until `tatbot ros deploy --clean`")
    return out


def colcon(t: Target, packages: str) -> str:
    """The build command, ROS sourced: on the node, or in a sim lane's image (no Trossen SDK)."""
    if t.container:
        return f"{t.run_sh} build {t.root} {packages}".rstrip()
    return (f"source /opt/ros/jazzy/setup.bash && cd {t.root} && colcon build --base-paths repo/ros --build-base build --install-base install "
            f"--symlink-install {packages}--event-handlers console_direct- "
            f"--cmake-args -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_BASE_DIR={t.root}/deps")


# The Rerun bridge's SDK: the fleet pin, and the newest pyarrow that still takes numpy 1 (22+ needs numpy 2).
RERUN_SDK = {"rerun": "0.36.0", "pyarrow": "21.0.0"}


def rerun_sdk(t: Target) -> str:
    """Install the Rerun bridge's SDK, isolated in <root>/rerun-sdk, unless the pinned versions already import there.

    `--no-deps --target`, never the user or system site: rerun-sdk asks for numpy 2, which there would shadow the apt
    numpy 1.26 that the stack's cv2 and typesupport are built against; it runs on the apt numpy, attrs, Pillow and
    psutil. stack.launch.py puts the directory on the bridge's PYTHONPATH alone. Offline the deploy goes on and says
    loudly that the bridge will not start (display only)."""
    sdk = f"{t.root}/rerun-sdk"
    pins = " ".join(f"{name.replace('rerun', 'rerun-sdk')}=={version}" for name, version in RERUN_SDK.items())
    want = ", ".join(f"'{version}'" for version in RERUN_SDK.values())
    check = (f'PYTHONPATH="{sdk}:{sdk}/rerun_sdk" /usr/bin/python3 -c "import pyarrow, rerun; '
             f'assert (rerun.__version__, pyarrow.__version__) == ({want})"')
    return (f'{{ {check} 2>/dev/null && echo "rerun-sdk: present in {sdk}"; }} || {{ rm -rf "{sdk}" && /usr/bin/python3 '
            f'-m pip install --quiet --no-deps --retries 1 --timeout 20 --target "{sdk}" {pins} && {check} && echo '
            f'"rerun-sdk: installed in {sdk}"; }} || echo "WARNING: rerun-sdk is not installed in {sdk} (offline?): the '
            f'Rerun bridge will not start until a deploy installs it"')


def install_rerun_sdk(t: Target, r: Runner) -> None:
    if t.production:   # a lane's stack runs without the bridge unless `up --rerun`, which then needs its own install
        r.step("rerun-sdk", t.shell(rerun_sdk(t)))


def deploy_target(ns) -> tuple[Target, tuple]:
    """Where a deploy goes, and the tracked directories it syncs: a sim lane's also gets its container's."""
    t = ros_target()
    if not t.container:
        return t, DEPLOY_DIRS
    if ns.units:
        raise VerbError(EXIT_USAGE, "a sim lane runs in its container: it has no units to install")
    return t, DEPLOY_DIRS + SIM_DEPLOY_DIRS


def deploy(ns, r: Runner) -> None:
    t, dirs = deploy_target(ns)
    started = time.monotonic()
    lane = f", sim lane {t.lane}, built in containers/ros-sim" if t.container else ""
    r.say(f"# deploy {REPO} -> {t.node}:{t.root_arg} (domain {t.domain}, router port {t.port}){lane}")
    r.step("dirs", t.host(f"mkdir -p {t.root}/repo/config {t.root}/calib {t.root}/inbox {t.root}/deps"))
    # The owner's palette declarations (tatbot ros palette load) stay where they are; a first deploy gets this
    # checkout's.
    sync = ["rsync", "-a", "--delete", "--delete-excluded", "--prune-empty-dirs", "--itemize-changes",
            "--filter=P /config/palette_load.yaml", "--exclude=/config/palette_load.yaml"]
    if r.dry:  # a dry run starts no process and writes no file: not even git or the filter file
        r.say(f"# the tracked files under {', '.join(dirs)} (git ls-files), as an rsync include filter")
        r.step("sync", [*sync, "--filter=merge <tracked-files filter>", f"{REPO}/", t.dest("repo/")])
        itemized = ""
    else:
        files = tracked_files(dirs)
        with tempfile.NamedTemporaryFile("w", suffix=".rsync-filter", delete=False) as fh:
            fh.write(rsync_filter(files))
        try:
            r.say(f"# {len(files)} tracked files under {', '.join(dirs)}")
            _, itemized = r.step("sync", [*sync, f"--filter=merge {fh.name}", f"{REPO}/", t.dest("repo/")], capture=True)
        finally:
            os.unlink(fh.name)
    changed = buildable(changed_packages(itemized), r)
    if synced_sources(itemized):
        r.step("touch", t.host(f"cd {t.root}/repo && xargs -0 touch -c --"), stdin="\0".join(synced_sources(itemized)))
    r.say(f"# changed: {len([ln for ln in itemized.splitlines() if ln.strip()])} paths; ros packages: "
          f"{', '.join(sorted(changed)) or 'none'}" if not r.dry else "# changed: (from rsync --itemize-changes)")
    # Recorded on the node before anything else can fail or be skipped (--no-build, Ctrl-C): the build
    # step builds these plus whatever an earlier deploy left, and clears the file only on success.
    if changed or r.dry:
        r.step("pending", t.host(f"printf '%s\\n' {' '.join(sorted(changed)) or '<changed packages>'} "
                                 f">> {t.root}/.build-pending"))
    r.step("palette load", ["rsync", "-a", "--ignore-existing", str(REPO / "config/palette_load.yaml"),
                            t.dest("repo/config/palette_load.yaml")])
    registrations = sorted(REGISTRATIONS.glob("arm-registration-*-current.json"))
    if registrations:
        # --update: a registration adopted since on the ros node (a setup run from another node keeps its copy
        # there, 2026-10-12 from the station agent's node) is newer than this node's and stays; this node's replaces only older ones
        r.step("calib", ["rsync", "-a", "--update", *map(str, registrations), t.dest("calib/")])
    else:
        r.say(f"# no adopted registrations in {REGISTRATIONS}: each arm hangs at its nominal URDF mount")
    rev = "<git rev-parse HEAD> dirty=<0|1>" if r.dry else revision(dirs)
    r.say(f"# REVISION {' '.join(rev.split())}; env.sh from env.sh.in (domain {t.domain}, port {t.port}), on stdin")
    r.step("revision", t.host(f"cat > {t.root}/repo/REVISION"), stdin=rev)
    r.step("env", t.host(f"cat > {t.root}/env.sh"), stdin=env_sh(t))
    install_rerun_sdk(t, r)
    if not ns.no_build:
        build(t, r, ns, changed)
    elif not r.dry:
        r.say(f"# --no-build: {t.root_arg}/.build-pending keeps the synced packages for the next deploy")
    if ns.units:
        install_units(t, r)
    r.say(f"# deployed {rev.splitlines()[0][:12]} to {t.node}:{t.root_arg} in {time.monotonic() - started:.1f} s")
    note_running_stack(t, r)


def build(t: Target, r: Runner, ns, changed: set) -> None:
    """colcon, after the sync: every package on the first build or with --clean, else the packages rsync changed
    plus any an earlier deploy left in .build-pending, and those above them."""
    if ns.clean:
        stop_for_rebuild(t, r, "a clean build deletes the install the stack runs from")
        r.step("clean", t.host(f"rm -rf {t.root}/build {t.root}/install {t.root}/log"))
    if r.dry:
        r.say("# build: every package on the first build or with --clean, else the packages rsync changed "
              "and those above them (--packages-above); none when no ros/ file changed. A running stack is landed "
              "and stopped first when the build includes tatbot_interfaces or is clean (stop_for_rebuild)")
        r.step("build", t.host(colcon(t, "")))
        return
    # .build-pending holds every package synced but not yet built (this deploy's and any earlier one's).
    _, probe = r.step("probe", t.host(f"test -f {t.root}/install/setup.bash && echo installed; "
                                      f"cat {t.root}/.build-pending 2>/dev/null; true"), capture=True)
    words = probe.split()
    first = ns.clean or "installed" not in words or "ALL" in words
    changed |= {w for w in words if w != "installed"}
    if not (first or changed):
        r.say("# build: no ros/ package changed; nothing to build")
        return
    if "tatbot_interfaces" in changed or "ALL" in words:
        stop_for_rebuild(t, r, "the build includes tatbot_interfaces, after which no client of this install hears "
                               "the running stack (rmw_zenoh keys topics by type hash)")
    pending = "ALL" if first else " ".join(sorted(changed))
    select = "" if first else f"--packages-above {pending} "
    r.step("build", t.host(f"echo {pending} > {t.root}/.build-pending && "
                           f"{colcon(t, select)} && rm -f {t.root}/.build-pending"))


def stop_for_rebuild(t: Target, r: Runner, why: str) -> None:
    """Land and stop a running stack before a build that would leave this install unable to reach it, while it still
    can: afterwards `up` and `down` could neither land its arms nor see them landed (refuse_unanswered). Refused,
    exit 6, while a draw, touch or calibration runs there: deploy never lands an arm out of a goal."""
    if r.dry or subprocess.run(t.shell(stack_up(t)), capture_output=True).returncode != 0:
        return
    ros_busy(t)
    r.say(f"# {why}: landing and stopping the running stack first; `tatbot ros up` (same flags) starts the new build")
    down(argparse.Namespace(router=False, hold=False), r)


def note_running_stack(t: Target, r: Runner) -> None:
    """The running stack loaded its packages when it started: a deployed fix to the session, dips or motion is not
    live until it restarts (10-13: a dip fix deployed under a running stack refused the next draw's first dip)."""
    if not r.dry and subprocess.run(t.shell(stack_up(t)), capture_output=True).returncode == 0:
        r.say("# the stack is running the build it started with: `tatbot ros up` (same flags) loads this one")


def install_units(t: Target, r: Runner) -> None:
    render = " && ".join(
        f"sed -e \"s|@USER@|$USER|g\" -e \"s|@HOME@|$HOME|g\" {t.root}/repo/ros/tatbot_bringup/systemd/{u}.in "
        f"| sudo tee /etc/systemd/system/{u} >/dev/null" for u in UNITS)
    r.step("units", t.shell(f"{render} && sudo systemctl daemon-reload && sudo systemctl enable {UNITS[0]} "
                            f"&& systemctl cat {UNITS[1]} | head -3"))
    r.step("udev", t.shell(install_udev(f"{t.root}/repo/{UDEV_RULE}")))
    if not t.production:
        r.say(f"# note: the installed units run {PRODUCTION_ROOT}; up/down on {t.root_arg} use transient units")


def install_udev(rule: str, subsystem: str = "tty", device: str = "/dev/tatbot-estop",
                 missing: str = "plug the Pico in") -> str:
    """A device's symlink rule, installed and applied: by default the Pico's /dev/tatbot-estop (the node reading
    `--estop serial`, and the estop-relay node)."""
    return (f"sudo install -m 644 {rule} /etc/udev/rules.d/ && sudo udevadm control --reload-rules "
            f"&& sudo udevadm trigger --subsystem-match={subsystem} "
            f"&& (ls -l {device} 2>/dev/null || echo 'no {device} yet: {missing}')")


# --- up / down / status -------------------------------------------------------------------------
def launch_args(ns) -> list[str]:
    args = []
    for key, value in (("hardware", ns.hardware), ("estop_source", ns.estop), ("page_source", ns.page),
                       ("arms", ns.arms), ("page_surface", getattr(ns, "surface", None)),
                       ("tuning", getattr(ns, "tuning", None))):
        if value:
            args.append(f"{key}:={value}")
    args += senders(ns, stack_defaults())
    if getattr(ns, "pattern_id", None):   # up() resolved --pattern on the ros node
        args.append(f"pattern_id:={ns.pattern_id}")
    if ns.no_touch:
        args.append("touch:=false")
    if ns.rerun:
        args.append("rerun:=true")
    return args


def senders(ns, d: dict) -> list[str]:
    """The addresses the stack takes frames and keys from: the relay Pi's for the e-stop, probe and machine, and the
    station's for the keypad, resolved from config/nodes.json unless stated."""
    args = []   # stack.yaml's udp needs the relay too, except on mock hardware (no e-stop reader)
    implied_udp = not ns.estop and d.get("estop") == "udp" and (ns.hardware or d.get("hardware")) != "mock"
    if ns.estop == "udp" or implied_udp:
        args.append(f"estop_relay_addr:={relay_addr(ns, '--estop udp')}")
    elif ns.relay_addr:
        args.append(f"estop_relay_addr:={ns.relay_addr}")
    if ns.probe:   # the probe and machine relays are the same Pi as the e-stop relay: its `lan`
        args += ["probe:=true", f"probe_relay_addr:={relay_addr(ns, '--probe')}"]
    if d.get("machine_switch") == "pi" and not d.get("machine_addr"):
        args.append(f"machine_addr:={relay_addr(ns, 'machine.switch pi')}")
    if d.get("keypad_port", "0") != "0" and not d.get("keypad_from") and (station := role_lan("station")):
        args.append(f"keypad_from:={station}")
    return args


def relay_addr(ns, what: str) -> str:
    """--relay-addr, else the estop-relay node's `lan` (the palette Pi); `what` names the need in the refusal."""
    relay = ns.relay_addr or relay_lan()
    if not relay:
        raise VerbError(EXIT_USAGE, f"{what} needs the relay's address: pass --relay-addr ADDR, or record the "
                                    "estop-relay node's `lan` in config/nodes.json")
    return relay


def stack_defaults() -> dict:
    """hardware, estop.source, estop.relay_gpio, page.source, page.pattern_id, arms, machine.*, keypad.port and
    keypad.from_addr from this checkout's stack.yaml (the one deploy sends). A line reader, so the backend stays
    stdlib."""
    out, block = {}, None
    for line in (REPO / STACK_YAML).read_text().splitlines():
        body = line.split("#", 1)[0].rstrip()
        key, _, value = body.strip().partition(":")
        if not body.strip():
            continue
        if not body.startswith(" "):
            block = key
            if key in ("hardware", "arms"):
                out[key] = value.strip().strip("[]").replace(" ", "")
        elif key == "source" and block in ("estop", "page"):
            out[block] = value.strip()
        elif key.strip() == "pattern_id" and block == "page":
            out["pattern_id"] = value.strip()
        elif key == "relay_gpio" and block in ("estop", "probe"):
            out["relay_gpio" if block == "estop" else "probe_gpio"] = value.strip().strip('"')
        elif block == "machine":
            out[f"machine_{key.strip()}"] = value.strip().strip('"')
        elif block == "keypad" and key.strip() in ("port", "from_addr"):
            out["keypad_" + key.strip().removesuffix("_addr")] = value.strip().strip('"')
    return out


def effective(ns) -> str:
    """What the stack will run with: the flags given, else stack.yaml."""
    d = stack_defaults()
    pattern = (("pattern", ns.pattern_id),) if getattr(ns, "pattern_id", None) else ()
    surface = (("surface", ns.surface),) if getattr(ns, "surface", None) else ()
    tuning = (("tuning", ns.tuning),) if getattr(ns, "tuning", None) else ()
    return " ".join(f"{k}={v}" for k, v in (("hardware", ns.hardware or d.get("hardware")),
                                            ("estop", ns.estop or d.get("estop")), ("page", ns.page or d.get("page")),
                                            ("arms", ns.arms or d.get("arms")), *pattern, *surface, *tuning))


def relay_lan() -> str | None:
    return role_lan("estop-relay")


def role_lan(role: str) -> str | None:
    """The `lan` of the one node holding `role`, or None (no such node, or more than one)."""
    nmap = fleet.load(REPO)
    holders = fleet.nodes_with(nmap, role)
    return nmap[holders[0]].get("lan") if len(holders) == 1 else None


def lane_start(t: Target) -> str:
    """Start a lane root's router and stack as transient system units with the installed units' settings."""
    router, stack = t.units
    run = 'sudo systemd-run --quiet --collect --uid "$USER" --setenv=HOME="$HOME" --unit'
    router_cmd = f'source {t.root}/env.sh && ZENOH_CONFIG_OVERRIDE="$ZENOH_ROUTER_OVERRIDE" exec ros2 run rmw_zenoh_cpp rmw_zenohd'
    stack_cmd = f"source {t.root}/env.sh && exec ros2 launch tatbot_bringup stack.launch.py $TATBOT_ROS_ARGS"
    return (f"sudo systemctl stop {stack} {router} 2>/dev/null; sudo systemctl reset-failed {stack} {router} 2>/dev/null; "
            f"{run} {router} -p Restart=on-failure -p RestartSec=2 /bin/bash -c {shlex.quote(router_cmd)} && "
            f"{run} {stack} -p Requires={router} -p After={router} -p LimitRTPRIO=95 -p LimitMEMLOCK=infinity "
            f"-p KillSignal=SIGINT -p TimeoutStopSec=20 -p EnvironmentFile=-{t.root}/stack.env "
            f"/bin/bash -c {shlex.quote(stack_cmd)}")


# `systemctl is-active` prints "Failed to ..." when systemd itself does not answer (a full system bus):
# that is not "the stack is stopped", so the probes go on and ask the stack.
UNIT_UP = 'case "$(systemctl is-active {unit} 2>&1)" in active|*"Failed to"*) true;; *) false;; esac'
# A sim lane's stack, from inside its container; the bracket keeps pgrep off this probe's own command line.
LANE_UP = "pgrep -f '[s]tack.launch.py' >/dev/null"


def stack_up(t) -> str:
    """A shell test, where the stack runs, that is true while it does: its unit, or a sim lane's launch."""
    return LANE_UP if getattr(t, "container", False) else UNIT_UP.format(unit=t.units[1])


def unit_lines(t) -> str:
    """The status script's `unit:<name>=<state>` lines."""
    if getattr(t, "container", False):
        return f'echo "unit:{t.units[0]}=$({LANE_UP} && echo active || echo inactive)"'
    return f'for u in {" ".join(t.units)}; do echo "unit:$u=$(systemctl is-active $u 2>/dev/null)"; done'

# The real-SDK driver holds DRIVER_LOCK from configure until its arm has landed idle, and to its exit while an
# arm is commanded (tatbot_hardware runtime.hpp DriverLock, tatbot_arm.cpp): free = no driver holds an arm.
# lslocks reads it without taking it; no lslocks, or one that fails, is "unknown".
DRIVER_LOCK = "/tmp/tatbot-arm-driver.lock"

LAND_PROBE = r"""
{unit_up} || exit 0
echo "effective=$(sed -n 's/^TATBOT_ROS_EFFECTIVE=//p' {root}/stack.env 2>/dev/null)"
lock=unknown; held=$(lslocks -n -u -o PATH 2>/dev/null) && {{ echo "$held" | grep -qx """ + DRIVER_LOCK + r""" && lock=held || lock=free; }}
echo "driver_lock=$lock"
source {root}/env.sh && echo "client=$(timeout 15 ros2 run tatbot_session client status --json 2>/dev/null | tail -1)"
"""


def land_before_stop(t: Target, r: Runner, ns) -> list[str]:
    """Land every arm of a running stack before it stops. The controller firmware idles when its driver's
    connection ends, so an arm that has not landed can drop when the stack process exits (README 8).
    Prints each arm's state; `--hold` stops without landing. A landing that fails leaves the stack up.
    Returns the arms whose controller reported an error (those cannot land until the stack restarts)."""
    if ns.hold:
        refuse_hold_in_cap(t, r)
        r.say("# --hold: stopping without landing; an arm that has not landed can drop when its driver exits")
        return []
    probe = LAND_PROBE.format(unit_up=stack_up(t), root=t.root)
    if r.dry:
        r.step("arms", t.shell(probe))
        r.say("# then, when the stack runs hardware fake or real: `client land --arm <arm>` for each arm that has "
              "not landed and reports no controller error, before the stop (--hold skips this)")
        return []
    _, out = r.step("arms", t.shell(probe), capture=True)
    report = parse_status(out, t.node)
    if "effective" not in report:
        return []  # the stack is not running: nothing to land
    safety = (report.get("stack") or {}).get("safety") or {}
    r.say(f"# the stack runs with {report['effective'] or '(stack.yaml defaults)'}")
    hardware = stack_hardware(report["effective"])
    if not safety:
        refuse_unanswered(report, hardware)
        r.say(f"# no SafetyState from the stack (hardware {hardware}, driver lock {report.get('driver_lock')}): "
              "no driver holds an arm, so nothing can drop")
        return []
    mock = hardware == "mock"
    faulted = []
    for arm, row in sorted(safety.items()):
        r.say(f"# {arm}: landed={row.get('landed')} latched={row.get('latched')} ({row.get('latch_reason')}) "
              f"estop_ok={row.get('estop_ok')} controller_error={row.get('controller_error')}")
        if mock or row.get("landed"):
            continue
        if row.get("controller_error"):
            r.say(f"# {arm}: the controller reports an error and refuses to land; it lands after the restart")
            faulted.append(arm)
            continue
        try:
            r.step(f"land {arm}", t.shell(t.in_env(f"timeout 150 ros2 run tatbot_session client land --arm {arm}")))
        except VerbError as error:
            if error.code == EXIT_UNREACHABLE:
                raise
            raise VerbError(1, f"{arm} did not land ({error}); the stack is still up and holding. Release the "
                               f"e-stop and run `tatbot ros decide {arm} land`, or stop without landing: --hold") from None
    return faulted


def stack_hardware(effective: str) -> str:
    """The hardware a running stack drives, from its stack.env TATBOT_ROS_EFFECTIVE; stack.yaml's when that line is
    absent (the stack then runs stack.yaml's defaults). '' when the line names none."""
    if not effective:
        return stack_defaults().get("hardware") or ""
    return next((item.partition("=")[2] for item in effective.split() if item.startswith("hardware=")), "")


SIMULATED_HARDWARE = ("mock", "fake", "sim")   # stacks that drive no arm


def refuse_unanswered(report: dict, hardware: str) -> None:
    """A running stack none of whose arms answered can neither be landed nor be seen landed, and an arm that has not
    landed can drop when its driver exits (README 8): refused, exit 3, while a driver may hold an arm, i.e. on
    hardware that drives one with DRIVER_LOCK held or unreadable. A stack whose arm never came up (its spawner
    timed out, 10-18 03:00) holds no lock and restarts as before. The causes: a session that hung while its driver
    holds the arm, or a client deaf to the stack after its message types changed under it (rmw_zenoh keys topics by
    type hash; deploy now lands and stops a stack before rebuilding them, stop_for_rebuild)."""
    if hardware in SIMULATED_HARDWARE or report.get("driver_lock") == "free":
        return
    why = str((report.get("stack") or {}).get("error") or "no SafetyState")[:200]
    raise VerbError(EXIT_GATE, f"the stack is running (hardware {hardware or 'unknown'}, driver lock "
                               f"{report.get('driver_lock') or 'unknown'}) but no arm answered ({why}), so none can "
                               "be landed, or seen landed, before it stops; an arm that has not landed can drop when "
                               "its driver exits. Either its session hung while the driver holds the arm, or this "
                               "install's message types no longer match the running stack's. Look at the arm: once "
                               "it is at rest (or you accept the drop), `tatbot ros down --hold`, then go on (`tatbot "
                               "ros up` with the same flags, or the command that refused)")


def land_after_fault(t: Target, r: Runner, arm: str) -> None:
    """An arm whose controller reported an error before a restart: the restarted driver's configure cleared it
    and holds the arm where it lies, so land it now. The session first withdraws a pen that may be in the page
    along its own axis (README 8); on 2026-10-06 the controller dropped 49 s into a stroke and the 3RL's needles
    stayed in a half-skin until the operator came. The e-stop held down refuses both."""
    r.say(f"# {arm}: its controller reported an error before the restart: withdrawing the pen and landing it")
    r.step("ready", t.shell(ready_wait(t.root, arm, False)))
    r.step(f"land {arm}", t.shell(t.in_env(f"timeout 150 ros2 run tatbot_session client land --arm {shlex.quote(arm)}")))


def recover_controller(t: Target, r: Runner, arm: str) -> None:
    """After a draw that failed: an arm whose controller dropped cannot be moved by the stack that lost it (the SDK
    expects its process to end after one exception), so restart the stack and land it (land_after_fault)."""
    if r.dry:
        r.say(f"# a draw that ends with {arm}'s controller reporting an error restarts the stack and lands {arm}, "
              "the pen withdrawn first")
        return
    probe = LAND_PROBE.format(unit_up=stack_up(t), root=t.root)
    _, out = r.step("arms", t.shell(probe), capture=True)
    row = ((parse_status(out, t.node).get("stack") or {}).get("safety") or {}).get(arm) or {}
    if not row.get("controller_error"):
        return
    r.say(f"# {arm}: the controller dropped during the run; restarting the stack to withdraw the pen and land it")
    faulted = land_before_stop(t, r, argparse.Namespace(hold=False))   # the other arms land; this one cannot
    if not faulted:
        return
    restart(t, r)
    for each in faulted:
        land_after_fault(t, r, each)


def refuse_hold_in_cap(t: Target, r: Runner) -> None:
    """An arm that has not landed sags when its driver exits: 9 mm down and 6.7 mm aside at the palette on
    2026-10-05, with the nozzle in a cap. The session's in-cap marker names that state; stopping unlanded is
    refused (exit 3) while it stands."""
    _, out = r.step("in-cap", t.shell("ls $HOME/.local/state/tatbot/in-cap-*.json 2>/dev/null || true"), capture=True)
    if out.strip() and not r.dry:
        raise VerbError(EXIT_GATE, "a tool may be in a palette cap (" + out.strip().replace("\n", ", ") + "): "
                        "stopping without landing would drop it into the cap. Withdraw it first: resume the run "
                        "(tatbot ros draw <program> --resume <run>), whose next draw withdraws before anything, or, when "
                        "the stack cannot be reached, lift it out by hand and remove the marker")


REFERENCES = "$HOME/tatbot-logs/stencils/references"   # where the stencil observer reads installed prints


def installed_pattern(t: Target, r: Runner, pattern: str) -> str:
    """The full pattern id of the print installed on the ros node that `pattern` names: the id, or the 12 hex digits
    its sheet prints after `pattern`. The stack takes the page's size and clear centre from that print."""
    prefix = pattern if pattern.startswith("stencil-") else f"stencil-{pattern}"
    if not re.fullmatch(r"stencil-[0-9a-f]{6,64}", prefix):
        raise VerbError(EXIT_USAGE, f"--pattern {pattern!r}: a pattern id, or at least 6 of the hex digits a print "
                                    "sheet shows after `pattern`")
    _, out = r.step("pattern", t.shell(f"cd {REFERENCES} 2>/dev/null && ls -d {prefix}*/ 2>/dev/null; true"),
                    capture=True)
    if r.dry:
        return prefix
    found = sorted({line.strip().rstrip("/") for line in out.splitlines() if line.strip().startswith("stencil-")})
    if len(found) != 1:
        raise VerbError(EXIT_GATE, f"--pattern {pattern}: {'no print is' if not found else f'{len(found)} prints are'} "
                                   f"installed under it on {t.node}; install the print there first (tatbot vision "
                                   "stencil reference --settings <artwork>/settings.json --install)")
    return found[0]


def up(ns, r: Runner) -> None:
    lane = sim_lane(getattr(ns, "hardware", None))
    if lane is not None:
        return sim_up(ns, r, sim_target(lane))
    t = ros_target()
    if getattr(ns, "config", None):
        raise VerbError(EXIT_USAGE, "--config is a sim lane's stack (--hardware sim); the ros node runs deploy's")
    if getattr(ns, "pattern", None):
        ns.pattern_id = installed_pattern(t, r, ns.pattern)
    # stack.yaml starts the Rerun bridge into the fleet viewer's one standing recording: a lane's would draw over it
    args = " ".join(launch_args(ns) + ([] if t.production or ns.rerun else ["rerun:=false"]))
    eff = effective(ns)
    r.say(f"# up on {t.node}:{t.root_arg} ({' and '.join(t.units)}): {eff}")
    if "hardware=real" in eff and "estop=none" in eff:
        r.say("# note: hardware real with estop none: no e-stop input; the arm's rocker switch is the only stop "
              "(--estop udp|serial)")
    r.say(f"#   ros2 launch tatbot_bringup stack.launch.py {args}".rstrip())
    r.say(f"# {t.root_arg}/stack.env: TATBOT_ROS_ARGS={args} (on stdin)")
    faulted = land_before_stop(t, r, ns)  # a running stack lands before the restart
    r.step("stack.env", t.shell(f"cat > {t.root}/stack.env"), stdin=f"TATBOT_ROS_ARGS={args}\nTATBOT_ROS_EFFECTIVE={eff}\n")
    restart(t, r)
    for arm in faulted:   # its controller's error: cleared by the restart's configure
        land_after_fault(t, r, arm)


# Before a sim lane starts: CI first (a running job or CI container on the node), and the lane free.
SIM_BUSY = r"""
systemctl --user list-units 'tatbot-ci-job-*' --state=active --no-legend 2>/dev/null | grep -q . && echo "busy=a CI job runs on this node"
for e in docker podman; do command -v $e >/dev/null && $e ps -q --filter label=tatbot.ci.run 2>/dev/null | grep -q . && echo "busy=a CI container runs on this node"; done
[ -x {run_sh} ] && {run_sh} running {root} 2>/dev/null && echo "busy=lane {lane} is up: TATBOT_ROS_LANE={lane} tatbot ros down first, or another lane"
[ -f {root}/env.sh ] && [ -f {root}/install/setup.bash ] || echo "missing=1"
true
"""
SIM_REFUSED = ("estop", "relay_addr", "probe", "pattern", "rerun")


def sim_up(ns, r: Runner, t: Target) -> None:
    """A sim lane's stack: hardware sim with the run's --config, in the lane's container on the ros-sim owner, its
    driver stepping the sim server at the lane's endpoint. Refused (exit 6) while CI runs there or the lane is up."""
    given = [f"--{name.replace('_', '-')}" for name in SIM_REFUSED if getattr(ns, name, None)]
    if given:
        raise VerbError(EXIT_USAGE, f"{', '.join(given)}: a sim lane takes its e-stop, probe, page and Rerun from "
                                    "--config (the sim overlay)")
    if not ns.config:
        raise VerbError(EXIT_USAGE, "--hardware sim needs --config <stack.yaml>: the run's stack with the sim overlay, "
                                    f"a path on the {SIM_ROLE} owner")
    config = Path(ns.config).expanduser()
    if not config.is_absolute() or (t.local and not config.is_file()):
        raise VerbError(EXIT_USAGE, f"--config {ns.config}: an absolute path to a stack.yaml on {t.node}")
    args = " ".join(["hardware:=sim", f"config:={config}", f"sim_endpoint:={t.sim_endpoint}", "rerun:=false",
                     *(f"{key}:={value}" for key, value in (("arms", ns.arms), ("page_surface", ns.surface)) if value),
                     *(["touch:=false"] if ns.no_touch else [])])
    machine = f"udp://127.0.0.1:{SIM_ENDPOINT_PORT + 2 * t.lane + 1}"
    eff = f"hardware=sim lane={t.lane} config={config}"
    r.say(f"# up on {t.node}:{t.root_arg} (sim lane {t.lane}, container {t.units[0]}): {eff}")
    r.say(f"#   ros2 launch tatbot_bringup stack.launch.py {args}")
    r.say(f"#   ROS_DOMAIN_ID {t.domain}, router 127.0.0.1:{t.port}, sim server {t.sim_endpoint}, machine relay {machine}")
    _, out = r.step("busy", t.host(SIM_BUSY.format(run_sh=t.run_sh, root=t.root, lane=t.lane)), capture=True)
    if r.dry:
        r.say("# then refuse (exit 6) while CI runs on the node or the lane is up; exit 1 before a lane deploy")
    busy = [line[5:] for line in out.splitlines() if line.startswith("busy=")]
    if busy:
        raise VerbError(EXIT_BUSY, "; ".join(busy))
    if "missing=1" in out:
        raise VerbError(1, f"no built workspace at {t.node}:{t.root_arg}: TATBOT_ROS_LANE={t.lane} tatbot ros deploy")
    r.say(f"# {t.root_arg}/stack.env: TATBOT_ROS_ARGS={args} (on stdin)")
    r.step("stack.env", t.host(f"cat > {t.root}/stack.env"), stdin=f"TATBOT_ROS_ARGS={args}\nTATBOT_ROS_EFFECTIVE={eff}\n")
    r.step("start", t.host(f"{t.run_sh} up {t.root} --mount {shlex.quote(str(config.parent))} && sleep 3 && "
                           f"{t.run_sh} running {t.root}"))
    r.say(f"# lane {t.lane} is up: draw, status, logs and down follow it with TATBOT_ROS_LANE={t.lane}")


def restart(t: Target, r: Runner) -> None:
    """(Re)start the router and the stack with the stack.env already on the node."""
    if getattr(t, "container", False):   # the lane's container again, with the mounts its `up` recorded
        r.step("restart", t.host(f"{t.run_sh} down {t.root} && {t.run_sh} up {t.root} && sleep 3 && "
                                 f"{t.run_sh} running {t.root}"))
        return
    start = f"sudo systemctl restart {' '.join(t.units)}" if t.production else lane_start(t)
    try:
        # `systemctl is-active A B` is 0 when either is active: check each, so a stack that exited is a failure.
        # systemd not answering is reported and not failed on: the stack's own answer decides (wait_ready).
        check = " && ".join(UNIT_UP.format(unit=u) for u in t.units)
        r.step("restart", t.shell(f"{start} && sleep 3 && {check}"))
    except VerbError as error:
        hint = f"journalctl -u {t.units[1]} on {t.node}" + ("; units installed? (tatbot ros deploy --units)"
                                                            if t.production else "")
        # systemctl's own codes (3 inactive, 4 no such unit) would read as this CLI's; a failed start is 1
        raise VerbError(error.code if error.code == EXIT_UNREACHABLE else 1, f"{error}; {hint}") from None


# The arm answering, active and not landed: what a draw needs after a restart.
READY_WAIT = r"""
source {root}/env.sh
for i in $(seq 1 45); do
  row=$(timeout 15 ros2 run tatbot_session client status --json 2>/dev/null | tail -1)
  echo "$row" | python3 -c 'import json, sys; a = json.load(sys.stdin)["safety"][sys.argv[1]]; sys.exit(a["landed"] or a["latched"])' {arm} 2>/dev/null && break
  [ $i = 45 ] && {{ echo "{arm} not active 90 s after waking it: $row"; exit 1; }}
  sleep 2
done
"""
# Then, when the page comes from the stencil tracker, a measured page on the bus: the print the stack runs with
# (`ros up --pattern` writes it into stack.env), not stack.yaml's default, which watched an absent print for 3 min
# and refused the draw (2026-10-06). At most PAGE_WAIT_S, and then the draw goes on: its set-up places the page
# from the wrist and the gauge holds, and refuses there when they cannot. An inked print the overhead no longer
# reads held a person still 3 min here before any set-up began (owner, 2026-10-21).
PAGE_WAIT_S = 60
STACK_PATTERN = r"""pattern=$(sed -n 's/.*pattern_id:=\([^ ]*\).*/\1/p' {root}/stack.env 2>/dev/null)"""
READY_PAGE = STACK_PATTERN + r"""
end=$((SECONDS + {wait_s}))
while [ $SECONDS -lt $end ]; do
  timeout 10 ros2 run tatbot_bridge page_watch --seconds 2 --json ${{pattern:+--pattern-id "$pattern"}} 2>/dev/null | grep -q '"measured": [1-9]' && exit 0
  sleep 1
done
echo "no measured page from the overhead in {wait_s} s: the draw's set-up places it from the wrist"; exit 0
"""


def page_on_bus(effective: str) -> bool:
    """Whether a restarted stack's page arrives on the bus: the stencil tracker's does, a fixed page never
    does. A stack.env with no page source (written before TATBOT_ROS_EFFECTIVE) waits, as it always did."""
    source = next((item.partition("=")[2] for item in effective.split() if item.startswith("page=")), "")
    return source in ("", "stencil")


def ready_wait(root: str, arm: str, page: bool) -> str:
    return READY_WAIT.format(root=root, arm=shlex.quote(arm)) + (READY_PAGE.format(root=root, wait_s=PAGE_WAIT_S) if page else "exit 0\n")


def wake(t: Target, r: Runner, arm: str, page: bool = True, rest: bool = False) -> None:
    """A draw on an arm that landed (the end of the last draw, or a Decide LAND): the driver keeps a landed arm
    idle until it is woken, so wake that arm alone (`client wake`: its controllers and hardware cycled, the other
    arm untouched; a stack restart would take the other arm down mid-goal, 2026-09-30) and wait for it, and for a
    stencil page on the bus (a fixed page is never there) unless `page` is False: a calibration draws on no page.
    With `rest` an arm that has not landed lands first, so it is taken back at rest: a calibration's way to the
    station starts there (2026-09-29: from 0.3 m out, the planner's crossing to it met joint_1's limit)."""
    probe = LAND_PROBE.format(unit_up=stack_up(t), root=t.root)
    land = t.shell(t.in_env(f"timeout 150 ros2 run tatbot_session client land --arm {shlex.quote(arm)}"))
    awake = t.shell(t.in_env(f"timeout 60 ros2 run tatbot_session client wake --arm {shlex.quote(arm)}"))
    if r.dry:
        r.step("arms", t.shell(probe))
        r.say((f"# then land {arm} unless it has landed" if rest else f"# then, when {arm} reports landed") +
              f": wake {arm} alone and wait for it active"
              + (" and, on the stencil page source, a measured page" if page else ""))
        return
    _, out = r.step("arms", t.shell(probe), capture=True)
    report = parse_status(out, t.node)
    row = (((report.get("stack") or {}).get("safety") or {}).get(arm)) or {}
    if not row.get("landed"):
        if not (rest and row):   # awake and wanted where it is, or no stack answering (the run reports that)
            return
        r.say(f"# {arm} is awake where its last goal left it: landing it, so it is taken back at rest")
        r.step(f"land {arm}", land)
    r.say(f"# {arm} landed and idle: waking it alone")
    r.step(f"wake {arm}", awake)
    try:
        r.step("ready", t.shell(ready_wait(t.root, arm, page and page_on_bus(report.get("effective", "")))))
    except VerbError:
        # The draw will not run: put the arm back where it was, idle, rather than holding.
        r.say(f"# not ready: landing {arm} again")
        try:
            r.step(f"land {arm}", land)
        except VerbError as error:
            r.say(f"# {arm} did not land ({error}); it holds (tatbot ros decide {arm} land)")
        raise


# A sim lane's runs, once it is down, into the node's own log root and index (tatbot_runlog.adopt), with the
# deployed revision's logger: `tatbot logs` finds them there and retention prunes them. Its page-carry file and
# any unfinished run stay in the lane's log root. A failed adopt leaves them there and says so; the lane is down.
SIM_ADOPT = ('python3 {root}/repo/scripts/lib/tatbot_runlog.py adopt {root}/logs '
             '|| echo "# adopt failed (exit $?): the lane\'s runs stay in {root}/logs"')


def down(ns, r: Runner) -> None:
    t = ros_target()
    if getattr(t, "container", False):   # its router lives in the container too
        r.say(f"# down on {t.node}:{t.root_arg}: stop sim lane {t.lane} ({t.units[0]})")
        land_before_stop(t, r, ns)
        r.step("stop", t.host(f"{t.run_sh} down {t.root}"))
        r.step("adopt", t.host(SIM_ADOPT.format(root=t.root)))
        return
    units = t.units[::-1] if ns.router else t.units[1:]
    r.say(f"# down on {t.node}:{t.root_arg}: stop {' '.join(units)}")
    land_before_stop(t, r, ns)
    r.step("stop", t.shell(f"sudo systemctl stop {' '.join(units)}; systemctl is-active {' '.join(t.units)} || true"))


STATUS_SCRIPT = r"""
cd {root} 2>/dev/null || {{ echo "missing=1"; exit 0; }}
echo "root={root_arg}"
echo "revision=$(head -1 repo/REVISION 2>/dev/null)"
echo "dirty=$(sed -n 2p repo/REVISION 2>/dev/null | cut -d= -f2)"
echo "args=$(sed -n 's/^TATBOT_ROS_ARGS=//p' stack.env 2>/dev/null)"
echo "effective=$(sed -n 's/^TATBOT_ROS_EFFECTIVE=//p' stack.env 2>/dev/null)"
{unit_lines}
newest=$(ls -1t "${{TATBOT_LOG_ROOT:-$HOME/tatbot-logs}}"/ros-draw 2>/dev/null | grep -v latest | head -1); echo "newest_run=$newest"
""" + STACK_PATTERN + r"""
[ -f env.sh ] && source env.sh && echo "page=$(timeout 10 ros2 run tatbot_bridge page_watch --seconds 2 --json ${{pattern:+--pattern-id "$pattern"}} 2>/dev/null | grep '"summary"')"
if {unit_up} && [ -f env.sh ]; then
  echo "client=$(timeout 15 ros2 run tatbot_session client status --json 2>&1 | tail -1)"
fi
"""


def parse_status(out: str, node: str) -> dict:
    """The status script's `key=value` lines as one report."""
    report: dict = {"node": node, "units": {}}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        if key.startswith("unit:"):
            report["units"][key[5:]] = value
        elif key in ("client", "page"):
            try:
                report["stack" if key == "client" else key] = json.loads(value)
            except ValueError:
                report["stack" if key == "client" else key] = {"error": value}
        elif key:
            report[key] = value
    return report


def print_status(report: dict, root_arg: str) -> None:
    print(f"ros node   {report['node']}:{report.get('root', root_arg)}  revision {report.get('revision', '?')[:12]}"
          f"{' (dirty)' if report.get('dirty') == '1' else ''}")
    print(f"runs with  {report.get('effective') or '(stack.yaml defaults: no tatbot ros up yet)'}")
    print("units      " + "  ".join(f"{u} {s}" for u, s in report["units"].items()))
    print(f"args       {report.get('args') or '(stack.yaml defaults)'}")
    print(f"newest run {report.get('newest_run') or '-'}")
    page = (report.get("page") or {}).get("summary") or {}
    last = page.get("last") or {}
    print(f"page       {page.get('measured', '?')} measured / {page.get('lost', '?')} lost in 2 s on the bus; last "
          f"{last.get('source', '-')} age {last.get('age_ms', '-')} ms, in base {last.get('base_xyz_mm', '-')} mm")
    if "stack" in report:
        print("stack      " + json.dumps(report["stack"], sort_keys=True))


def status(ns, r: Runner) -> None:
    t = ros_target()
    script = STATUS_SCRIPT.format(root=t.root, root_arg=t.root_arg, unit_lines=unit_lines(t), unit_up=stack_up(t))
    if r.dry:
        r.step("status", t.shell(script))
        return
    proc = subprocess.run(t.shell(script), capture_output=True, text=True)
    if proc.returncode == 255 or not proc.stdout:
        raise VerbError(EXIT_UNREACHABLE, f"no answer from {t.node}: {proc.stderr.strip()[-300:]}")
    report = parse_status(proc.stdout, t.node)
    if report.get("missing"):
        raise VerbError(1, f"no workspace at {t.node}:{t.root_arg} (tatbot ros deploy)")
    if ns.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_status(report, t.root_arg)


# The stencil page as the tracker publishes it (page_watch, read only), and the page the newest set-up placed.
PAGE_SCRIPT = STACK_PATTERN + r"""
source {root}/env.sh
echo "setup=$(python3 -c 'import json, sys; p = json.load(open(sys.argv[1])); print(json.dumps({{"run": sys.argv[1].split("/")[-2], "used": p.get("used")}}))' "$(ls -t ~/tatbot-logs/ros-touch/*/page.json 2>/dev/null | head -1)" 2>/dev/null)"
timeout {timeout} ros2 run tatbot_bridge page_watch --seconds {seconds} --arm {arm} --json ${{pattern:+--pattern-id "$pattern"}}
"""


def page_moved(base_from_page: list, xyz_mm: list, rpy_deg: list) -> dict:
    """Where a tracked pose stands in a placed page's frame: along its x and y, mm, and turned about its normal, deg."""
    d = [xyz_mm[i] / 1000.0 - base_from_page[i][3] for i in range(3)]
    along = [sum(base_from_page[r][c] * d[r] for r in range(3)) * 1000.0 for c in range(2)]
    placed_yaw = math.degrees(math.atan2(base_from_page[1][0], base_from_page[0][0]))
    turn = (rpy_deg[2] - placed_yaw + 180.0) % 360.0 - 180.0
    return {"x_mm": round(along[0], 1), "y_mm": round(along[1], 1), "turn_deg": round(turn, 2)}


def page_report(out: str) -> dict | None:
    """PAGE_SCRIPT's output as a report: the watch's counts, the samples' mean pose and spread, and where it stands from
    the set-up page; None when the watch printed no summary."""
    setup, samples, summary = None, [], None
    for line in out.splitlines():
        if line.startswith("setup="):
            with contextlib.suppress(ValueError):
                setup = json.loads(line[6:]) or None
        elif line.startswith("{"):
            row = json.loads(line)
            if "summary" in row:
                summary = row["summary"]
            elif row.get("base_xyz_mm"):
                samples.append(row)
    if summary is None:
        return None
    report = {"pattern_id": summary.get("pattern_id"), "seconds": summary.get("seconds"),
              "measured": summary.get("measured"), "lost": summary.get("lost")}
    if not samples:
        return report
    xyz = [row["base_xyz_mm"] for row in samples]
    mean = [sum(v[i] for v in xyz) / len(xyz) for i in range(3)]
    last = samples[-1]
    report.update(base_xyz_mm=[round(v, 1) for v in mean], heading_deg=round(last["base_rpy_deg"][2], 2),
                  spread_mm=round(max(max(v[i] for v in xyz) - min(v[i] for v in xyz) for i in range(3)), 2),
                  sigma_mm=last.get("sigma_mm"), age_ms=last.get("age_ms"))
    if setup and setup.get("used"):
        report["from_setup"] = {"run": setup["run"], "surface_z_mm": round(setup["used"][2][3] * 1000.0, 1),
                                **page_moved(setup["used"], mean, last["base_rpy_deg"])}
    return report


def print_page(report: dict) -> None:
    """Three short lines a small agent can repeat as they stand: tracked or not, where (with the surface height the
    set-up measured, since the overhead alone reads the skin 3-8 mm high), and how far off the set-up page."""
    seen, n = report["measured"], report["measured"] + report["lost"]
    if "base_xyz_mm" not in report:
        print(f"stencil NOT seen: 0 of {n} samples in {report['seconds']:g} s (covered, out of view, or another print)")
        return
    x, y, z = report["base_xyz_mm"]
    where = f"centre {x:.0f} mm ahead, {abs(y):.0f} mm {'left' if y >= 0 else 'right'} of the arm"
    print(f"stencil tracked: {seen} of {n} samples in {report['seconds']:g} s, steady to {report['spread_mm']:.1f} mm")
    moved = report.get("from_setup")
    if not moved:
        print(f"  {where}; no set-up yet, so its height is the "
              f"overhead's {z:.0f} mm, which reads high")
        return
    print(f"  {where}; surface {moved['surface_z_mm']:.0f} mm up (the "
          f"set-up's; the overhead alone reads {z:.0f})")
    print(f"  off the set-up page: {moved['x_mm']:+.1f} / {moved['y_mm']:+.1f} mm, {moved['turn_deg']:+.1f} deg "
          "(1-3 mm is the set-up's own correction; more means the skin moved)")


def page(ns, r: Runner) -> int:
    """The stencil page as the overhead tracker sees it now, read only (no arm moves): how many samples it measured or
    lost over `--seconds`, where the page stands in the arm's base and how steadily, and how far it stands from the
    page the newest set-up (`tatbot ros touch` or `setup`) placed: move the skin and it follows."""
    t = ros_target()
    script = PAGE_SCRIPT.format(root=t.root, seconds=float(ns.seconds), timeout=int(ns.seconds) + 20,
                                arm=shlex.quote(ns.arm))
    if r.dry:
        r.step("page", t.shell(script))
        return 0
    proc = subprocess.run(t.shell(script), capture_output=True, text=True)
    if proc.returncode == 255:
        raise VerbError(EXIT_UNREACHABLE, f"no answer from {t.node}: {proc.stderr.strip()[-300:]}")
    report = page_report(proc.stdout)
    if report is None:
        raise VerbError(1, f"no page on the bus from {t.node}: {(proc.stderr or proc.stdout).strip()[-300:]}")
    if ns.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_page(report)
    return 0


# --- offline, motion, decisions -----------------------------------------------------------------
def compile_argv(design, ns) -> tuple[list[str], dict]:
    designs = [design] if isinstance(design, str) else list(design)
    argv = [str(REPO / 'scripts/lib/drawing_python.sh'), "-m", "tatbot_ink", "compile", *designs, "--arm", ns.arm]
    for flag, value in (("--ee-tool", ns.ee_tool), ("--speed", ns.speed), ("--inks", ns.inks), ("-o", ns.out),
                        ("--stencil", getattr(ns, "stencil", None))):
        if value:
            argv += [flag, value]
    for flag, dest in (("--max-segment-s", "max_segment_s"), ("--min-stroke-mm", "min_stroke_mm"),
                       ("--mark-width-mm", "mark_width_mm"), ("--max-stroke-mm", "max_stroke_mm")):
        if getattr(ns, dest, None):
            argv += [flag, getattr(ns, dest)]
    # `--at=X,Y` joined: a negative X would otherwise read as a flag
    argv += [f"{flag}={value}" for flag, value in (("--at", getattr(ns, "at", None)), ("--width", getattr(ns, "width", None)),
                                                   ("--keep-region", getattr(ns, "keep_region", None))) if value]
    from tatbot_paths import DRAWING_ROOTS

    env = {"PYTHONPATH": os.pathsep.join([*(str(REPO / root) for root in DRAWING_ROOTS),
                                          os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
           "TATBOT_REPO": str(REPO)}
    return argv, env


def compile_(ns, r: Runner) -> int:
    argv, env = compile_argv(ns.design, ns)
    r.say(f"[compile] {' '.join(f'{k}={v}' for k, v in env.items())} {shlex.join(argv)}")
    if r.dry:
        return 0
    os.execvpe(argv[0], argv, {**os.environ, **env})


def program_file(path: str, ns, r: Runner) -> Path:
    """The program to send: a program.json as is; acquired artwork or a placed design compiled here first."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise VerbError(EXIT_USAGE, f"no such file: {source}")
    try:
        from tatbot_contracts.canonical import parse_json

        doc = parse_json(source.read_bytes())
    except ValueError as error:
        raise VerbError(EXIT_USAGE, f"{source}: not JSON ({error})") from None
    if doc.get("format") == "tatbot-program":
        from tatbot_contracts.ros_program import validate_for_execution

        try:
            validate_for_execution(doc)
        except ValueError as error:
            raise VerbError(EXIT_USAGE, str(error)) from error
        if getattr(ns, 'arm', doc['arm']) != doc['arm']:
            raise VerbError(EXIT_USAGE, f"the program was prepared for the {doc['arm']} arm")
        if ns.at or ns.width or ns.speed or ns.stencil:
            raise VerbError(EXIT_USAGE, "--at, --width, --speed and --stencil apply to artwork or a placed design, "
                                        "not a compiled program")
        return source
    out = Path("<a new temporary directory>") if r.dry else Path(tempfile.mkdtemp(prefix="tatbot-ros-compile-"))
    argv, env = compile_argv(str(source), argparse.Namespace(arm=ns.arm, ee_tool=ns.ee_tool, speed=ns.speed, inks=None,
                                                             out=str(out), at=ns.at, width=ns.width,
                                                             stencil=ns.stencil))
    r.say(f"# {source.name} is not a tatbot-program: compiling it first")
    r.say(f"[compile] {shlex.join(argv)}")
    if not r.dry:
        subprocess.run(argv, env={**os.environ, **env}, check=True)
    return out / "program.json"


def rehearsal_file(program: Path, ns, r: Runner) -> Path:
    """The program drawn in the air (tatbot_contracts.ros_program.rehearsal): the same strokes, page setup, gauge
    holds, hover stops and stencil follow, the pen --standoff-mm over the measured page (default motion.yaml's
    pen.hover.height_m), the tattoo machine off, no dips. Written beside a temporary copy; the run is its own."""
    from tatbot_contracts.canonical import parse_json
    from tatbot_contracts.ros_program import rehearsal

    height = None if ns.standoff_mm is None else float(ns.standoff_mm) / 1000.0
    at = "motion.yaml's pen.hover height" if height is None else f"{height * 1e3:g} mm"
    if not program.is_file():   # a dry run's design, not compiled
        r.say(f"# rehearse: the compiled program in the air at {at}, the machine off, no dips")
        return program
    source = parse_json(program.read_bytes())
    try:
        out = rehearsal(source, height)
    except ValueError as error:
        raise VerbError(EXIT_USAGE, f"rehearse: {error}") from error
    dips = sum(op["op"] == "dip" for op in source["ops"])
    say = (f"# rehearse: {sum(op['op'] == 'stroke' for op in out['ops'])} strokes in the air at {at}, the machine off, "
           f"{dips} dip(s) left out, no inspection")
    if r.dry:
        r.say(say)
        return program
    path = Path(tempfile.mkdtemp(prefix="tatbot-ros-rehearse-")) / "program.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    r.say(f"{say}; {path}")
    return path


def exec_remote(t: Target, command: str, r: Runner, *, tty: bool) -> int:
    argv = t.shell(t.in_env(command), tty=tty and sys.stdin.isatty())
    r.say(f"[{t.node}] {shlex.join(argv)}")
    if r.dry:
        return 0
    t.reachable(r)
    os.execvp(argv[0], argv)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def run_cancellable(t: Target, command: str, r: Runner) -> int:
    """Run a draw or touch on the ros node, and on Ctrl-C or SIGTERM here cancel its goal there
    (`client cancel`) before returning. An exec'd ssh cannot: without a tty the interrupt only killed ssh
    and the goal kept drawing; with one it reached a client that could not send the cancel."""
    logged = r.run is not None and not sys.stdin.isatty()   # a setup's touch, in its own session: its lines to the log
    argv = t.shell(t.in_env(command), tty=sys.stdin.isatty())
    r.say(f"[{t.node}] {shlex.join(argv)}")
    if r.dry:
        return 0
    t.reachable(r)
    # SIGINT too: a job a script starts in the background inherits SIGINT ignored.
    previous = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    proc = subprocess.Popen(argv, **({"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
                                     if logged else {}))
    try:
        if logged:
            for line in proc.stdout:
                r.say(line.rstrip("\n"))
        return proc.wait()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cancel = t.shell(t.in_env("ros2 run tatbot_session client cancel"))
        r.say(f"[cancel] {shlex.join(cancel)}")
        done = subprocess.run(cancel, capture_output=True, text=True, timeout=60)
        r.say((done.stdout + done.stderr).strip() or f"cancel exit {done.returncode}")
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.terminate()
        return 130 if done.returncode == 0 else 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def cancel(ns, r: Runner) -> int:
    return exec_remote(ros_target(), "exec ros2 run tatbot_session client cancel", r, tty=False)


def draw(ns, r: Runner) -> int:
    if not ns.program:
        raise VerbError(EXIT_USAGE, "draw needs a program (or --progress)")
    if getattr(ns, "standoff_mm", None) is not None and not ns.rehearse:
        raise VerbError(EXIT_USAGE, "--standoff-mm is a rehearsal's height: add --rehearse")
    t = ros_target()
    program = program_file(ns.program, ns, r)
    stem = Path(ns.program).name.split(".")[0]
    if getattr(ns, "rehearse", False):
        program, stem = rehearsal_file(program, ns, r), f"{stem}-rehearse"
    name = f"{stem}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    copy = ["cp", str(program), t.dest(f"inbox/{name}")] if t.local else ["scp", "-q", "-o", "BatchMode=yes", str(program),
                                                                            t.dest(f"inbox/{name}")]
    r.step("inbox", t.shell(f"mkdir -p {t.root}/inbox"))
    r.step("send", copy)
    resume = draw_run_id(ns.resume) if ns.resume else ""
    if resume != (ns.resume or ""):
        r.say(f"# --resume {ns.resume} is this node's tatbot ros draw; its drawing on the ros node is {resume}")
    extra = "".join(f" {flag} {shlex.quote(value)}" for flag, value in (("--from-op", ns.from_op), ("--run-id", resume))
                    if value)
    arm = shlex.quote(ns.arm)
    if ns.wake:
        wake(t, r, ns.arm)
    draw_cmd = f"ros2 run tatbot_session client draw {t.root}/inbox/{name} --arm {arm}{extra}"
    # A complete draw is looked at with the wrist camera (inspect ends at rest); then the arm lands, the draw done or
    # not: sleep pose, controller idle, motors off, the pen lifted off the skin first. A draw that ended in an error
    # left it awake over the skin (a refused set-up on a forearm, 2026-10-18); a pause still waits inside the draw
    # (Decide). --hold leaves it holding.
    inspect = f"ros2 run tatbot_session client inspect --arm {arm}" if ns.inspect and not ns.rehearse else ""
    land = f"timeout 150 ros2 run tatbot_session client land --arm {arm} || rc=1; " if ns.land else ""
    script = (f"{draw_cmd}; rc=$?; " + (f"if [ $rc -eq 0 ]; then {inspect} || rc=1; fi; " if inspect else "")
              + f"{land}exit $rc")
    rc = run_cancellable(t, script if inspect or land else f"exec {draw_cmd}", r)
    if r.dry or rc not in (0, 130):   # 130: cancelled here
        recover_controller(t, r, ns.arm)
    return rc


def touch(ns, r: Runner) -> int:
    extra = "".join(f" {flag} {' '.join(shlex.quote(str(v)) for v in value)}" for flag, value in
                    (("--start", ns.start), ("--rpy", ns.rpy), ("--direction", ns.direction)) if value)
    extra += "".join(f" {flag} {float(value)}" for flag, value in
                     (("--prior", ns.prior), ("--speed", ns.speed), ("--max-travel", ns.max_travel)) if value)
    extra += " --probe" if ns.probe else ""
    extra += " --move" if ns.move else ""
    return run_cancellable(ros_target(), f"exec ros2 run tatbot_session client touch --arm {shlex.quote(ns.arm)}{extra}",
                           r)


def jog(ns, r: Runner) -> int:
    extra = f" --speed {ns.speed}" if ns.speed else ""
    return exec_remote(ros_target(), f"exec ros2 run tatbot_session client jog --arm {shlex.quote(ns.arm)} "
                                     f"--joint {int(ns.joint)} --delta {float(ns.delta)}{extra}", r, tty=True)


def inspect(ns, r: Runner) -> int:
    extra = "".join(f" {flag} {shlex.quote(str(value))}" for flag, value in
                    (("--run", ns.run), ("--frames", ns.frames), ("--poses", ns.poses)) if value)
    extra += " --depth-only" if ns.depth_only else ""
    return exec_remote(ros_target(), f"exec ros2 run tatbot_session client inspect --arm {shlex.quote(ns.arm)}{extra}",
                       r, tty=True)


def evidence(ns, r: Runner) -> int:
    command = shlex.join(["python3", "-m", "tatbot_session.research", "--page", ns.page, "--slot", ns.slot])
    t = ros_target()
    if r.dry:
        r.say(f"[evidence] {shlex.join(t.shell(t.in_env(command)))}")
        return 0
    result = subprocess.run(t.shell(t.in_env(command)), capture_output=True, text=True, check=True)
    print(result.stdout, end="")
    return 0


def ready(ns, r: Runner) -> None:
    """Complete the ordinary draw's wake-up before a caller freezes its runtime."""
    wake(ros_target(), r, ns.arm)


STATION_REFUSED = {1: "the station moved since the fix it was compared with: the data taken between them is not used",
                   3: "the D555 cannot measure the station now (the line above says why)",
                   5: "the D555's owner did not answer (tatbot-visiond-d555 on the overhead-depth node)"}


def station(ns, r: Runner) -> int:
    """The station in an arm's frame, measured now from the palette's roof tag, never from a stored pose (the
    probe-calibration plan's principle 8): `tatbot_calib station fix` on the ros node, the overhead D555 through the
    arm's adopted registration. `--against` compares with an earlier fix; a move exits 1, a station the D555 cannot
    measure 3."""
    station_fix(ns.arm, ns.against, r)
    return 0


# A D555 owner just restarted (a registration's adoption) tells systemd it is ready once its frame socket is bound,
# seconds before the camera streams: the agent's setup fixed the station 9 s after one and got no frame (2026-10-22).
OWNER_TRIES, OWNER_WAIT_S = 4, 10.0


def station_fix(arm: str, against: str, r: Runner) -> str:
    """One fresh station fix (see `station`); returns its run directory on the ros node. An owner not streaming yet
    is asked again, OWNER_WAIT_S apart."""
    t = ros_target()
    compare = f" --against {shlex.quote(against)}" if against else ""
    command = t.shell(t.in_env(f"exec ros2 run tatbot_calib station fix --arm {shlex.quote(arm)}{compare}"))
    for attempt in range(1, OWNER_TRIES + 1):
        try:
            _, out = r.step("fix", command)
            break
        except VerbError as error:
            if error.code != 5 or attempt == OWNER_TRIES:
                raise VerbError(error.code, STATION_REFUSED.get(error.code, str(error))) from None
            r.say(f"# the D555's owner gave no frame yet: the fix again in {OWNER_WAIT_S:g} s")
            time.sleep(OWNER_WAIT_S)
    fixed = next((json.loads(line) for line in reversed(out.splitlines()) if line.startswith('{"ok"')), {})
    return fixed.get("run_dir", "<station run>")


def register(ns, r: Runner) -> int:
    """The arm's registration to the overhead D555 from its wrist tags (`tatbot_calib register`): the stack drives
    the arm through holds over the table while the D555 owner captures each, and the solve places the arm's base in
    the camera's frame. `--adopt` installs a result that passes its gates, on the ros node where the stack and the
    D555 owner read it, then restarts the owner so its frames carry the bundle. The stack must be up with the arm."""
    t = ros_target()
    extra = "".join(f" {flag} {' '.join(shlex.quote(str(v)) for v in (value if isinstance(value, list) else [value]))}"
                    for flag, value in (("--center", ns.center), ("--table-z", ns.table_z), ("--spread", ns.spread),
                                        ("--heights", ns.heights), ("--max-holds", ns.max_holds),
                                        ("--prior", ns.prior), ("--holds", ns.holds))
                    if value not in (None, "", []))   # 0.0 is a table height, not unset
    flags = extra + (" --adopt" if ns.adopt else "") + (" --dry-run" if ns.plan_only else "")
    _, out = r.step("register", t.shell(t.in_env(f"exec ros2 run tatbot_calib register --arm {shlex.quote(ns.arm)}"
                                                  f"{flags}")))
    if ns.plan_only:   # the reach check prints holds, no {"ok"} line; a failed one has already raised
        return 0
    done = next((json.loads(line) for line in reversed(out.splitlines()) if line.startswith('{"ok"')), {})
    if done.get("adopted"):
        # the owner reads the bundle when it starts: restart it so every frame is stamped with the new world
        r.step("owner", t.shell("sudo systemctl restart tatbot-visiond-d555.service"))
        r.say(f"# adopted on {t.node}: the D555 owner restarted under bundle {done.get('bundle_id', '')[:12]}")
        keep_registration(t, r, ns.arm)
    return 0 if done.get("ok", r.dry) else 3


def keep_registration(t: Target, r: Runner, arm: str) -> None:
    """The adopted registration into this checkout's REGISTRATIONS, which `tatbot ros deploy` carries to the stack:
    left behind, the next deploy put the old one back (2026-09-29). The one it replaces is kept beside it."""
    if t.local:   # adopt wrote it there itself
        return
    name = f"arm-registration-{arm}-current.json"
    REGISTRATIONS.mkdir(parents=True, exist_ok=True)
    if (REGISTRATIONS / name).is_file() and not r.dry:
        shutil.copy2(REGISTRATIONS / name, REGISTRATIONS / f"{name}.replaced-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    r.step("keep", ["rsync", "-a", t.dest(f"calib/{name}"), str(REGISTRATIONS / name)])
    r.say(f"# {name} kept in {REGISTRATIONS}: the next deploy carries this registration")


def chain(ns, r: Runner) -> int:
    """What an arm's registrations leave, modelled over them on the ros node, where their runs are
    (`tatbot_calib chain`): the wrist tags' seat and the joints' offsets, each scored on runs it was not fitted to.
    Nothing moves; a wrist layout candidate is written to the run log and nothing is adopted."""
    t = ros_target()
    runs = " ".join(shlex.quote(run) for run in ns.runs)
    _, out = r.step("chain", t.shell(t.in_env(f"exec ros2 run tatbot_calib chain --arm {shlex.quote(ns.arm)} {runs}")))
    done = next((json.loads(line) for line in reversed(out.splitlines()) if line.startswith('{"ok"')), {})
    return 0 if done.get("ok", r.dry) else 1


def calib_apply(ns, r: Runner) -> int:
    if not ns.run:
        raise VerbError(EXIT_USAGE, "calib apply needs --run <ros-calib run id>")
    t = ros_target()
    _, text = r.step("candidate", t.shell(f"cat $HOME/tatbot-logs/ros-calib/{shlex.quote(ns.run)}/candidate.json"),
                     capture=True)
    if r.dry:
        return 0
    import probe_calibration_adopt

    candidate = json.loads(text)
    if candidate.get("arm") != ns.arm:
        raise VerbError(EXIT_USAGE, f"run {ns.run} calibrated the {candidate.get('arm')} arm, not {ns.arm}")
    path = REPO / "config" / "workspace.yaml"
    edited, why = probe_calibration_adopt.apply(candidate, path.read_text())
    if why:
        r.say("# not adopted: " + "; ".join(why))
        return 3
    path.write_text(edited)
    tip, tip0 = candidate["fit"]["tip_m"], candidate["tip0_m"]
    length = (f" and {(tip[2] - tip0[2]) * 1000:+.2f} mm along it" if candidate["fit"].get("length_measured") else "")
    r.say(f"# adopted {ns.run}: the {ns.arm} tip moved ({(tip[0] - tip0[0]) * 1000:+.2f}, "
          f"{(tip[1] - tip0[1]) * 1000:+.2f}) mm across its axis{length} in {path} (commit and land it; `tatbot ros "
          "deploy` carries it to the stack)")
    return 0


def palette(ns, r: Runner) -> int:
    argv = ['python3', '-m', 'tatbot_cli.ros_palette', ns.action, *ns.caps]
    for value in ns.level_mm:
        argv.extend(('--level-mm', value))
    command = 'PYTHONPATH="$TATBOT_REPO/scripts/lib:$PYTHONPATH" '+shlex.join(argv)
    return exec_remote(ros_target(), command, r, tty=False)


# A still on the palette camera's node: full resolution, focus fixed at the ball, exposure and gain fixed.
STILL = ("timeout 60 rpicam-still -n --mode 4624:3472:10 --autofocus-mode manual --lens-position 15 --shutter 60000 "
         "--gain 8 -o {path}")
STILL_TMP = "/tmp/tatbot-sweep-still.jpg"


def _at(t: Target, path: str) -> str:
    return path if t.local else f"{t.ssh}:{path}"


def still(cam: Target, ros: Target, dest: str, r: Runner) -> str:
    """One palette-camera still into `dest` on the ros node, by way of this node, which reaches the camera's node
    (the ros node holds no key there): "ok", or why not."""
    local = Path(tempfile.gettempdir()) / f"tatbot-sweep-{os.getpid()}.jpg"
    try:
        r.step("still", cam.shell(STILL.format(path=STILL_TMP)), capture=True)
        r.step("fetch", ["scp", "-q", "-o", "BatchMode=yes", _at(cam, STILL_TMP), str(local)])
        r.step("send", ["scp", "-q", "-o", "BatchMode=yes", str(local), _at(ros, dest)])
    except VerbError as error:
        return f"error: {error}"
    finally:
        local.unlink(missing_ok=True)
    return "ok"


def calib_sweep(ns, r: Runner) -> int:
    """`calib sweep`: the fitted tool's tip across its axis, contact-free, from turns of joint 6 the palette camera
    watches (tatbot_calib.sweep), after a fresh station fix; the stack must be up with the arm and --probe. The ros
    node holds the arm and asks for each still on its stdout; this node takes it on the palette camera's node (role
    estop-relay), copies it into the run and answers on its stdin."""
    started = time.time()
    t, cam = ros_target(), Target(owner("estop-relay"))
    opening = station_fix(ns.arm, "", r)
    wake(t, r, ns.arm, page=False, rest=True)
    tool = f" --tool {shlex.quote(ns.ee_tool)}" if ns.ee_tool else ""
    yaw = f" --yaw {float(ns.heading_deg):g}" if ns.heading_deg else ""
    height = " --height" if ns.action == "height" else ""
    argv = t.shell(t.in_env(f"exec ros2 run tatbot_calib sweep --arm {shlex.quote(ns.arm)} --station "
                             f"{opening}/station.json --run-started {started:.3f}{tool}{yaw}{height}"))
    r.say(f"[sweep] {shlex.join(argv)}")
    if r.dry:
        r.say(f"# at each hold: {shlex.join(cam.shell(STILL.format(path=STILL_TMP)))}, fetched here and sent into "
              "the run's stills/")
        return 0
    t.reachable(r)
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    done = {}
    for line in proc.stdout:
        r.say(line.rstrip("\n"))
        if line.startswith('{"still"'):
            proc.stdin.write(still(cam, t, json.loads(line)["still"], r) + "\n")
            proc.stdin.flush()
        elif line.startswith('{"ok"'):
            done = json.loads(line)
    if proc.wait() == 3:   # a tool the station cannot calibrate, or a height the overhead fix refutes
        raise VerbError(3, f"calib {ns.action} refused (the line above says why)")
    return 0 if done.get("ok") else 1


def calib_wristcam(ns, r: Runner) -> int:
    """`calib wristcam`: where the wrist camera sits on the tool mount, from the palette's roof tag seen at several
    holds, in the frame the station touch placed the ball in (tatbot_calib.wristcam); a check that adopts nothing:
    its holds put the gauge's tip within 1-2 mm, and `calib gauge` measures it. Needs this registration's station touch and its height (`calib run --station-only`,
    `calib height`)."""
    started = time.time()
    t = ros_target()
    opening = station_fix(ns.arm, "", r)
    wake(t, r, ns.arm, page=False, rest=True)
    tool = f" --tool {shlex.quote(ns.ee_tool)}" if ns.ee_tool else ""
    _, out = r.step("wristcam", t.shell(t.in_env(
        f"exec ros2 run tatbot_calib wristcam --arm {shlex.quote(ns.arm)} --station {opening}/station.json "
        f"--run-started {started:.3f}{tool}")))
    done = next((json.loads(line) for line in reversed(out.splitlines()) if line.startswith('{"ok"')), {})
    return 0 if done.get("ok", r.dry) else 1


def calib_gauge(ns, r: Runner) -> int:
    """`calib gauge`: the wrist gauge's tip along the pen from known heights over the probe body's top
    (tatbot_calib.gaugecal); adopted when the holds agree. Needs this registration's station touch and its height by
    the palette camera (`calib run --station-only`, `calib height`)."""
    started = time.time()
    t = ros_target()
    opening = station_fix(ns.arm, "", r)
    wake(t, r, ns.arm, page=False, rest=True)
    tool = f" --tool {shlex.quote(ns.ee_tool)}" if ns.ee_tool else ""
    _, out = r.step("gauge", t.shell(t.in_env(
        f"exec ros2 run tatbot_calib gaugecal --arm {shlex.quote(ns.arm)} --station {opening}/station.json "
        f"--run-started {started:.3f}{tool} --adopt")))
    done = next((json.loads(line) for line in reversed(out.splitlines()) if line.startswith('{"ok"')), {})
    return 0 if done.get("ok", r.dry) else 1


def calib_overhead(ns, r: Runner) -> int:
    """`calib overhead`: the overhead D555 registered through the wrist D405, not the wrist's cube
    (tatbot_calib.overhead): a D555 frame with the arm at rest, the wrist's holds over the palette's roof tag and its
    views of the fiducial tile on the holder (`--pattern`, the stack's own when unnamed), one fit. `--adopt` installs
    it on the ros node, keeps it here for deploys, and restarts what read the old one at start: stencild, the EE
    tracker and the stack (its stack.env unchanged). The station touch must be made again after."""
    started = time.time()
    t = ros_target()
    pattern = ns.pattern or stack_pattern(t, r) or stack_defaults().get("pattern_id", "")
    if not pattern:
        raise VerbError(EXIT_USAGE, "calib overhead needs the fiducial tile on the holder: --pattern <stencil id>")
    opening = station_fix(ns.arm, "", r)
    wake(t, r, ns.arm, page=False, rest=True)
    tool = f" --tool {shlex.quote(ns.ee_tool)}" if ns.ee_tool else ""
    _, out = r.step("overhead", t.shell(t.in_env(
        f"exec ros2 run tatbot_calib overhead --arm {shlex.quote(ns.arm)} --station {opening}/station.json "
        f"--run-started {started:.3f} --pattern {shlex.quote(pattern)}{tool}" + (" --adopt" if ns.adopt else ""))))
    done = next((json.loads(line) for line in reversed(out.splitlines()) if line.startswith('{"ok"')), {})
    if ns.adopt and done.get("ok"):
        r.step("restart", t.shell("sudo systemctl restart tatbot-stencild.service tatbot-trackd.service "
                                  "tatbot-ros.service"))
        r.step("ready", t.shell(ready_wait(t.root, ns.arm, False)))   # a setup's next step wakes it at once
        keep_registration(t, r, ns.arm)
        r.say("# adopted: stencild, the EE tracker and the stack restarted on it; touch the station again "
              f"(tatbot ros calib run --arm {ns.arm} --station-only, then tatbot ros calib height)")
    return 0 if done.get("ok", r.dry) else 1


def stack_pattern(t: Target, r: Runner) -> str:
    """The pattern the running stack was launched on (stack.env's TATBOT_ROS_EFFECTIVE), or ""."""
    pattern = stack_flag(t, r, "pattern")
    return "<the stack's pattern>" if r.dry else pattern


def stack_flag(t: Target, r: Runner, key: str) -> str:
    """One `key=value` of the running stack's stack.env TATBOT_ROS_EFFECTIVE, or ""."""
    _, out = r.step(key, t.shell("cat $HOME/tatbot-ros/stack.env"), capture=True)
    effective = next((line.split("=", 1)[1] for line in out.splitlines() if line.startswith("TATBOT_ROS_EFFECTIVE=")), "")
    return next((word.split("=", 1)[1] for word in effective.split() if word.startswith(f"{key}=")), "")


def calib(ns, r: Runner) -> int:
    """`calib run`: the arm's fitted tool tip across its axis, from side pairs on the station probe at upright yaws
    after a fresh station fix from the overhead D555. The stack must be up with the arm and --probe. A tool the
    station cannot probe is refused (exit 3) before anything moves. It writes candidate.json and report.md; `calib
    apply --run ID` writes that tip into this checkout's config/workspace.yaml when asked. `calib sweep` measures it
    contact-free (calib_sweep)."""
    if ns.action == "apply":
        return calib_apply(ns, r)
    if ns.action in ("sweep", "height"):
        return calib_sweep(ns, r)
    if ns.action == "wristcam":
        return calib_wristcam(ns, r)
    if ns.action == "gauge":
        return calib_gauge(ns, r)
    if ns.action == "overhead":
        return calib_overhead(ns, r)
    started = time.time()
    t = ros_target()
    arm = shlex.quote(ns.arm)
    tool = f" --tool {shlex.quote(ns.ee_tool)}" if ns.ee_tool else ""
    opening = station_fix(ns.arm, "", r)
    wake(t, r, ns.arm, page=False, rest=True)   # the run starts from rest; no page is drawn on
    # flag=value: an attitude list or heading that starts with a minus would read as a flag (2026-09-30)
    extra = tool + "".join(f" {flag}={shlex.quote(str(value))}" for flag, value in
                           (("--attitudes", ns.attitudes), ("--heading-deg", ns.heading_deg))
                           if value is not None and value != "") + (" --station-only" if ns.station_only else "")
    try:
        _, out = r.step("calibrate", t.shell(t.in_env(f"exec ros2 run tatbot_calib calib run --arm {arm} --station "
                                                       f"{opening}/station.json --run-started {started:.3f}{extra}")))
    except VerbError as error:
        raise VerbError(error.code, "the tool cannot be calibrated on the station probe (the line above says why)"
                        if error.code == 3 else str(error)) from None
    done = next((json.loads(line) for line in reversed(out.splitlines()) if line.startswith('{"ok"')), {})
    return 0 if done.get("ok", r.dry) else 1


# A setup's registration: holds planned from the rest pose's tags (no earlier layout assumed), 40 of them at four
# heights, as the 2026-10-05 re-layout's 38-of-40 registration took them.
SETUP_HOLDS = 40
SETUP_HEIGHTS = (0.10, 0.14, 0.18, 0.22)


SETUP_STEP = re.compile(r"^# setup: (.+?)(?: (\d+\.\d) min)?$")
SETUP_ACTIVITY = re.compile(r"^(\d\d:\d\d:\d\d |hold \d\d|# fit:|# rest prior)")
DRAW_START = re.compile(r"run_start\s+\S+\s+(\d+) ops")   # the stack pads its event column: `run_start  <id>`
DRAW_END = re.compile(r"\brun_end\s+\S+?:\s+(.*)$")
DRAW_PAUSE = re.compile(r"\[\w+\] pause \S+ (.*)$")
DRAW_WAIT = re.compile(r"\[\w+\] page\s+(the page has been lost .*waiting.*)$")
DRAW_STEP = re.compile(r"\[\w+\] \d+/\d+ \S+ (plan|travel|descend|draw|lift)$")
# The verbs that run in their own session, so a caller's end cannot stop them partway, and what they are called.
OWN_SESSION = {"setup": "calibration", "draw": "drawing"}


def draw_run_id(run: str) -> str:
    """The ros node's drawing run for `run`: itself, or, when it names this node's `tatbot ros draw` (what
    `draw --progress` shows; a station agent resumed with that id, 2026-10-13), the run_start its log carries,
    following a failed resume back to the run it resumed."""
    import tatbot_runlog

    seen = set()
    while run not in seen:   # a resume that failed before its run_start names, in its argv, the run it resumed
        seen.add(run)
        local = tatbot_runlog.log_root() / "ros-cli" / run
        with contextlib.suppress(OSError):
            found = re.search(r"run_start\s+(\S+)", (local / "console.log").read_text(errors="replace"))
            if found:
                return found.group(1)
        argv = []
        with contextlib.suppress(OSError, ValueError):
            argv = json.loads((local / "meta.json").read_text()).get("cmd", {}).get("argv", [])
        if "--resume" not in argv[:-1]:
            break
        run = argv[argv.index("--resume") + 1]
    return run


def setup_runs(verb: str = "setup") -> list[Path]:
    """This node's `tatbot ros <verb>` runs (setup or draw), newest first (each its ros-cli run directory)."""
    import tatbot_runlog

    root = tatbot_runlog.log_root() / "ros-cli"
    runs = []
    for run in sorted(root.glob("2*"), reverse=True):
        with contextlib.suppress(OSError, ValueError):
            if verb in json.loads((run / "meta.json").read_text()).get("cmd", {}).get("argv", [])[1:2]:
                runs.append(run)
        if len(runs) >= 3:
            break
    return runs


def _read_run_log(log: Path) -> dict:
    """A setup's or draw's console log: steps finished with their minutes, the step under way, the last activity
    line, the error it ended on; a draw's op count and ops done, how its run ended on the ros node, and why it is
    held now (a pause, or a lost page it waits for) unless a later stroke step moved on, and why the land after it
    left the arm over the skin (`client land`'s result: no lift off the skin planned, the session holds it)."""
    out = {"done": [], "step": None, "last": None, "error": None, "ops": None, "ops_done": 0, "ended": None,
           "held": None, "not_landed": None}
    with contextlib.suppress(OSError):
        for line in log.read_text(errors="replace").splitlines():
            _read_draw_line(line, out)
            m = SETUP_STEP.match(line)
            if m and m.group(2):
                out["done"].append((m.group(1), float(m.group(2))))
            elif m:
                out["step"], out["last"] = m.group(1), None
            elif SETUP_ACTIVITY.match(line):
                out["last"] = line.strip()
            elif line.startswith(("tatbot ros setup: ", "tatbot ros draw: ")):
                out["error"] = line.split(": ", 1)[1].strip()
    return out


def _read_draw_line(line: str, out: dict) -> None:
    """One console line's share of a draw's state: op count, ops done, how it ended, why it is held."""
    if (started := DRAW_START.search(line)) is not None:
        out["ops"] = int(started.group(1))
    if " op_done " in line:
        out["ops_done"] += 1
    if (end := DRAW_END.search(line)) is not None:
        out["ended"], out["held"] = end.group(1).strip(), None
    elif (held := DRAW_PAUSE.search(line) or DRAW_WAIT.search(line)) is not None:
        out["held"] = held.group(1).strip()
    elif DRAW_STEP.search(line):
        out["held"] = None
    if line.startswith('{"ok": ') and '"landed": [' in line:   # the land `tatbot ros draw` sends once the draw ends
        with contextlib.suppress(ValueError):
            land = json.loads(line)
            out["not_landed"] = None if land.get("ok") else str(land.get("message", "not landed"))


def draw_program(argv: list[str]) -> str | None:
    """The program a `tatbot ros draw` run was given (its first operand), from its argv."""
    takes = {"--arm", "--from-op", "--resume", "--speed", "--at", "--width", "--stencil", "--run-id", "--ee-tool",
             "--standoff-mm", "--from"}
    rest = argv[argv.index("draw") + 1:] if "draw" in argv else []
    skip = False
    for arg in rest:
        if skip:
            skip = False
        elif arg in takes:
            skip = True
        elif not arg.startswith("-"):
            return arg
    return None


def setup_state(run: Path) -> dict:
    """A setup run read from its log: its status (running only while its process lives), minutes since it
    started, the steps finished with their minutes, the step under way, its last activity line and how it ended."""
    import datetime as dt

    import tatbot_runlog

    meta = json.loads((run / "meta.json").read_text())
    status = meta.get("status") or "unknown"
    if status == "running" and not tatbot_runlog._pid_alive((meta.get("node") or {}).get("pid")):
        status = "stopped: its process was killed partway (a closed terminal or a command timeout), not a robot fault"
    started = dt.datetime.fromisoformat(meta["started_at"].replace("Z", "+00:00"))
    ended, log = meta.get("ended_at"), run / "console.log"
    if ended:
        until = dt.datetime.fromisoformat(ended.replace("Z", "+00:00"))
    elif status == "running" or not log.exists():
        until = dt.datetime.now(dt.timezone.utc)
    else:   # died: its last log line
        until = dt.datetime.fromtimestamp(log.stat().st_mtime, dt.timezone.utc)
    read = _read_run_log(log)
    argv = meta.get("cmd", {}).get("argv") or []
    verb = (argv or ["", "setup"])[1:2][0] if meta.get("cmd") else "setup"
    state = {"run": run.name, "verb": verb, "status": status,
             "minutes": round((until - started).total_seconds() / 60.0, 1), "exit_code": meta.get("exit_code"), **read}
    if verb == "draw":
        state["step"] = None if read["ops"] is None else f"{read['ops_done']} of {read['ops']} strokes done"
        state["program"] = draw_program(argv)
        state["resumes"] = argv[argv.index("--resume") + 1] if "--resume" in argv[:-1] else None
        state["rehearse"] = "--rehearse" in argv
    return state


SETUP_TYPICAL_MIN = 15
# A running setup asked about again within this, at the same step, answers in one line: the station's small agent
# called --progress 41 times in the first minutes of one setup (10-13).
PROGRESS_REPEAT_S = 120


def print_setup_state(state: dict, *, repeat: bool = False) -> None:
    """A few short lines a small agent repeats as they stand; a running one's say to stop and wait to be asked."""
    what = OWN_SESSION.get(state.get("verb", "setup"), "run")
    if state["status"] == "running":
        if repeat:
            print(f"unchanged: still {state['step'] or 'starting'} ({state['minutes']:.0f} min). Stop here; check "
                  "again only when the person asks.")
            return
        typical = f" of about {SETUP_TYPICAL_MIN}" if state.get("verb", "setup") == "setup" else ""
        print(f"{what} running, {state['minutes']:.0f}{typical} min, now: {state['step'] or 'starting'}")
        if state.get("held"):
            print(f"  HELD: {state['held'][:200]}")
            print("  The arm waits: tell the person this line. Fn+Enter on the station keyboard continues a pause.")
        print("Tell the person it is running and stop here; check again only when they ask.")
        return
    print(f"{what} {state['run']}: {state['status']}, {state['minutes']:.1f} min"
          + (f", {state['step']}" if state.get("verb") == "draw" and state["step"] else ""))
    if state["done"]:
        print("  took: " + ", ".join(f"{label} {minutes:.1f} min" for label, minutes in state["done"]))
    elif state["step"] and state.get("verb") != "draw":
        print(f"  stopped at: {state['step']}" + (f" -- {state['last'][:110]}" if state["last"] else ""))
    if state.get("ended") and state["ended"] != "complete":
        print(f"  ended on the ros node: {state['ended'][:200]}")
    if state.get("not_landed"):
        print(f"  NOT LANDED, the arm holds over the skin: {state['not_landed'][:300]}")
    if state["error"]:
        print(f"  failed: {state['error'][:160]}")


def setup_progress(ns, r: Runner) -> int:
    """The newest setup (or draw) run on this node, read only (but for the note of when it was last asked)."""
    runs = setup_runs(ns.verb)
    if not runs:
        print(f"no {OWN_SESSION[ns.verb]} has run from this computer")
        return 0
    state = setup_state(runs[0])
    if ns.json:
        print(json.dumps(state, sort_keys=True))
        return 0
    seen, repeat = runs[0] / "progress-asked.json", False
    with contextlib.suppress(OSError, ValueError):
        last = json.loads(seen.read_text())
        repeat = last.get("step") == state["step"] and time.time() - float(last.get("t", 0)) < PROGRESS_REPEAT_S
    if not repeat:
        with contextlib.suppress(OSError):
            seen.write_text(json.dumps({"step": state["step"], "t": time.time()}))
    print_setup_state(state, repeat=repeat)
    return 0


def setup_detach(ns, r: Runner) -> int:
    """Start this setup in its own session, so the caller's end cannot stop it partway: on 10-13 the station agent's
    tool timeout (opencode's ten minutes) killed three setups mid-touch, twice after --detach existed, because
    the agent left it out. Then, unless --detach, follow its log here until it ends and return its exit code; a
    caller killed meanwhile stops only the following. `--progress` reads it from anywhere on this node."""
    verb = ns.verb
    if verb == "draw" and not (ns.program and Path(ns.program).expanduser().exists()):
        raise VerbError(EXIT_USAGE, f"draw needs an existing program file, not {ns.program!r}")
    running = [run for v in OWN_SESSION for run in setup_runs(v) if setup_state(run)["status"] == "running"]
    if running:
        raise VerbError(6, f"a {setup_state(running[0])['verb']} is already running ({running[0].name}): "
                           f"tatbot ros {setup_state(running[0])['verb']} --progress")
    if r.dry:
        r.say("# then refuse (exit 6) while the ros node runs a registration, a station touch, a page touch or a draw")
        r.say(f"# then run this {verb} in its own session" + (", and return" if ns.detach else ", following its log"))
        return 0
    ros_busy(ros_target())
    argv = [sys.executable, str(Path(__file__).resolve()), verb, "--child",
            *[a for a in sys.argv[sys.argv.index(verb) + 1:] if a != "--detach"]]
    before = {run.name for run in setup_runs(verb)}
    subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    for _ in range(40):
        new = [run for run in setup_runs(verb) if run.name not in before]
        if new:
            break
        time.sleep(0.25)
    else:
        raise VerbError(1, f"the {verb} did not start a run log in 10 s")
    if ns.detach:
        return detach_report(new[0], verb, setup_minutes(getattr(ns, "start", None)) if verb == "setup" else None)
    return setup_follow(new[0])


# How long `--detach` watches the run it started before returning. A resume handed the station's id failed 20 s in
# while the agent, told "started in the background", said it was drawing (2026-10-14, three times); a draw reaches
# the ros node (run_start) about 15 s in, after the wake.
DETACH_WATCH_S = {"draw": 60.0, "setup": 45.0}
# Each setup step's minutes, in order (2026-10-22, the station agent's runs): a resumed setup takes those from its
# step on (a page alone 2.4 min, told "about 35"). A deploy first, when the stack runs an older build, adds its build.
SETUP_MIN = {"register": 12, "overhead": 10, "station": 6, "height": 2, "gauge": 13, "page": 3}


def setup_minutes(start: str | None) -> int:
    keys = list(SETUP_MIN)
    return sum(SETUP_MIN[k] for k in keys[keys.index(start or "register"):])


def detach_report(run: Path, verb: str, minutes: int | None = None) -> int:
    """Watch a detached run until it has really started (a draw the ros node took) or ended, or for its verb's
    DETACH_WATCH_S; print that and return its exit code (0 while it runs on)."""
    what, deadline, state = OWN_SESSION[verb], time.monotonic() + DETACH_WATCH_S[verb], None
    while True:
        with contextlib.suppress(OSError, ValueError, KeyError):
            state = setup_state(run)
        if state is None or state["status"] != "running" or (verb == "draw" and state["ops"] is not None) \
                or time.monotonic() >= deadline:
            break
        time.sleep(1.0)
    if state is not None and state["status"] != "running":
        print(f"the {what} did NOT start: it ended at once.")
        print_setup_state(state)
        return int(state["exit_code"]) if state["exit_code"] else 1
    now = (state or {}).get("step") or "starting"
    taken = verb == "draw" and state is not None and state["ops"] is not None
    print(f"{what} started in the background: run {run.name}"
          + (f"; it takes about {minutes} min" if minutes else "")
          + (f"; the ros node took it, {state['ops']} ops" if taken else f"; now: {now}") + ".")
    if state is not None and state.get("held"):
        print(f"  HELD: {state['held'][:200]}")
    print(f"Follow it with `tatbot ros {verb} --progress`.")
    return 0


# A calibration step in flight on the ros node, whichever node started it: a setup from the workstation and one
# from the station agent ran at once on 10-13, the second restarting the stack under the first's registration.
ROS_BUSY = r"""pgrep -af 'install/tatbot_calib/lib/tatbot_calib/|tatbot_session client (touch|draw)' | grep -v pgrep | head -1"""


def ros_busy(t: Target) -> None:
    """Exit 6 when the ros node is running a registration, station touch, page touch or draw."""
    proc = subprocess.run(t.shell(ROS_BUSY), capture_output=True, text=True)
    if proc.returncode == 255:
        raise VerbError(EXIT_UNREACHABLE, f"no answer from {t.node}: {proc.stderr.strip()[-300:]}")
    if proc.stdout.strip():
        what = proc.stdout.split(None, 1)[-1].strip()[:120]
        raise VerbError(6, f"{t.node} is already running a calibration step or a draw ({what}): wait for it, or "
                           "`tatbot ros setup --progress` where it was started")


def setup_follow(run: Path) -> int:
    """Print a setup run's log as it grows until the run ends; its exit code (1 when its process died)."""
    log, seen = run / "console.log", 0
    while True:
        state = setup_state(run)
        with contextlib.suppress(OSError):
            text = log.read_text(errors="replace")
            print(text[seen:], end="", flush=True)
            seen = len(text)
        if state["status"] != "running":
            return int(state["exit_code"]) if state["exit_code"] is not None else 1
        time.sleep(1.0)


CALIBRATION_TUNING = "calibration"   # config/trossen/tatbot.yaml ros_tuning.profiles: the gains setup calibrates on


def setup(ns, r: Runner) -> int:
    """An arm, the D555, the palette and the stencil measured afresh, as after setting the rig up anywhere: nothing
    from an earlier layout is assumed. The stack restarts with a fixed page and the probe (a stale registration
    put a tracked page at z 195 mm, where the holds' joint fallback refused them all, 2026-10-05), registers the arm
    from holds planned off the rest pose's tags, restarts on the stencil, registers the D555 again through the wrist
    camera (calib overhead) and keeps that registration here, touches the station probe (calib --station-only, its
    opening fix included), measures the ball's height and the wrist gauge, sets the page up from the wrist camera and
    lands the arm. Each step stops the setup when it fails, but for a gauge reading its holds do not agree on, which
    keeps the installed gauge; the arm lands either way. The calibration steps run on the `calibration` gains
    (CALIBRATION_TUNING), the page on the role's own after a restart; a setup that stops before it restarts the stack
    on those too, once the arm has landed."""
    t = ros_target()
    stack = {"hardware": ns.hardware, "estop": ns.estop, "relay_addr": ns.relay_addr, "arms": ns.arms, "probe": True,
             "no_touch": False, "rerun": ns.rerun, "hold": False, "pattern": ns.pattern, "surface": ns.surface,
             "tuning": CALIBRATION_TUNING}
    timings = []

    def timed(label, body):
        started = time.monotonic()
        r.say(f"# setup: {label}")
        code = body() or 0
        timings.append((label, time.monotonic() - started))
        if code:
            raise VerbError(code, f"{label} failed (exit {code})")

    steps = setup_steps(ns, r, t, stack)
    land = t.shell(t.in_env(f"timeout 150 ros2 run tatbot_session client land --arm {shlex.quote(ns.arm)}"))
    try:
        if not r.dry:
            deploy_if_behind(t, r)
        run_setup_steps(ns, steps, timed)
    finally:
        if not r.dry:
            try:
                r.step(f"land {ns.arm}", land)
                if stack_flag(t, r, "tuning") == CALIBRATION_TUNING:   # stopped before the page: draw on the role's
                    r.say("# setup: the stack still runs the calibration gains; restarting it on the drawing gains")
                    up(argparse.Namespace(**{**stack, "tuning": None}, page="stencil"), r)
            except VerbError as error:
                r.say(f"# {ns.arm} did not land, or the stack did not restart on the drawing gains ({error}); "
                      f"it holds (tatbot ros decide {ns.arm} land)")
        for label, seconds in timings:
            r.say(f"# setup: {label} {seconds / 60:.1f} min")
        if not r.dry:
            for line in measured_lines(t, ns.arm, r):
                r.say(f"# measured: {line}")
    return 0


def setup_steps(ns, r: Runner, t: Target, stack: dict) -> list:
    """(--from key or None, label, body) for each setup step, in order: a failed step names its key, so a re-run
    starts there instead of re-registering (three setups re-ran ten minutes of registration for a later step's
    failure, 2026-10-16)."""

    def calib_step(action, station_only=False):
        return lambda: calib(argparse.Namespace(action=action, run="", arm=ns.arm, ee_tool=ns.ee_tool, attitudes="",
                                                heading_deg="", station_only=station_only), r)

    def page():
        wake(t, r, ns.arm)
        return run_cancellable(t, f"exec ros2 run tatbot_session client touch --arm {shlex.quote(ns.arm)}", r)

    def gauge():
        # A reading its holds do not agree on (exit 3) keeps the installed gauge: the probe's top reads 1-4 mm apart
        # within one still hold (the arm moved under 0.25 mm), which no person at the station can fix (2026-10-16 and
        # 10-22 setups stopped here, the page never set up).
        try:
            return calib_step("gauge")()
        except VerbError as error:
            if error.code != 3:
                raise
        r.say("# setup: the gauge's new reading was not adopted (the line above says why); the installed gauge stays")
        return 0

    # The calibration steps run on the calibration gains (a lower velocity integral, which rang a third as much on
    # their moves, 2026-10-22), the page and every draw after on the role's own: gains change only at a restart.
    drawing = {**stack, "tuning": None}
    calibrating = (getattr(ns, "start", None) or "register") != "page"
    return [
        ("register", "restart with a fixed page", lambda: up(argparse.Namespace(**stack, page="fixed"), r)),
        ("register", "register the arm to the D555", lambda: register(argparse.Namespace(
            arm=ns.arm, center=None, table_z=None, spread=None, heights=list(SETUP_HEIGHTS), max_holds=None,
            prior="rest", holds=SETUP_HOLDS, adopt=True, plan_only=False), r)),
        (None, "restart on the stencil", lambda: up(argparse.Namespace(**(stack if calibrating else drawing),
                                                                     page="stencil"), r)),
        # the tags' registration is the coarse one, which a moved rig cannot mislead; the one kept is the wrist's,
        # through the palette's roof tag and the stencil on the holder (the tags' put the station 9 mm low, 2026-10-22)
        ("overhead", "register the D555 through the wrist", lambda: calib_overhead(argparse.Namespace(
            arm=ns.arm, ee_tool=ns.ee_tool, pattern=ns.pattern, adopt=True), r)),
        ("station", "touch the station probe", calib_step("run", station_only=True)),
        # the touch finds the ball across; its height comes from the palette camera (sweep --height): a needle
        # cartridge's touch never measures it, and dips aimed by its guess went too deep (2026-10-13)
        ("height", "measure the ball's height", calib_step("height")),
        ("gauge", "measure the wrist gauge on the probe's top", gauge),
        *([(None, "restart on the drawing gains", lambda: up(argparse.Namespace(**drawing, page="stencil"), r))]
          if calibrating else []),
        ("page", "set the page up", page),
    ]


def run_setup_steps(ns, steps: list, timed) -> None:
    """Each step from --from on (the stencil restart always: a resumed setup loads the newest build on the page it
    measures); a failed one says how to resume at it."""
    keys = [key for key, _, _ in steps if key]
    start = keys.index(getattr(ns, "start", None) or "register")
    for key, label, body in steps:
        if key is not None and keys.index(key) < start:
            continue
        try:
            timed(label, body)
        except VerbError as error:
            if key is None:
                raise
            raise VerbError(error.code, f"{error}; once fixed, resume from this step: tatbot ros setup "
                                        f"{shlex.join(resume_flags(ns))} --from {key} --detach") from error


def resume_flags(ns) -> list[str]:
    """The flags a setup was started with, for the line that resumes it."""
    out = ["--arm", ns.arm]
    for flag, value in (("--hardware", ns.hardware), ("--estop", ns.estop), ("--relay-addr", ns.relay_addr),
                        ("--pattern", ns.pattern), ("--surface", ns.surface), ("--arms", ns.arms)):
        if value:
            out += [flag, value]
    return out + (["--rerun"] if ns.rerun else [])


def deploy_if_behind(t: Target, r: Runner) -> None:
    """Deploy this checkout first when the ros node runs an older commit of it: a setup on 2026-10-16 ran on a
    build without the previous night's dip safeguards. A ros node that runs a commit this checkout does not have
    (newer, or deployed from elsewhere) is left alone and named: deploying from here would take it back."""
    here = revision().split()[0]
    _, there = r.step("deployed", t.shell(f"head -1 {t.root}/repo/REVISION 2>/dev/null; true"), capture=True)
    there = there.strip().split()[0] if there.strip() else ""
    if not there or there == here:
        return
    behind = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", there, here]).returncode == 0
    if behind:
        r.say(f"# the ros node runs {there[:12]}, older than this checkout's {here[:12]}: deploying it first")
        deploy(argparse.Namespace(no_build=False, units=False, clean=False), r)
    else:
        r.say(f"# the ros node runs {there[:12]}, which this checkout ({here[:12]}) does not have: left as it is "
              "(git pull here, or deploy from the newer checkout)")


def measured_lines(t: Target, arm: str, r: Runner) -> list[str]:
    """What positions the needle, each as measured (when, by what) or not: the tip in the mount (workspace.yaml's
    touch-off record), the ball's height (the station touch) and the wrist camera's tip (the gauge file). A setup
    that passed says nothing about a number it did not measure (2026-10-13: a 3RL tip and gauge stood in for the
    11RL through many passing setups)."""
    import tool_spec

    record = tool_spec.parse_simple_yaml((REPO / "config" / "workspace.yaml").read_text())[arm]
    touch, fitted = record.get("touchoff") or {}, record.get("tool_id")
    length = "measured" if "length, both measured" in str(touch.get("note", "")) else "NOT measured (the installed one)"
    lines = [f"tip across its axis: {touch.get('method', '?')} ({touch.get('session', '?')}); its length {length}"]
    try:
        gauge, station_touch = (json.loads(r.step("measured", t.shell(
            f"cat $HOME/tatbot-ros/calib/{name}-{shlex.quote(arm)}.json"), capture=True)[1])
            for name in ("wrist-gauge", "station-touch"))
    except (VerbError, ValueError):
        return [*lines, "the wrist gauge and the station touch could not be read"]
    with_ = gauge.get("measured_with")
    lines.append("wrist camera's tip: " + (f"measured with {with_} ({gauge.get('utc')})" if with_ == fitted else
                                            f"NOT measured with {fitted} ({gauge.get('note') or gauge.get('source')})"))
    lines.append("ball height: " + (station_touch.get("height_source") or "NOT measured (a needle's touch never trips "
                                                                       "on its top)"))
    return lines


def decide(ns, r: Runner) -> int:
    return exec_remote(ros_target(), f"exec ros2 run tatbot_session client decide {shlex.quote(ns.arm)} {ns.decision}",
                       r, tty=False)


def logs(ns, r: Runner) -> int:
    if ns.what in ("show", "tail") and not ns.run_id:
        raise VerbError(EXIT_USAGE, f"logs {ns.what} needs a run id")

    def command(tool: str) -> str:
        return {"last": f"{tool} last {ns.workflow}", "list": f"{tool} list {ns.workflow}",
                "show": f"{tool} show {shlex.quote(ns.run_id or '')}",
                "tail": f"{tool} tail -f {shlex.quote(ns.run_id or '')}"}[ns.what]

    if ns.workflow == "ros-cli":  # deploy/up/down/relay-install runs live where they ran: here
        argv = shlex.split(command(shlex.quote(sys.executable) + " " + shlex.quote(str(REPO / "scripts/lib/tatbot_runlog.py"))))
        r.say(f"[local] {shlex.join(argv)}")
        if r.dry:
            return 0
        os.execvp(argv[0], argv)
    t = ros_target()
    argv = t.shell(f"exec {command(f'python3 {t.root}/repo/scripts/lib/tatbot_runlog.py')}",
                   tty=ns.what == "tail" and sys.stdin.isatty())
    r.say(f"[{t.node}] {shlex.join(argv)}")
    if r.dry:
        return 0
    t.reachable(r)
    os.execvp(argv[0], argv)


def probe_dests() -> list[str]:
    """The probe relay's reader: the ros node's driver, on PROBE_PORT."""
    nmap = fleet.load(REPO)
    name = owner("ros")
    return [f"{nmap[name]['lan']}:{PROBE_PORT}"] if nmap[name].get("lan") else []


def probe_install(ns, r: Runner) -> None:
    """The station probe's relay: the same script as its own instance, directory and unit, so that
    installing or restarting it never touches the e-stop relay. Frames go to every --dest (default
    probe_dests)."""
    dests = ns.dest or probe_dests()
    if not dests:
        raise VerbError(EXIT_USAGE, "no --dest, and neither the ros node nor the arm node has a `lan` in "
                                    "config/nodes.json")
    gpio = stack_defaults().get("probe_gpio") or "GPIO27"
    if not re.fullmatch(r"\w+", gpio):
        raise VerbError(EXIT_USAGE, f"{STACK_YAML} probe.relay_gpio wants a GPIO line name such as GPIO27, not {gpio!r}")
    args = f"--probe {gpio}" + "".join(f" --dest {d}" for d in dests) + (f" --bind {ns.bind}" if ns.bind else "")
    install_instance(r, "probe", args, PROBE_UNIT, "tatbot station probe relay (PRB1 -> UDP)")


def install_instance(r: Runner, name: str, args: str, unit: str, description: str) -> None:
    """The relay script as its own instance, directory and unit on the estop-relay node, so that installing or
    restarting it never touches the e-stop relay."""
    t = Target(owner("estop-relay"), root=f"~/tatbot-{name}-relay")
    r.say(f"# {name} relay on {t.node}: {t.root_arg}/tatbot_estop_relay.py {args}")
    r.step("dir", t.shell(f"mkdir -p {t.root}"))
    relay = REPO / "ros/tatbot_estop_relay"
    for source in (relay / "tatbot_estop_relay.py", relay / f"{RELAY_UNIT}.in"):
        r.step(f"copy {source.name}", t.shell(f"cat > {t.root}/{source.name}"), stdin=source.read_text())
    r.step("unit", t.shell(
        f"sed -e \"s|@USER@|$USER|g\" -e \"s|@DIR@|{t.root}|g\" -e \"s|@ARGS@|{args}|g\" "
        f"-e \"s|^Description=.*|Description={description}|\" {t.root}/{RELAY_UNIT}.in "
        f"| sudo tee /etc/systemd/system/{unit} >/dev/null && sudo systemctl daemon-reload "
        f"&& sudo systemctl enable {unit} && sudo systemctl restart {unit} && sleep 1 "
        f"&& systemctl is-active {unit}"))


def machine_install(ns, r: Runner) -> None:
    """The tattoo machine's switch: commands from the ros node's `lan` only, the e-stop from the e-stop relay
    over the Pi's loopback (estop_dests)."""
    d = stack_defaults()
    gpio, lan = d.get("machine_gpio", ""), fleet.load(REPO)[owner("ros")].get("lan")
    if not re.fullmatch(r"\w+", gpio):
        raise VerbError(EXIT_USAGE, f"{STACK_YAML} machine.gpio names no GPIO line that powers the machine: wire it, "
                                    "then record it (such as GPIO22)")
    if not lan:
        raise VerbError(EXIT_USAGE, f"the ros node {owner('ros')} has no `lan` in config/nodes.json to take commands from")
    args = (f"--machine {gpio} --from {lan} --listen {d.get('machine_port', 7642)} "
            f"--estop-port {d.get('machine_estop_port', 7643)} --timeout {d.get('machine_timeout_s', 0.2)}")
    install_instance(r, "machine", args, MACHINE_UNIT, "tatbot tattoo machine switch (MCH1 -> GPIO)")


def keys_install(r: Runner) -> None:
    """The station keyboard's relay on the station node: its drawing keys to the keypad node at the ros node's
    `lan` and stack.yaml keypad.port."""
    lan, port = fleet.load(REPO)[owner("ros")].get("lan"), stack_defaults().get("keypad_port")
    if not lan:
        raise VerbError(EXIT_USAGE, f"the ros node {owner('ros')} has no `lan` in config/nodes.json to send keys to")
    if not port or port == "0":
        raise VerbError(EXIT_USAGE, f"{STACK_YAML} keypad.port is 0: the stack runs no keypad node to send keys to")
    t, args = Target(owner("station"), root="~/tatbot-keys-relay"), f"--dest {lan}:{port}"
    r.say(f"# keys relay on {t.node}: {t.root_arg}/tatbot_keys_relay.py {args}")
    r.step("dir", t.shell(f"mkdir -p {t.root}"))
    relay = REPO / "ros/tatbot_keys_relay"
    for source in (relay / "tatbot_keys_relay.py", relay / f"{KEYS_UNIT}.in"):
        r.step(f"copy {source.name}", t.shell(f"cat > {t.root}/{source.name}"), stdin=source.read_text())
    r.step("unit", t.shell(
        f"sed -e \"s|@USER@|$USER|g\" -e \"s|@DIR@|{t.root}|g\" -e \"s|@ARGS@|{args}|g\" {t.root}/{KEYS_UNIT}.in "
        f"| sudo tee /etc/systemd/system/{KEYS_UNIT} >/dev/null && sudo systemctl daemon-reload "
        f"&& sudo systemctl enable {KEYS_UNIT} && sudo systemctl restart {KEYS_UNIT} && sleep 1 "
        f"&& systemctl is-active {KEYS_UNIT}"))


def arm_estop_port() -> int | None:
    """The UDP port the arm node's monitor reads the relay on when its profile e-stop is this relay
    (driver.estop_device `udp://[HOST]:PORT?from=...`); None for a serial e-stop or no profile."""
    import tatbot_profile
    try:
        device = str((tatbot_profile.load(REPO).get("driver") or {}).get("estop_device") or "")
    except tatbot_profile.ProfileError:
        return None
    return (urllib.parse.urlsplit(device).port or UDP_PORT) if device.startswith("udp://") else None


def estop_dests() -> list[str]:
    """The relay's readers: the ros node's driver; when the arm node's profile e-stop is the relay, every node
    with role `estop` at that port, so one button stops both arms; and the tattoo machine's switch on the Pi
    itself, over its loopback, whether or not it is installed."""
    nmap = fleet.load(REPO)
    lan = nmap[owner("ros")].get("lan")
    dests = [f"{lan}:{UDP_PORT}"] if lan else []
    port = arm_estop_port()
    for name in fleet.nodes_with(nmap, "estop") if port else []:
        if not nmap[name].get("lan"):
            raise VerbError(EXIT_USAGE, f"{name} reads the e-stop relay (its profile e-stop is udp) but has no "
                                        "`lan` in config/nodes.json")
        dests.append(f"{nmap[name]['lan']}:{port}")
    dests.append(f"127.0.0.1:{stack_defaults().get('machine_estop_port', 7643)}")
    return list(dict.fromkeys(dests))


def relay_install(ns, r: Runner) -> None:
    ros = owner("ros")
    if ns.bind and not re.fullmatch(r"[0-9.]+", ns.bind):
        raise VerbError(EXIT_USAGE, f"--bind wants the relay node's IPv4 source address, not {ns.bind!r}")
    if ns.keys:
        keys_install(r)
        return
    if ns.probe or ns.machine:
        (probe_install if ns.probe else machine_install)(ns, r)
        return
    dests = ns.dest or estop_dests()
    if not dests:
        raise VerbError(EXIT_USAGE, f"no --dest and the ros node {ros} has no `lan` in config/nodes.json")
    gpio = stack_defaults().get("relay_gpio")
    if gpio and not re.fullmatch(r"\w+", gpio):
        raise VerbError(EXIT_USAGE, f"{STACK_YAML} estop.relay_gpio wants a GPIO line name such as GPIO17, not {gpio!r}")
    args = (f"{f'--gpio {gpio}' if gpio else '--device /dev/tatbot-estop'}" + "".join(f" --dest {d}" for d in dests)
            + f"{f' --bind {ns.bind}' if ns.bind else ''}")
    t = Target(owner("estop-relay"), root="~/tatbot-estop-relay")
    r.say(f"# relay on {t.node}: {t.root_arg}/tatbot_estop_relay.py {args} (each reader accepts frames only from "
          f"the relay's address: tatbot ros up --estop udp --relay-addr <it>, and the arm node's udp profile e-stop)")
    r.step("dir", t.shell(f"mkdir -p {t.root}"))
    relay = REPO / "ros/tatbot_estop_relay"
    for source in (relay / "tatbot_estop_relay.py", relay / f"{RELAY_UNIT}.in", REPO / UDEV_RULE):
        # over the ssh pipe: the relay node needs nothing but ssh, python3 and sudo
        r.step(f"copy {source.name}", t.shell(f"cat > {t.root}/{source.name}"), stdin=source.read_text())
    if not gpio:   # a Pico's /dev/tatbot-estop
        r.step("udev", t.shell(install_udev(f"{t.root}/{Path(UDEV_RULE).name}")))
    r.step("unit", t.shell(
        f"sed -e \"s|@USER@|$USER|g\" -e \"s|@DIR@|{t.root}|g\" -e \"s|@ARGS@|{args}|g\" "
        f"{t.root}/{RELAY_UNIT}.in "
        f"| sudo tee /etc/systemd/system/{RELAY_UNIT} >/dev/null && sudo systemctl daemon-reload "
        f"&& sudo systemctl enable {RELAY_UNIT} && sudo systemctl restart {RELAY_UNIT} && sleep 1 "
        f"&& systemctl is-active {RELAY_UNIT}"))


# --- entry --------------------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tatbot ros", description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true", help="print every command; run nothing")
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("deploy")
    s.add_argument("--no-build", action="store_true")
    s.add_argument("--units", action="store_true")
    s.add_argument("--clean", action="store_true")
    s = sub.add_parser("up")
    s.add_argument("--hardware", choices=("mock", "fake", "real", "sim"))
    s.add_argument("--config", help="hardware sim: the run's stack.yaml on the ros-sim owner")
    s.add_argument("--estop", choices=("none", "serial", "udp"))
    s.add_argument("--relay-addr")
    s.add_argument("--page", choices=("stencil", "fixed"))
    s.add_argument("--pattern")
    s.add_argument("--surface", choices=("plane", "fitted"))
    s.add_argument("--arms")
    s.add_argument("--probe", action="store_true")
    s.add_argument("--no-touch", action="store_true")
    s.add_argument("--rerun", action="store_true")
    s.add_argument("--hold", action="store_true", help="restart a running stack without landing its arms first")
    s.add_argument("--tuning")
    s = sub.add_parser("down")
    s.add_argument("--router", action="store_true")
    s.add_argument("--hold", action="store_true", help="stop without landing the arms first")
    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("page")
    s.add_argument("--arm", default="right")
    s.add_argument("--seconds", type=float, default=5.0)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("compile")
    s.add_argument("design", nargs="+")
    s.add_argument("--arm", default="right")
    s.add_argument("--ee-tool")
    s.add_argument("--speed")
    s.add_argument("--inks")
    s.add_argument("--max-segment-s")
    s.add_argument("--max-stroke-mm")
    s.add_argument("--min-stroke-mm")
    s.add_argument("--mark-width-mm")
    s.add_argument("-o", "--out")
    s.add_argument("--at")
    s.add_argument("--width")
    s.add_argument("--stencil")
    s.add_argument("--keep-region")
    s = sub.add_parser("draw")
    s.add_argument("program", nargs="?")
    s.add_argument("--detach", action="store_true")
    s.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    s.add_argument("--from", dest="start", choices=("register", "overhead", "station", "height", "gauge", "page"))
    s.add_argument("--progress", action="store_true")
    s.add_argument("--json", action="store_true")
    s.add_argument("--at")
    s.add_argument("--width")
    s.add_argument("--stencil")
    s.add_argument("--speed")
    s.add_argument("--arm", default="right")
    s.add_argument("--ee-tool")
    s.add_argument("--from-op")
    s.add_argument("--resume")
    s.add_argument("--no-inspect", dest="inspect", action="store_false")
    s.add_argument("--rehearse", action="store_true",
                   help="draw it in the air: the pen held over the page, the machine off, no dips, no inspection")
    s.add_argument("--standoff-mm", dest="standoff_mm", type=float,
                   help="with --rehearse: the pen this far over the measured page (default motion.yaml pen.hover)")
    s.add_argument("--hold", dest="land", action="store_false",
                   help="leave the arm holding at rest after the draw instead of landing it (idle)")
    s.add_argument("--no-wake", dest="wake", action="store_false",
                   help="use the current runtime without automatically restarting a landed arm")
    s = sub.add_parser("ready")
    s.add_argument("--arm", default="right")
    s = sub.add_parser("station")
    s.add_argument("--arm", default="right")
    s.add_argument("--against", default="")
    s = sub.add_parser("register")
    s.add_argument("--arm", default="right")
    s.add_argument("--center", type=float, nargs=2)
    s.add_argument("--table-z", dest="table_z", type=float)
    s.add_argument("--spread", type=float)
    s.add_argument("--heights", type=float, nargs="+")
    s.add_argument("--max-holds", dest="max_holds", type=int)
    s.add_argument("--prior", default="")
    s.add_argument("--holds", type=int)
    s.add_argument("--adopt", action="store_true")
    s.add_argument("--plan-only", dest="plan_only", action="store_true")
    s = sub.add_parser("setup")
    s.add_argument("--arm", default="right")
    s.add_argument("--ee-tool", dest="ee_tool", default="")
    s.add_argument("--hardware", choices=("mock", "fake", "real"))
    s.add_argument("--estop", choices=("none", "serial", "udp"))
    s.add_argument("--relay-addr")
    s.add_argument("--pattern")
    s.add_argument("--surface", choices=("plane", "fitted"))
    s.add_argument("--arms")
    s.add_argument("--rerun", action="store_true")
    s.add_argument("--detach", action="store_true")
    s.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    s.add_argument("--from", dest="start", choices=("register", "overhead", "station", "height", "gauge", "page"))
    s.add_argument("--progress", action="store_true")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("chain")
    s.add_argument("runs", nargs="+")
    s.add_argument("--arm", default="right")
    s = sub.add_parser("calib")
    s.add_argument("action", choices=("run", "sweep", "height", "wristcam", "gauge", "overhead", "apply"))
    s.add_argument("--run", default="")
    s.add_argument("--arm", default="right")
    s.add_argument("--ee-tool", dest="ee_tool", default="")
    s.add_argument("--attitudes", default="")
    s.add_argument("--heading-deg", dest="heading_deg", default="")
    s.add_argument("--station-only", dest="station_only", action="store_true")
    s.add_argument("--pattern", default="")
    s.add_argument("--adopt", action="store_true")
    s = sub.add_parser('palette')
    s.add_argument('action', choices=('status', 'load'))
    s.add_argument('caps', nargs='*')
    s.add_argument('--level-mm', action='append', default=[])
    s = sub.add_parser("touch")
    s.add_argument("--arm", default="right")
    s.add_argument("--start", type=float, nargs=3)
    s.add_argument("--rpy", type=float, nargs=3)
    s.add_argument("--direction", type=float, nargs=3)
    s.add_argument("--prior", type=float)
    s.add_argument("--probe", action="store_true")
    s.add_argument("--move", action="store_true")
    s.add_argument("--speed", type=float)
    s.add_argument("--max-travel", dest="max_travel", type=float)
    s = sub.add_parser("jog")
    s.add_argument("--arm", default="right")
    s.add_argument("--joint", type=int, required=True)
    s.add_argument("--delta", type=float, required=True)
    s.add_argument("--speed", type=float)
    s = sub.add_parser("inspect")
    s.add_argument("--depth-only", action="store_true")
    s.add_argument("--arm", default="right")
    s.add_argument("--run")
    s.add_argument("--frames", type=int)
    s.add_argument("--poses", type=int)
    s = sub.add_parser("evidence")
    s.add_argument("--page", required=True)
    s.add_argument("--slot", required=True)
    s = sub.add_parser("decide")
    s.add_argument("arm")
    s.add_argument("decision", choices=("continue", "land", "skip", "redraw"))
    sub.add_parser("cancel")
    s = sub.add_parser("logs")
    s.add_argument("what", nargs="?", default="last", choices=("last", "list", "show", "tail"))
    s.add_argument("run_id", nargs="?")
    s.add_argument("--workflow", default="ros-draw", choices=("ros-draw", "ros-touch", "ros-stack", "ros-cli", "ros-station",
                                                                "ros-calib", "ros-register", "ros-chain"))
    s = sub.add_parser("relay-install")
    s.add_argument("--dest", action="append")
    s.add_argument("--bind")
    s.add_argument("--probe", action="store_true", help="install the station probe's relay (its own unit)")
    s.add_argument("--machine", action="store_true", help="install the tattoo machine's switch (its own unit)")
    s.add_argument("--keys", action="store_true", help="install the station keyboard's drawing keys (station node)")
    return p


LOGGED = {"deploy": deploy, "up": up, "down": down, "relay-install": relay_install, "ready": ready, "station": station,
          "calib": calib, "register": register, "setup": setup}
DIRECT = {"evidence": evidence, "status": status, "page": page, "compile": compile_, "draw": draw, "touch": touch, "jog": jog, "inspect": inspect, "decide": decide,
          "cancel": cancel, "logs": logs, "chain": chain, "palette": palette}


def main(argv: list[str]) -> int:
    ns = parser().parse_args(argv)
    try:
        if ns.verb in OWN_SESSION and not ns.child and (ns.progress or not ns.dry_run):   # the run is its own child
            return (setup_progress(ns, Runner(ns.dry_run)) if ns.progress
                    else setup_detach(ns, Runner(ns.dry_run))) or 0
        if ns.verb in LOGGED or ns.verb == "draw" and ns.child:   # a draw's child keeps a run log to follow
            return with_runlog(ns.verb, ns.dry_run, lambda r: {**LOGGED, "draw": draw}[ns.verb](ns, r))
        return DIRECT[ns.verb](ns, Runner(ns.dry_run)) or 0
    except VerbError as error:
        print(f"tatbot ros {ns.verb}: {error}", file=sys.stderr)
        return error.code
    except subprocess.CalledProcessError as error:
        print(f"tatbot ros {ns.verb}: {shlex.join(map(str, error.cmd))} failed (exit {error.returncode})", file=sys.stderr)
        return EXIT_UNREACHABLE if error.returncode == 255 else error.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
