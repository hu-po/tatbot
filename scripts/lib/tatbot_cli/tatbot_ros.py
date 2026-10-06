#!/usr/bin/env python3
"""Backend of `tatbot ros <verb>` (scripts/lib/tatbot_cli/verbs/ros.py). Stdlib only.

It runs on any node. A verb that acts on the stack resolves the node owning the `ros` role in
config/nodes.json and runs there over ssh inside the workspace root (~/tatbot-ros; never that node's
own git checkout), or locally on the owner. `--dry-run` prints every command and runs nothing.

Overrides for a parallel workspace (a lane, the integration run), all environment:
  TATBOT_ROS_ROOT    the workspace root on the ros node (default ~/tatbot-ros)
  TATBOT_ROS_DOMAIN  ROS_DOMAIN_ID written into its env.sh (default 0)
  TATBOT_ROS_PORT    its rmw_zenohd port on 127.0.0.1 (default 7450)
  TATBOT_ROS_NODE    another host for the stack than the `ros` role's owner: the arm node, to drive the
                     blue arm there (ROS plan decision 21); every verb, `deploy` included, acts on it
Every verb acts on that root. The installed units run ~/tatbot-ros; `up`/`down`/`status` on any other
root use transient units named after it (tatbot-ros-lane-<dir>[-router], systemd-run, same limits).

Exit codes: 0 ok, 2 usage, 4 no ros node in config/nodes.json, 5 the node is unreachable, else the
remote command's own code. deploy, up, down and relay-install write a `ros-cli` run log here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
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

VERBS = ("deploy", "up", "down", "status", "compile", "draw", "touch", "jog", "inspect", "decide", "logs",
         "relay-install", "evidence", "ready", "station", "calib", "register", "chain", "palette")
EXIT_USAGE, EXIT_GATE, EXIT_WRONG_NODE, EXIT_UNREACHABLE = 2, 3, 4, 5
# AGENTS.md marks repo/ as a checkout for scripts/lib/tatbot_runlog.py, which then reads config/runlog.json
# (log_root ~/tatbot-logs); without it every run on the ros node lands in ~/.local/state/tatbot/logs.
# rust/visiond/config: the D405 registry, which the calibration reads to find its arm's wrist camera
DEPLOY_DIRS = ("ros", "config", "urdf", "scripts", "python/tatbot_contracts", "rust/visiond/config", "AGENTS.md")
UNITS = ("tatbot-ros-router.service", "tatbot-ros.service")
RELAY_UNIT = "tatbot-estop-relay.service"
PROBE_UNIT = "tatbot-probe-relay.service"   # the same relay script as its own instance: never the e-stop's
MACHINE_UNIT = "tatbot-machine-relay.service"   # and the tattoo machine's switch, its own instance again
REGISTRATIONS = Path("~/tatbot-logs/vision").expanduser()
UDP_PORT = 7640  # stack.yaml estop.udp_port
PROBE_PORT = 7641  # stack.yaml probe.udp_port
UDEV_RULE = "config/udev/99-tatbot-estop.rules"  # /dev/tatbot-estop, wherever a Pico is read
KEYPAD_RULE = "config/udev/99-tatbot-keypad.rules"  # /dev/tatbot-keypad, the operator's numpad (stack.yaml keypad)
PRODUCTION_ROOT = "~/tatbot-ros"
STACK_YAML = "ros/tatbot_bringup/config/stack.yaml"


class VerbError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class Target:
    """A node and a workspace root there, and how to run things in it."""

    def __init__(self, node: str, *, root: str = "~/tatbot-ros", domain: str = "0", port: str = "7450"):
        nmap = fleet.load(REPO)
        self.node, self.domain, self.port = node, domain, port
        self.local = fleet.this_node(nmap) == node
        self.ssh = None if self.local else fleet.ssh_target(nmap, node)
        if not self.local and not self.ssh:
            raise VerbError(EXIT_WRONG_NODE, f"{node} has no ssh target in config/nodes.json")
        if not (root.startswith("~/") or root.startswith("/")):
            raise VerbError(EXIT_USAGE, f"workspace root must be ~/... or absolute, not {root!r}")
        self.root_arg = root
        self.production = root.rstrip("/") == PRODUCTION_ROOT
        if self.production:
            self.units = UNITS
        else:  # transient units for a lane's root, named after its directory
            slug = re.sub(r"[^A-Za-z0-9_-]", "-", root.rstrip("/").rsplit("/", 1)[-1])
            self.units = (f"tatbot-ros-lane-{slug}-router.service", f"tatbot-ros-lane-{slug}.service")
        # The root as a remote shell expands it, and as rsync/scp address it.
        self.root = "$HOME/" + root[2:] if root.startswith("~/") else root
        self.root_path = os.path.expanduser(root) if self.local else (root[2:] if root.startswith("~/") else root)

    def dest(self, sub: str = "") -> str:
        path = f"{self.root_path}/{sub}" if sub else self.root_path
        return path if self.local else f"{self.ssh}:{path}"

    def shell(self, script: str, *, tty: bool = False) -> list[str]:
        if self.local:
            return ["bash", "-lc", script]
        # ssh hands the script to the login user's shell (bash on the fleet); env.sh brings ROS.
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", *(["-t"] if tty else []), self.ssh, script]

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


def ros_target() -> Target:
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
def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--", *DEPLOY_DIRS], check=True,
                         capture_output=True).stdout.decode()
    return sorted(f for f in out.split("\0") if f and (REPO / f).is_file())


def rsync_filter(files: list[str]) -> str:
    """Include exactly `files` (and their directories), exclude everything else."""
    dirs: set[str] = set()
    for f in files:
        parts = f.split("/")[:-1]
        dirs.update("/".join(parts[:i]) for i in range(1, len(parts) + 1))
    return "".join(f"+ /{d}/\n" for d in sorted(dirs)) + "".join(f"+ /{f}\n" for f in files) + "- *\n"


def revision() -> str:
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *DEPLOY_DIRS],
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
    return (f"cd {t.root} && colcon build --base-paths repo/ros --build-base build --install-base install "
            f"--symlink-install {packages}--event-handlers console_direct- "
            f"--cmake-args -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_BASE_DIR={t.root}/deps")


def deploy(ns, r: Runner) -> None:
    t = ros_target()
    started = time.monotonic()
    r.say(f"# deploy {REPO} -> {t.node}:{t.root_arg} (domain {t.domain}, router port {t.port})")
    r.step("dirs", t.shell(f"mkdir -p {t.root}/repo/config {t.root}/calib {t.root}/inbox {t.root}/deps"))
    # The owner's palette declarations (tatbot ros palette load) stay where they are; a first deploy gets this
    # checkout's.
    sync = ["rsync", "-a", "--delete", "--delete-excluded", "--prune-empty-dirs", "--itemize-changes",
            "--filter=P /config/palette_load.yaml", "--exclude=/config/palette_load.yaml"]
    if r.dry:  # a dry run starts no process and writes no file: not even git or the filter file
        r.say(f"# the tracked files under {', '.join(DEPLOY_DIRS)} (git ls-files), as an rsync include filter")
        r.step("sync", [*sync, "--filter=merge <tracked-files filter>", f"{REPO}/", t.dest("repo/")])
        itemized = ""
    else:
        files = tracked_files()
        with tempfile.NamedTemporaryFile("w", suffix=".rsync-filter", delete=False) as fh:
            fh.write(rsync_filter(files))
        try:
            r.say(f"# {len(files)} tracked files under {', '.join(DEPLOY_DIRS)}")
            _, itemized = r.step("sync", [*sync, f"--filter=merge {fh.name}", f"{REPO}/", t.dest("repo/")], capture=True)
        finally:
            os.unlink(fh.name)
    changed = buildable(changed_packages(itemized), r)
    r.say(f"# changed: {len([ln for ln in itemized.splitlines() if ln.strip()])} paths; ros packages: "
          f"{', '.join(sorted(changed)) or 'none'}" if not r.dry else "# changed: (from rsync --itemize-changes)")
    # Recorded on the node before anything else can fail or be skipped (--no-build, Ctrl-C): the build
    # step builds these plus whatever an earlier deploy left, and clears the file only on success.
    if changed or r.dry:
        r.step("pending", t.shell(f"printf '%s\\n' {' '.join(sorted(changed)) or '<changed packages>'} "
                                  f">> {t.root}/.build-pending"))
    r.step("palette load", ["rsync", "-a", "--ignore-existing", str(REPO / "config/palette_load.yaml"),
                            t.dest("repo/config/palette_load.yaml")])
    registrations = sorted(REGISTRATIONS.glob("arm-registration-*-current.json"))
    if registrations:
        r.step("calib", ["rsync", "-a", *map(str, registrations), t.dest("calib/")])
    else:
        r.say(f"# no adopted registrations in {REGISTRATIONS}: each arm hangs at its nominal URDF mount")
    rev = "<git rev-parse HEAD> dirty=<0|1>" if r.dry else revision()
    r.say(f"# REVISION {' '.join(rev.split())}; env.sh from env.sh.in (domain {t.domain}, port {t.port}), on stdin")
    r.step("revision", t.shell(f"cat > {t.root}/repo/REVISION"), stdin=rev)
    r.step("env", t.shell(f"cat > {t.root}/env.sh"), stdin=env_sh(t))
    if not ns.no_build:
        if ns.clean:
            r.step("clean", t.shell(f"rm -rf {t.root}/build {t.root}/install {t.root}/log"))
        if r.dry:
            r.say("# build: every package on the first build or with --clean, else the packages rsync changed "
                  "and those above them (--packages-above); none when no ros/ file changed")
            r.step("build", t.shell(f"source /opt/ros/jazzy/setup.bash && {colcon(t, '')}"))
        else:
            # .build-pending holds every package synced but not yet built (this deploy's and any earlier one's).
            _, probe = r.step("probe", t.shell(f"test -f {t.root}/install/setup.bash && echo installed; "
                                               f"cat {t.root}/.build-pending 2>/dev/null; true"), capture=True)
            words = probe.split()
            first = ns.clean or "installed" not in words or "ALL" in words
            changed |= {w for w in words if w != "installed"}
            if first or changed:
                pending = "ALL" if first else " ".join(sorted(changed))
                select = "" if first else f"--packages-above {pending} "
                r.step("build", t.shell(f"echo {pending} > {t.root}/.build-pending && source /opt/ros/jazzy/setup.bash "
                                        f"&& {colcon(t, select)} && rm -f {t.root}/.build-pending"))
            else:
                r.say("# build: no ros/ package changed; nothing to build")
    elif not r.dry:
        r.say(f"# --no-build: {t.root_arg}/.build-pending keeps the synced packages for the next deploy")
    if ns.units:
        install_units(t, r)
    r.say(f"# deployed {rev.splitlines()[0][:12]} to {t.node}:{t.root_arg} in {time.monotonic() - started:.1f} s")


def install_units(t: Target, r: Runner) -> None:
    render = " && ".join(
        f"sed -e \"s|@USER@|$USER|g\" -e \"s|@HOME@|$HOME|g\" {t.root}/repo/ros/tatbot_bringup/systemd/{u}.in "
        f"| sudo tee /etc/systemd/system/{u} >/dev/null" for u in UNITS)
    r.step("units", t.shell(f"{render} && sudo systemctl daemon-reload && sudo systemctl enable {UNITS[0]} "
                            f"&& systemctl cat {UNITS[1]} | head -3"))
    r.step("udev", t.shell(install_udev(f"{t.root}/repo/{UDEV_RULE}")))
    r.step("udev keypad", t.shell(install_udev(f"{t.root}/repo/{KEYPAD_RULE}", "input", "/dev/tatbot-keypad",
                                               "plug the numpad in")))
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
                       ("arms", ns.arms)):
        if value:
            args.append(f"{key}:={value}")
    d = stack_defaults()   # stack.yaml's udp needs the relay too, except on mock hardware (no e-stop reader)
    implied_udp = not ns.estop and d.get("estop") == "udp" and (ns.hardware or d.get("hardware")) != "mock"
    if ns.estop == "udp" or implied_udp:
        args.append(f"estop_relay_addr:={relay_addr(ns, '--estop udp')}")
    elif ns.relay_addr:
        args.append(f"estop_relay_addr:={ns.relay_addr}")
    if ns.probe:   # the probe and machine relays are the same Pi as the e-stop relay: its `lan`
        args += ["probe:=true", f"probe_relay_addr:={relay_addr(ns, '--probe')}"]
    if d.get("machine_switch") == "pi" and not d.get("machine_addr"):
        args.append(f"machine_addr:={relay_addr(ns, 'machine.switch pi')}")
    if getattr(ns, "pattern_id", None):   # up() resolved --pattern on the ros node
        args.append(f"pattern_id:={ns.pattern_id}")
    if ns.no_touch:
        args.append("touch:=false")
    if ns.rerun:
        args.append("rerun:=true")
    return args


def relay_addr(ns, what: str) -> str:
    """--relay-addr, else the estop-relay node's `lan` (the palette Pi); `what` names the need in the refusal."""
    relay = ns.relay_addr or relay_lan()
    if not relay:
        raise VerbError(EXIT_USAGE, f"{what} needs the relay's address: pass --relay-addr ADDR, or record the "
                                    "estop-relay node's `lan` in config/nodes.json")
    return relay


def stack_defaults() -> dict:
    """hardware, estop.source, estop.relay_gpio, page.source, arms and machine.* from this checkout's stack.yaml
    (the one deploy sends). A line reader, so the backend stays stdlib."""
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
        elif key == "relay_gpio" and block in ("estop", "probe"):
            out["relay_gpio" if block == "estop" else "probe_gpio"] = value.strip().strip('"')
        elif block == "machine":
            out[f"machine_{key.strip()}"] = value.strip().strip('"')
    return out


def effective(ns) -> str:
    """What the stack will run with: the flags given, else stack.yaml."""
    d = stack_defaults()
    pattern = (("pattern", ns.pattern_id),) if getattr(ns, "pattern_id", None) else ()
    return " ".join(f"{k}={v}" for k, v in (("hardware", ns.hardware or d.get("hardware")),
                                            ("estop", ns.estop or d.get("estop")), ("page", ns.page or d.get("page")),
                                            ("arms", ns.arms or d.get("arms")), *pattern))


def relay_lan() -> str | None:
    nmap = fleet.load(REPO)
    relays = fleet.nodes_with(nmap, "estop-relay")
    return nmap[relays[0]].get("lan") if len(relays) == 1 else None


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

LAND_PROBE = r"""
{unit_up} || exit 0
echo "effective=$(sed -n 's/^TATBOT_ROS_EFFECTIVE=//p' {root}/stack.env 2>/dev/null)"
source {root}/env.sh && echo "client=$(timeout 15 ros2 run tatbot_session client status --json 2>/dev/null | tail -1)"
"""


def land_before_stop(t: Target, r: Runner, ns) -> None:
    """Land every arm of a running stack before it stops. The controller firmware idles when its driver's
    connection ends, so an arm that has not landed can drop when the stack process exits (README 8).
    Prints each arm's state; `--hold` stops without landing. A landing that fails leaves the stack up."""
    if ns.hold:
        refuse_hold_in_cap(t, r)
        r.say("# --hold: stopping without landing; an arm that has not landed can drop when its driver exits")
        return
    probe = LAND_PROBE.format(unit_up=UNIT_UP.format(unit=t.units[1]), root=t.root)
    if r.dry:
        r.step("arms", t.shell(probe))
        r.say("# then, when the stack runs hardware fake or real: `client land --arm <arm>` for each arm that has "
              "not landed and reports no controller error, before the stop (--hold skips this)")
        return
    _, out = r.step("arms", t.shell(probe), capture=True)
    report = parse_status(out, t.node)
    if "effective" not in report:
        return  # the stack is not running: nothing to land
    safety = (report.get("stack") or {}).get("safety") or {}
    r.say(f"# the stack runs with {report['effective'] or '(stack.yaml defaults)'}")
    if not safety:
        r.say("# no SafetyState from the stack: nothing to land")
        return
    mock = "hardware=mock" in (report["effective"] or "hardware=" + (stack_defaults().get("hardware") or ""))
    for arm, row in sorted(safety.items()):
        r.say(f"# {arm}: landed={row.get('landed')} latched={row.get('latched')} ({row.get('latch_reason')}) "
              f"estop_ok={row.get('estop_ok')} controller_error={row.get('controller_error')}")
        if mock or row.get("landed"):
            continue
        if row.get("controller_error"):
            r.say(f"# {arm}: the controller reports an error and refuses to land; it holds until the stack stops")
            continue
        try:
            r.step(f"land {arm}", t.shell(t.in_env(f"timeout 150 ros2 run tatbot_session client land --arm {arm}")))
        except VerbError as error:
            if error.code == EXIT_UNREACHABLE:
                raise
            raise VerbError(1, f"{arm} did not land ({error}); the stack is still up and holding. Release the "
                               f"e-stop and run `tatbot ros decide {arm} land`, or stop without landing: --hold") from None


def refuse_hold_in_cap(t: Target, r: Runner) -> None:
    """An arm that has not landed sags when its driver exits: 9 mm down and 6.7 mm aside at the palette on
    2026-10-05, with the nozzle in a cap. The session's in-cap marker names that state; stopping unlanded is
    refused (exit 3) while it stands."""
    _, out = r.step("in-cap", t.shell("ls $HOME/.local/state/tatbot/in-cap-*.json 2>/dev/null || true"), capture=True)
    if out.strip() and not r.dry:
        raise VerbError(EXIT_GATE, "a tool may be in a palette cap (" + out.strip().replace("\n", ", ") + "): "
                        "stopping without landing would drop it into the cap. Withdraw it first: resume the run "
                        "(tatbot ros draw <program> --resume <run>), whose next draw withdraws before anything")


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
    t = ros_target()
    if getattr(ns, "pattern", None):
        ns.pattern_id = installed_pattern(t, r, ns.pattern)
    args = " ".join(launch_args(ns))
    eff = effective(ns)
    r.say(f"# up on {t.node}:{t.root_arg} ({' and '.join(t.units)}): {eff}")
    if "hardware=real" in eff and "estop=none" in eff:
        r.say("# note: hardware real with estop none: no e-stop input; the arm's rocker switch is the only stop "
              "(--estop udp|serial)")
    r.say(f"#   ros2 launch tatbot_bringup stack.launch.py {args}".rstrip())
    r.say(f"# {t.root_arg}/stack.env: TATBOT_ROS_ARGS={args} (on stdin)")
    land_before_stop(t, r, ns)  # a running stack lands before the restart
    r.step("stack.env", t.shell(f"cat > {t.root}/stack.env"), stdin=f"TATBOT_ROS_ARGS={args}\nTATBOT_ROS_EFFECTIVE={eff}\n")
    restart(t, r)


def restart(t: Target, r: Runner) -> None:
    """(Re)start the router and the stack with the stack.env already on the node."""
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
# and refused the draw (2026-10-06).
STACK_PATTERN = r"""pattern=$(sed -n 's/.*pattern_id:=\([^ ]*\).*/\1/p' {root}/stack.env 2>/dev/null)"""
READY_PAGE = STACK_PATTERN + r"""
for i in $(seq 1 60); do
  timeout 10 ros2 run tatbot_bridge page_watch --seconds 2 --json ${{pattern:+--pattern-id "$pattern"}} 2>/dev/null | grep -q '"measured": [1-9]' && exit 0
  sleep 1
done
echo "no measured page 3 min after the restart"; exit 1
"""


def page_on_bus(effective: str) -> bool:
    """Whether a restarted stack's page arrives on the bus: the stencil tracker's does, a fixed page never
    does. A stack.env with no page source (written before TATBOT_ROS_EFFECTIVE) waits, as it always did."""
    source = next((item.partition("=")[2] for item in effective.split() if item.startswith("page=")), "")
    return source in ("", "stencil")


def ready_wait(root: str, arm: str, page: bool) -> str:
    return READY_WAIT.format(root=root, arm=shlex.quote(arm)) + (READY_PAGE.format(root=root) if page else "exit 0\n")


def wake(t: Target, r: Runner, arm: str, page: bool = True, rest: bool = False) -> None:
    """A draw on an arm that landed (the end of the last draw, or a Decide LAND): the driver keeps a landed arm
    idle until it is woken, so wake that arm alone (`client wake`: its controllers and hardware cycled, the other
    arm untouched; a stack restart would take the other arm down mid-goal, 2026-09-30) and wait for it, and for a
    stencil page on the bus (a fixed page is never there) unless `page` is False: a calibration draws on no page.
    With `rest` an arm that has not landed lands first, so it is taken back at rest: a calibration's way to the
    station starts there (2026-09-29: from 0.3 m out, the planner's crossing to it met joint_1's limit)."""
    probe = LAND_PROBE.format(unit_up=UNIT_UP.format(unit=t.units[1]), root=t.root)
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


def down(ns, r: Runner) -> None:
    t = ros_target()
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
for u in {units}; do echo "unit:$u=$(systemctl is-active $u 2>/dev/null)"; done
newest=$(ls -1t ~/tatbot-logs/ros-draw 2>/dev/null | grep -v latest | head -1); echo "newest_run=$newest"
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
    script = STATUS_SCRIPT.format(root=t.root, root_arg=t.root_arg, units=" ".join(t.units),
                                  unit_up=UNIT_UP.format(unit=t.units[1]))
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


# --- offline, motion, decisions -----------------------------------------------------------------
def compile_argv(design: str, ns) -> tuple[list[str], dict]:
    argv = [str(REPO / 'scripts/lib/drawing_python.sh'), "-m", "tatbot_ink", "compile", design, "--arm", ns.arm]
    for flag, value in (("--ee-tool", ns.ee_tool), ("--speed", ns.speed), ("--inks", ns.inks), ("-o", ns.out),
                        ("--stencil", getattr(ns, "stencil", None))):
        if value:
            argv += [flag, value]
    if getattr(ns, "max_segment_s", None):
        argv += ["--max-segment-s", ns.max_segment_s]
    # `--at=X,Y` joined: a negative X would otherwise read as a flag
    argv += [f"{flag}={value}" for flag, value in (("--at", getattr(ns, "at", None)), ("--width", getattr(ns, "width", None)))
             if value]
    env = {"PYTHONPATH": os.pathsep.join([str(REPO / "ros/tatbot_ink"), str(REPO / "ros/tatbot_motion"), str(REPO / "ros/tatbot_description"), str(REPO / "python/tatbot_contracts/src"),
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
    argv = t.shell(t.in_env(command), tty=sys.stdin.isatty())
    r.say(f"[{t.node}] {shlex.join(argv)}")
    if r.dry:
        return 0
    t.reachable(r)
    # SIGINT too: a job a script starts in the background inherits SIGINT ignored.
    previous = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    proc = subprocess.Popen(argv)
    try:
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
    t = ros_target()
    program = program_file(ns.program, ns, r)
    stem = Path(ns.program).name.split(".")[0]
    name = f"{stem}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    copy = ["cp", str(program), t.dest(f"inbox/{name}")] if t.local else ["scp", "-q", "-o", "BatchMode=yes", str(program),
                                                                            t.dest(f"inbox/{name}")]
    r.step("inbox", t.shell(f"mkdir -p {t.root}/inbox"))
    r.step("send", copy)
    extra = "".join(f" {flag} {shlex.quote(value)}" for flag, value in (("--from-op", ns.from_op), ("--run-id", ns.resume))
                    if value)
    arm = shlex.quote(ns.arm)
    if ns.wake:
        wake(t, r, ns.arm)
    draw_cmd = f"ros2 run tatbot_session client draw {t.root}/inbox/{name} --arm {arm}{extra}"
    # A complete draw is looked at with the wrist camera (inspect ends at rest), then lands: sleep pose, controller
    # idle, motors off. --hold leaves it holding at rest. A draw that fails holds where it stopped (Decide).
    after = [f"ros2 run tatbot_session client inspect --arm {arm}"] if ns.inspect else []
    if ns.land:
        after.append(f"timeout 150 ros2 run tatbot_session client land --arm {arm}")
    if not after:
        return run_cancellable(t, f"exec {draw_cmd}", r)
    tail = "; ".join(f"{c} || rc=1" for c in after)
    return run_cancellable(t, f"{draw_cmd} && {{ rc=0; {tail}; exit $rc; }}", r)


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


def station_fix(arm: str, against: str, r: Runner) -> str:
    """One fresh station fix (see `station`); returns its run directory on the ros node."""
    t = ros_target()
    compare = f" --against {shlex.quote(against)}" if against else ""
    try:
        _, out = r.step("fix", t.shell(t.in_env(f"exec ros2 run tatbot_calib station fix --arm {shlex.quote(arm)}"
                                                 f"{compare}")))
    except VerbError as error:
        raise VerbError(error.code, STATION_REFUSED.get(error.code, str(error))) from None
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
        r.say(f"# adopted on {t.node}: the D555 owner restarted under bundle {done.get('bundle_id', '')[:12]}; "
              "copy the registration into this node's ~/tatbot-logs/vision before the next `tatbot ros deploy`, "
              "which carries that directory's registrations to the stack")
    return 0 if done.get("ok", r.dry) else 3


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
    r.say(f"# adopted {ns.run}: the {ns.arm} tip moved ({(tip[0] - tip0[0]) * 1000:+.2f}, "
          f"{(tip[1] - tip0[1]) * 1000:+.2f}) mm across its axis in {path} (commit and land it; `tatbot ros deploy` "
          "carries it to the stack)")
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
    argv = t.shell(t.in_env(f"exec ros2 run tatbot_calib sweep --arm {shlex.quote(ns.arm)} --station "
                             f"{opening}/station.json --run-started {started:.3f}{tool}"))
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
    if proc.wait() == 3:
        raise VerbError(3, "the tool cannot be calibrated on the station (the line above says why)")
    return 0 if done.get("ok") else 1


def calib(ns, r: Runner) -> int:
    """`calib run`: the arm's fitted tool tip across its axis, from side pairs on the station probe at upright yaws
    after a fresh station fix from the overhead D555. The stack must be up with the arm and --probe. A tool the
    station cannot probe is refused (exit 3) before anything moves. It writes candidate.json and report.md; `calib
    apply --run ID` writes that tip into this checkout's config/workspace.yaml when asked. `calib sweep` measures it
    contact-free (calib_sweep)."""
    if ns.action == "apply":
        return calib_apply(ns, r)
    if ns.action == "sweep":
        return calib_sweep(ns, r)
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
    """The probe relay's readers: the ros node's driver, and the arm node's, which runs the stack for the blue arm
    when that arm calibrates on the station (TATBOT_ROS_NODE; ROS plan decision 21). Each reads PROBE_PORT."""
    nmap = fleet.load(REPO)
    owners = [owner("ros"), *fleet.nodes_with(nmap, "arm")]
    return list(dict.fromkeys(f"{nmap[name]['lan']}:{PROBE_PORT}" for name in owners if nmap[name].get("lan")))


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
    s.add_argument("--hardware", choices=("mock", "fake", "real"))
    s.add_argument("--estop", choices=("none", "serial", "udp"))
    s.add_argument("--relay-addr")
    s.add_argument("--page", choices=("stencil", "fixed"))
    s.add_argument("--pattern")
    s.add_argument("--arms")
    s.add_argument("--probe", action="store_true")
    s.add_argument("--no-touch", action="store_true")
    s.add_argument("--rerun", action="store_true")
    s.add_argument("--hold", action="store_true", help="restart a running stack without landing its arms first")
    s = sub.add_parser("down")
    s.add_argument("--router", action="store_true")
    s.add_argument("--hold", action="store_true", help="stop without landing the arms first")
    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("compile")
    s.add_argument("design")
    s.add_argument("--arm", default="right")
    s.add_argument("--ee-tool")
    s.add_argument("--speed")
    s.add_argument("--inks")
    s.add_argument("--max-segment-s")
    s.add_argument("-o", "--out")
    s.add_argument("--at")
    s.add_argument("--width")
    s.add_argument("--stencil")
    s = sub.add_parser("draw")
    s.add_argument("program")
    s.add_argument("--at")
    s.add_argument("--width")
    s.add_argument("--stencil")
    s.add_argument("--speed")
    s.add_argument("--arm", default="right")
    s.add_argument("--ee-tool")
    s.add_argument("--from-op")
    s.add_argument("--resume")
    s.add_argument("--no-inspect", dest="inspect", action="store_false")
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
    s = sub.add_parser("chain")
    s.add_argument("runs", nargs="+")
    s.add_argument("--arm", default="right")
    s = sub.add_parser("calib")
    s.add_argument("action", choices=("run", "sweep", "apply"))
    s.add_argument("--run", default="")
    s.add_argument("--arm", default="right")
    s.add_argument("--ee-tool", dest="ee_tool", default="")
    s.add_argument("--attitudes", default="")
    s.add_argument("--heading-deg", dest="heading_deg", default="")
    s.add_argument("--station-only", dest="station_only", action="store_true")
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
    return p


LOGGED = {"deploy": deploy, "up": up, "down": down, "relay-install": relay_install, "ready": ready, "station": station,
          "calib": calib, "register": register}
DIRECT = {"evidence": evidence, "status": status, "compile": compile_, "draw": draw, "touch": touch, "jog": jog, "inspect": inspect, "decide": decide,
          "cancel": cancel, "logs": logs, "chain": chain, "palette": palette}


def main(argv: list[str]) -> int:
    ns = parser().parse_args(argv)
    try:
        if ns.verb in LOGGED:
            return with_runlog(ns.verb, ns.dry_run, lambda r: LOGGED[ns.verb](ns, r))
        return DIRECT[ns.verb](ns, Runner(ns.dry_run)) or 0
    except VerbError as error:
        print(f"tatbot ros {ns.verb}: {error}", file=sys.stderr)
        return error.code
    except subprocess.CalledProcessError as error:
        print(f"tatbot ros {ns.verb}: {shlex.join(map(str, error.cmd))} failed (exit {error.returncode})", file=sys.stderr)
        return EXIT_UNREACHABLE if error.returncode == 255 else error.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
