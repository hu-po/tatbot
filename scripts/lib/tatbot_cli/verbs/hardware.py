"""estop · arm · tool — the bench-level hardware verbs (ink has its own module)."""

from __future__ import annotations

import json
import sys

from tatbot_cli import EXIT_HW_UNREACHABLE, EXIT_OK, EXIT_USAGE
from tatbot_cli.registry import MOTION_HUMAN, MUTATES_CONFIG, OFFLINE, SENSOR, Plan, verb
from tatbot_cli.verbs._common import py, sh, tool_flag

# Arm addresses come from the hardware profile — the same source the
# launchers use, so `arm recover` can never command a different address
# than the profile-driven stack (audit 2026-08-31, finding 3).

def _profile_driver():
    """Driver stanza from the resolved hardware profile ({} when none)."""
    import tatbot_profile

    from tatbot_cli.registry import repo_root
    try:
        p = tatbot_profile.load(repo_root())
    except tatbot_profile.ProfileError:
        return {}
    return p.get("driver") or {}


def _arms() -> dict:
    d = _profile_driver()
    out = {}
    if d.get("leader_ip"):
        out["leader"] = str(d["leader_ip"])
    if d.get("follower_ip"):
        out["follower"] = str(d["follower_ip"])
    return out

ARMS = _arms()

# --- profile ---------------------------------------------------------------------


def _profile_args(p):
    p.add_argument("name", nargs="?", help="profile to inspect (default: the resolved one)")


@verb(effects=('read_files',), visibility="public", native=True, output="json", noun="profile", verb="show", tier=OFFLINE, summary="the resolved hardware profile: backend, arms, e-stop, provenance",
      wraps=("scripts/lib/tatbot_profile.py",), args=_profile_args, example=(), doc="docs/robot.md")
def profile_show(ctx, ns, rest):
    import tatbot_profile
    try:
        p = tatbot_profile.load(ctx.repo, ns.name)
    except tatbot_profile.ProfileError as e:
        print(f"profile: {e}", file=sys.stderr)
        return EXIT_USAGE
    if ctx.json:
        print(json.dumps(p, indent=2))
        return EXIT_OK
    errs = tatbot_profile.hardware_errors(p)
    print(f"profile   {p['name']}  ({p['_path']})")
    driver = p.get("driver") or {}
    print(f"  backend {driver.get('backend') or '-'}")
    print(f"  leader  {driver.get('leader_ip') or '(not stated)'}")
    print(f"  follower {driver.get('follower_ip') or '(not stated)'}")
    print(f"  e-stop  {driver.get('estop_device') or '(not stated)'}")
    arm = p.get("arm") or {}
    print(f"  arm     {arm.get('model', '-')} {arm.get('dof', '')}dof  "
          f"limits={(arm.get('limits') or {}).get('source', '-')}")
    print(f"  provenance {arm.get('provenance', '-')}")
    print("  hardware: " + ("YES — this profile may drive an arm" if not errs
                            else "NO — " + "; ".join(errs)))
    return EXIT_OK


@verb(effects=('read_files',), visibility="public", native=True, output="json", noun="profile", verb="list", tier=OFFLINE, summary="the profiles this checkout carries",
      wraps=("scripts/lib/tatbot_profile.py",), example=(), doc="docs/robot.md")
def profile_list(ctx, ns, rest):
    import tatbot_profile
    profiles = []
    for name in tatbot_profile.available(ctx.repo):
        try:
            p = tatbot_profile.load(ctx.repo, name)
            mark = "hardware" if not tatbot_profile.hardware_errors(p) else "synthetic/incomplete"
        except tatbot_profile.ProfileError as e:
            mark = f"INVALID: {e}"
        profiles.append({"name": name, "status": mark})
    if ctx.json:
        print(json.dumps({"schema": "tatbot.profiles/1", "profiles": profiles}))
    else:
        for p in profiles:
            print(f"{p['name']:<16} {p['status']}")
    return EXIT_OK


@verb(effects=('read_files',), visibility="public", native=True, output="json", noun="profile", verb="check", tier=OFFLINE, summary="can the resolved profile drive hardware? (exit 3 if not)",
      wraps=("scripts/lib/tatbot_profile.py",), args=_profile_args, example=(), doc="docs/robot.md",
      invariants=("The same validation the motion gate runs, without starting anything.",))
def profile_check(ctx, ns, rest):
    import tatbot_profile

    from tatbot_cli import EXIT_GATE_REFUSED
    try:
        p = tatbot_profile.load(ctx.repo, ns.name)
    except tatbot_profile.ProfileError as e:
        print(f"profile: {e}", file=sys.stderr)
        return EXIT_GATE_REFUSED
    errs = tatbot_profile.hardware_errors(p)
    if errs:
        print(f"profile '{p['name']}' cannot drive hardware: " + "; ".join(errs), file=sys.stderr)
        return EXIT_GATE_REFUSED
    if ctx.json:
        print(json.dumps({"schema": "tatbot.profile-check/1", "name": p["name"], "complete": True,
                          "env": tatbot_profile.env_exports(p)}))
    else:
        print(f"profile '{p['name']}' is complete: " +
              ", ".join(f"{k}={v}" for k, v in tatbot_profile.env_exports(p).items()))
    return EXIT_OK


# --- estop ---------------------------------------------------------------------


@verb(effects=('sensor_read',), noun="estop", verb="check", tier=SENSOR, summary="bench-check the Pico e-stop",
      role="estop", wraps=("scripts/estop_check.py",), passthrough="estop_check.py", example=(), doc="docs/estop.md",
      invariants=("Heartbeat silence = STOP; the box freezes every connected arm and never cuts power.",))
def estop_check(ctx, ns, rest):
    return py(ctx, "scripts/estop_check.py", *rest)


# --- arm -----------------------------------------------------------------------


def _recover_args(p):
    p.add_argument("role", nargs="?", choices=("leader", "follower"),
                   help="recover only this arm (default: follower, then leader)")
    p.add_argument("--ip", help="override the controller IP (requires an explicit role)")


@verb(effects=('read_files', 'network', 'human_motion', 'stop_process'), noun="arm", verb="recover", tier=MOTION_HUMAN,
      summary="stop active arm sessions, clear faults and land both arms staged→sleep→idle",
      role="arm", wraps=("scripts/il_recover_arms.sh", "scripts/il_recover_arm.sh", "cpp/teleop/arm_recover.cpp"),
      args=_recover_args, example=(), doc="docs/robot.md",
      invariants=("With no role, recovers follower first; leader starts only after follower success. An explicit role recovers only that arm.",
                  "Stops existing driver lease holders (SIGTERM, then SIGKILL after 10 seconds if needed) before acquiring exclusive ownership; this can stop a session using both arms even with an explicit role.",
                  "Keep the workspace clear: each arm moves slowly through staged and sleep.",
                  "Runs on the C++ arm stack (cpp/teleop/arm_recover, built by `scripts/check cpp`), not the LeRobot environment; same landing ritual the plugins use on a failed disconnect."))
def arm_recover(ctx, ns, rest):
    if ns.role is None:
        if ns.ip:
            print("arm recover: --ip requires an explicit leader or follower role", file=sys.stderr)
            return EXIT_USAGE
        return sh(ctx, "scripts/il_recover_arms.sh", *rest)
    ip = ns.ip or ARMS.get(ns.role)
    if not ip:
        print(f"arm recover: no {ns.role} address in the hardware profile — "
              "pass --ip or fix config/profiles", file=sys.stderr)
        return EXIT_HW_UNREACHABLE
    return sh(ctx, "scripts/il_recover_arm.sh", ip, ns.role, *rest)


@verb(effects=('network', 'device_config'), noun="arm", verb="set-ip", tier=MUTATES_CONFIG, summary="set a Trossen controller's IP address",
      role="arm", wraps=("cpp/teleop/arm_set_ip.cpp",), passthrough="arm_set_ip", example=("--", "--help"))
def arm_set_ip(ctx, ns, rest):
    return Plan(argv=[ctx.path("cpp/teleop/build/arm_set_ip"), *rest],
                notes=["built by `scripts/check cpp` (cmake -B cpp/teleop/build)"])


# --- tool ----------------------------------------------------------------------


@verb(effects=('read_files',), native=True, output="json", noun="tool", verb="list", tier=OFFLINE, summary="the tool datasheets in config/tools/",
      wraps=("scripts/lib/tool_spec.py",), example=(), doc="docs/tools.md")
def tool_list(ctx, ns, rest):
    from tatbot_cli import gates
    tools = list(gates.known_tools(ctx.repo))
    if ctx.json:
        print(json.dumps({"schema": "tatbot.tools/1", "tools": tools}))
    else:
        for t in tools:
            print(t)
    return EXIT_OK


def _tool_show_args(p):
    p.add_argument("tool_id", nargs="*")


@verb(effects=('read_files',), noun="tool", verb="show", tier=OFFLINE, summary="print a datasheet as the code sees it",
      wraps=("scripts/lib/tool_spec.py",), args=_tool_show_args, example=("lutin-ballpoint-dot",), doc="docs/tools.md")
def tool_show(ctx, ns, rest):
    return py(ctx, "scripts/lib/tool_spec.py", *ns.tool_id, *rest)


@verb(effects=('read_files',), noun="tool", verb="sync", tier=OFFLINE, summary="datasheet ↔ code agreement (carriage constants, measured tip)",
      wraps=("scripts/check_tool_sync.py",), passthrough="check_tool_sync.py", example=(), doc="docs/tools.md")
def tool_sync(ctx, ns, rest):
    return py(ctx, "scripts/check_tool_sync.py", *tool_flag(ctx), *rest)


@verb(effects=('read_files', 'write_config'), noun="tool", verb="urdf", tier=MUTATES_CONFIG, summary="put the fitted tool into urdf/tatbot.urdf so FK sees it",
      wraps=("scripts/gen_tool_urdf.py",), passthrough="gen_tool_urdf.py", example=("--", "--check"), doc="docs/tools.md")
def tool_urdf(ctx, ns, rest):
    return py(ctx, "scripts/gen_tool_urdf.py", *rest)
