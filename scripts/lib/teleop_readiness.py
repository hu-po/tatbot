#!/usr/bin/env python3
"""Read-only teleop readiness facts, shared by `tatbot teleop check` and the launcher.

Nothing here constructs an arm driver, opens the e-stop serial stream, or
changes host configuration. It reads files, reads this session's own resource
limits, and — for the arms — sends the same one-packet ping the launcher already
sends. That bounds what it can say: a configured prerequisite being satisfied is
not evidence that the hardware is safe or ready to move, and only the launcher
and the control executable can report what the scheduler and the arms actually
do at start.

Two consumers, one model. `tatbot teleop check` renders every observation;
`scripts/teleop_start.sh` renders the real-time explanation when the executable
refuses, so the operator gets the same account of the same facts either way.
"""

from __future__ import annotations

import grp
import json
import os
import pwd
import re
import resource
import subprocess
import sys
from pathlib import Path

# The installed policy the fleet uses, and the tracked source it is copied from.
LIMITS_INSTALLED = Path("/etc/security/limits.d/99-tatbot-realtime.conf")
LIMITS_SOURCE = "config/limits/99-tatbot-realtime.conf"
# The control executable's own default, parsed rather than copied: a second
# constant here would drift from the priority the loop actually requests.
PRIORITY_SOURCE = "cpp/teleop/wxai_teleop.cpp"
PRIORITY_RE = re.compile(r"^\s*int\s+rt_priority\s*=\s*(\d+)\s*;", re.MULTILINE)
FALLBACK_PRIORITY = 80
# What /proc/<pid>/limits prints instead of a number for no cap.
UNLIMITED = 2 ** 31 - 1

OK, FAILED, UNKNOWN, NOT_APPLICABLE = "ok", "failed", "unknown", "not_applicable"


def observation(ident: str, label: str, state: str, value=None, reason: str | None = None,
                detail: list[str] | None = None) -> dict:
    """One fact. `reason` is the single line an operator reads first; `detail`
    carries the rest of the account, already split into lines so no consumer has
    to guess where a sentence ended."""
    return {"id": ident, "label": label, "state": state, "value": value,
            "reason": reason, "detail": list(detail or [])}


# --- real-time scheduling ---------------------------------------------------


def required_priority(repo: Path, requested: int | None = None) -> int:
    """The SCHED_FIFO priority this launch will ask for.

    An explicit `--rt-priority` wins; otherwise the executable's own default is
    read out of its source, so the number quoted to an operator is the number
    the loop requests. A checkout without the source falls back and says so
    through `priority_is_measured`.
    """
    if requested is not None:
        return requested
    try:
        found = PRIORITY_RE.search((repo / PRIORITY_SOURCE).read_text(errors="replace"))
    except OSError:
        return FALLBACK_PRIORITY
    return int(found.group(1)) if found else FALLBACK_PRIORITY


def priority_is_measured(repo: Path) -> bool:
    try:
        return bool(PRIORITY_RE.search((repo / PRIORITY_SOURCE).read_text(errors="replace")))
    except OSError:
        return False


def _account() -> tuple[str, set[str]]:
    """(login name, group names) for this process, as PAM would match them."""
    try:
        name = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        name = os.environ.get("USER") or str(os.getuid())
    groups = set()
    for gid in {os.getgid(), *os.getgroups()}:
        try:
            groups.add(grp.getgrgid(gid).gr_name)
        except KeyError:
            continue
    return name, groups


def parse_limits(text: str) -> list[dict]:
    """The `<domain> <type> rtprio <value>` rules in a limits.conf fragment."""
    rules = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        fields = line.split()
        if len(fields) == 4 and fields[2] == "rtprio":
            domain, kind, _, value = fields
            try:
                rules.append({"domain": domain, "type": kind, "value": int(value)})
            except ValueError:
                continue
    return rules


def limits_policy(path: Path = LIMITS_INSTALLED) -> dict:
    """What the installed limits fragment says, and whether it can reach us.

    "Applicable" is about PAM's matching rules only — domain `*`, this account's
    login name, or `@group` for a group it belongs to. Whether an applicable
    rule was actually applied to this session is a separate question, answered
    by the measured limit, not by this file.
    """
    account, groups = _account()
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {"path": str(path), "installed": False, "rules": [], "applicable": [],
                "account": account, "groups": sorted(groups)}
    rules = parse_limits(text)
    applicable = [r for r in rules
                  if r["domain"] == "*" or r["domain"] == account
                  or (r["domain"].startswith("@") and r["domain"][1:] in groups)]
    return {"path": str(path), "installed": True, "rules": rules, "applicable": applicable,
            "account": account, "groups": sorted(groups)}


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def login_handling() -> dict:
    """How this session was opened, and whether SSH logins run pam_limits.

    A limits fragment only reaches a session that PAM built. This reads the
    host's own configuration; it changes nothing and grants nothing.
    """
    sshd_config = _read("/etc/ssh/sshd_config")
    for extra in sorted(Path("/etc/ssh/sshd_config.d").glob("*.conf")) if Path(
            "/etc/ssh/sshd_config.d").is_dir() else []:
        sshd_config += "\n" + _read(str(extra))
    use_pam = re.search(r"(?mi)^\s*UsePAM\s+yes\b", sshd_config) is not None
    pam_limits = re.search(r"(?m)^\s*session\s+.*pam_limits\.so", _read("/etc/pam.d/sshd")) is not None
    return {"over_ssh": bool(os.environ.get("SSH_CONNECTION")),
            "sshd_uses_pam": use_pam, "sshd_pam_limits": pam_limits}


def peer_sessions() -> list[dict]:
    """This account's live login sessions on this host, and the limit each got.

    Measured from `/proc/<leader>/limits`, because the limit a session actually
    holds is the only thing that answers the question. It matters because this
    process usually is NOT the session an operator starts teleop from: a routed
    command runs in a one-shot `ssh host cmd` session, and on this fleet those
    do not receive the policy while interactive logins do.
    """
    try:
        listing = subprocess.run(["loginctl", "list-sessions", "--no-legend"],
                                 capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    if listing.returncode != 0:
        return []
    account, _ = _account()
    found = []
    for line in listing.stdout.splitlines():
        ident = line.split()[0] if line.split() else ""
        if not ident:
            continue
        try:
            shown = subprocess.run(["loginctl", "show-session", ident, "-p", "Name", "-p", "Type",
                                    "-p", "Remote", "-p", "Class", "-p", "Leader"],
                                   capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        record = dict(item.split("=", 1) for item in shown.stdout.splitlines() if "=" in item)
        if record.get("Name") != account or not record.get("Leader", "").isdigit():
            continue
        soft, hard = _process_rtprio(int(record["Leader"]))
        if soft is None:
            continue
        found.append({"session": ident, "type": record.get("Type", "?"),
                      "remote": record.get("Remote") == "yes", "class": record.get("Class", "?"),
                      "soft": soft, "hard": hard})
    return found


def _process_rtprio(pid: int) -> tuple[int | None, int | None]:
    try:
        text = Path(f"/proc/{pid}/limits").read_text(errors="replace")
    except OSError:
        return None, None
    label = "max realtime priority"
    for line in text.splitlines():
        if not line.lower().startswith(label):
            continue
        # This row carries no units column, unlike "Max realtime timeout ... us",
        # so index from the label rather than from the end of the line.
        fields = line[len(label):].split()[:2]
        try:
            values = [UNLIMITED if field == "unlimited" else int(field) for field in fields]
        except ValueError:
            return None, None
        return (values + [None, None])[:2]
    return None, None


def realtime_facts(repo: Path, requested: int | None = None) -> dict:
    """Everything known about this session's real-time scheduling, measured here."""
    soft, hard = resource.getrlimit(resource.RLIMIT_RTPRIO)
    priority = required_priority(repo, requested)
    sessions = peer_sessions()
    # The best any of this account's live sessions holds. A routed command runs
    # in a session an operator never starts teleop from, so judging the HOST by
    # this process's own limit answers the wrong question.
    capable = [row for row in sessions if row["soft"] >= priority]
    return {"priority": priority, "priority_measured": priority_is_measured(repo),
            "soft": soft, "hard": hard, "sufficient": soft >= priority,
            "sessions": sessions, "capable_sessions": capable,
            "host_capable": bool(capable),
            "limits": limits_policy(), "login": login_handling()}


def explain_realtime(facts: dict, *, node: str | None = None, from_check: bool = False) -> list[str]:
    """The operator-facing account: what failed, why it matters, what is next.

    The evidence decides the wording. An absent file gets the exact install
    command; an installed one gets the reason it did not take effect, because
    "install it again" is not a diagnosis and the file is already there.

    `from_check` says this is being rendered BY `tatbot teleop check`, so the
    next step must be something else: telling an operator to run the command
    they just ran is not advice.
    """
    priority, soft = facts["priority"], facts["soft"]
    where = f" on {node}" if node else ""
    if facts["sufficient"]:
        return [f"Real-time scheduling is available{where}: "
                f"the limit is {soft} and the control loop requires {priority}."]
    lines = [f"This session{where} cannot use real-time scheduling.",
             f"The control loop requires priority {priority}; this session's limit is {soft}."]
    limits, login = facts["limits"], facts["login"]
    if facts.get("host_capable"):
        # Measured, not inferred: another of this account's live sessions holds
        # the priority, so the policy works and only THIS kind of session misses
        # it. Saying "the host cannot" here would be false.
        best = max(facts["capable_sessions"], key=lambda row: row["soft"])
        kinds = ", ".join(sorted({f"{row['type']}{' (remote)' if row['remote'] else ''}"
                                  for row in facts["capable_sessions"]}))
        count = len(facts["capable_sessions"])
        held = (f"a live session here holds {best['soft']}" if count == 1
                else f"live sessions here hold {best['soft']}")
        lines.append(f"The host itself can: {held} ({kinds}), so the policy is in effect for "
                     "the sessions an operator logs into.")
        lines.append("A one-shot `ssh host command` session does not get it, which is exactly what "
                     "a routed command runs in.")
        target = node or "that node"
        lines.append(f"Next: log in to {target} and run `tatbot teleop start` there; "
                     "starting it from another node will refuse here for this reason.")
        return lines
    if not limits["installed"]:
        lines.append(f"No real-time limits policy is installed at {limits['path']}.")
        lines.append(f"Next: sudo cp {LIMITS_SOURCE} {limits['path']}")
        lines.append("Then open a NEW login session — limits are applied at login, "
                     "so the shell you are in keeps the old ones.")
        return lines
    if not limits["rules"]:
        lines.append(f"{limits['path']} is installed but sets no rtprio rule.")
        lines.append("Next: check that file against " + LIMITS_SOURCE)
        return lines
    if not limits["applicable"]:
        domains = ", ".join(sorted({r["domain"] for r in limits["rules"]}))
        lines.append(f"{limits['path']} grants rtprio to {domains}, and account "
                     f"'{limits['account']}' is not matched by any of them.")
        lines.append(f"That account is in: {', '.join(limits['groups']) or 'no groups'}.")
        lines.append("Next: add the account to a granted group, or add a rule for it, "
                     "then open a new login session.")
        return lines
    granted = max(r["value"] for r in limits["applicable"])
    lines.append(f"The limits file is installed and grants this account rtprio {granted}, "
                 "but its policy is not effective in this session.")
    if login["over_ssh"] and not login["sshd_uses_pam"]:
        # The more fundamental cause first: with UsePAM off, whether pam_limits
        # appears in the stack does not matter, because the stack never runs.
        lines.append("This is an SSH session and sshd is configured with UsePAM no, "
                     "so PAM never builds the session that would apply the limits.")
    elif login["over_ssh"] and not login["sshd_pam_limits"]:
        lines.append("This is an SSH session and /etc/pam.d/sshd does not call pam_limits.so, "
                     "so no limits fragment is applied to it.")
    elif login["over_ssh"]:
        lines.append("This is an SSH session, and sshd does call pam_limits.so, so the "
                     "reason has not been established here — a limit inherited from the "
                     "parent process or a service unit would also look like this.")
    else:
        lines.append("This is not an SSH session, so the limit was inherited from whatever "
                     "started it — a service unit or a shell opened before the policy landed.")
    if from_check:
        # The remaining causes differ in ONE observable way: whether a fresh
        # login session gets the limit. Name that, rather than a diagnosis this
        # has not established.
        # An SSH session already ran PAM, so another SSH session measures the
        # same thing. The console is the comparison that separates the two
        # remaining causes; neither is a fix to guess at from here.
        where = f"the console of {node}" if node else "that machine's own console"
        lines.append(f"Next: log in at {where} and run `ulimit -Sr -Hr` there.")
        lines.append("  90 at the console and 0 over SSH: PAM is not applying this file to SSH "
                     "sessions.")
        lines.append("  0 at both: the file is not being applied at all.")
    else:
        lines.append(f"Next: tatbot{' --on ' + node if node else ''} teleop check")
    lines.append("Do not work around this with --no-rt: that is a bench opt-out for a "
                 "hardware-free run, not a way to drive the arms.")
    return lines


# --- the other prerequisites ------------------------------------------------


def _pgrep(pattern: str) -> list[str]:
    try:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in result.stdout.split() if line]


def _reachable(address: str) -> bool | None:
    try:
        return subprocess.run(["ping", "-c1", "-W1", address], capture_output=True,
                              timeout=5).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return None


def observations(repo: Path, *, tool: dict | None = None, requested_priority: int | None = None,
                 probe_arms: bool = True, node: str | None = None) -> list[dict]:
    """Every readiness observation, in the order an operator reads them."""
    out = []

    binary = repo / "cpp/teleop/build/wxai_teleop"
    out.append(observation(
        "executable", "control executable", OK if os.access(binary, os.X_OK) else FAILED, str(binary),
        None if os.access(binary, os.X_OK) else
        "build it: cd cpp/teleop && cmake -B build -S . && cmake --build build"))

    profile = None
    try:
        sys.path.insert(0, str(repo / "scripts/lib"))
        import tatbot_profile
        profile = tatbot_profile.load(repo)
        errors = tatbot_profile.hardware_errors(profile)
        out.append(observation("profile", "hardware profile", FAILED if errors else OK,
                               profile["name"], "; ".join(errors) or None))
    except Exception as exc:  # noqa: BLE001 — any profile problem is one observation
        out.append(observation("profile", "hardware profile", FAILED, None, str(exc)))

    if tool:
        out.append(observation("tool", "fitted tool", FAILED if tool.get("error") else OK,
                               tool.get("id"), tool.get("error") or f"from {tool.get('source_label')}"))

    facts = realtime_facts(repo, requested_priority)
    account = explain_realtime(facts, node=node, from_check=True)
    # The question is whether this RIG can start teleop, not whether the
    # throwaway session this check runs in could. A routed check always runs in
    # a session an operator never starts teleop from.
    usable = facts["sufficient"] or facts["host_capable"]
    out.append(observation("realtime", "real-time scheduling", OK if usable else FAILED,
                           {"required": facts["priority"], "soft": facts["soft"], "hard": facts["hard"],
                            "host_capable": facts["host_capable"],
                            "sessions": facts["sessions"]},
                           account[0], account[1:]))

    driver = (profile or {}).get("driver") or {}
    device = driver.get("estop_device")
    if not device:
        out.append(observation("estop", "e-stop device", UNKNOWN, None,
                               "the profile names no e-stop device"))
    else:
        present = Path(device).exists()
        out.append(observation("estop", "e-stop device", OK if present else FAILED, device,
                               "present; a path is not a heartbeat — the launcher's monitor "
                               "decides that" if present else "not present"))

    running = _pgrep("[w]xai_teleop")
    out.append(observation("exclusivity", "arm driver free", FAILED if running else OK,
                           {"wxai_teleop_pids": running},
                           f"a teleop is already running (pid {running[0]}); it is yours to end"
                           if running else "no wxai_teleop seen; the driver is exclusive and the "
                                           "launcher re-checks at connect"))

    for role in ("leader", "follower"):
        address = driver.get(f"{role}_ip")
        if not address:
            out.append(observation(f"arm-{role}", f"{role} arm", UNKNOWN, None,
                                   "the profile names no address"))
        elif not probe_arms:
            out.append(observation(f"arm-{role}", f"{role} arm", UNKNOWN, address, "not probed"))
        else:
            up = _reachable(address)
            out.append(observation(f"arm-{role}", f"{role} arm",
                                   UNKNOWN if up is None else OK if up else FAILED, address,
                                   "no ping tool" if up is None else
                                   "answers ping; that is reachability, not driver readiness"
                                   if up else "does not answer ping — powered on? (~20 s to boot)"))
    return out


def report(repo: Path, **kwargs) -> dict:
    rows = observations(repo, **kwargs)
    return {"schema": "tatbot.teleop-readiness/1",
            "ready": all(row["state"] != FAILED for row in rows),
            "observations": rows,
            "limits": "configured prerequisites only; the launcher and the control "
                      "executable own what the scheduler and the arms actually do"}


def main(argv: list[str]) -> int:
    repo = Path(os.environ.get("TATBOT_REPO") or Path(__file__).resolve().parents[2])
    if "--explain-realtime" in argv:
        # The launcher's post-refusal explanation. Read-only; changes nothing.
        node = os.environ.get("TATBOT_NODE")
        for line in explain_realtime(realtime_facts(repo), node=node):
            print(line, file=sys.stderr)
        return 0
    print(json.dumps(report(repo), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
