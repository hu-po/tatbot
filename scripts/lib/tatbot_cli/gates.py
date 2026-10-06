"""Gates the CLI checks itself, before exec, so the exit code is distinguishable.

The launchers still run their own copies; this layer exists so an agent gets
exit 3 with a JSON reason instead of a bash usage message, and so `--dry-run`
can say which gate would have refused. Nothing here weakens a launcher gate —
the CLI has no flag that reaches a launcher's e-stop or arm-gate code path.
"""

from __future__ import annotations

import os
import re
import secrets
import socket
import time
from pathlib import Path

# Mirror of the case patterns in scripts/lib/estop_guard.sh; test_cli.py
# parses that file and asserts the two lists agree. (pattern, is_prefix)
ESTOP_OVERRIDES = (
    ("--no-estop", False),
    ("--estop", False),          # `--estop` and `--estop=*`
    ("--robot.estop_device", True),
    ("--robot.estop_required", True),
    ("--teleop.estop_device", True),
    ("--teleop.estop_required", True),
)

ARM_TOKEN = Path("/tmp/tatbot-arm-token")
TAG_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LAUNCH_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[A-Za-z0-9_-]{1,16}-[0-9a-f]{4}(?:-[A-Za-z0-9_-]{1,64})?$")


def estop_overrides(args: list[str]) -> list[str]:
    bad = []
    for a in args:
        for pat, prefix in ESTOP_OVERRIDES:
            if a == pat or a.startswith(pat + "=") or (prefix and a.startswith(pat)):
                bad.append(a)
                break
    return bad


def known_tools(repo: Path) -> list[str]:
    d = repo / "config" / "tools"
    return sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []


# Where a resolved tool identity came from, most explicit first. Rendered in
# startup notices, plans and diagnostics so a default is never mistaken for a
# statement about what is physically in the mount.
TOOL_SOURCES = {"flag": "--ee-tool", "environment": "TATBOT_EE_TOOL",
                "configured": "the configured fitted tool"}
FLAG_HINT = "state it with --ee-tool <id>; `tatbot tool list` names them"


def configured_tool(repo: Path, arm: str = "right") -> tuple[str | None, str | None]:
    """(tool_id, problem) — the fitted-tool pointer, read through tool_spec.

    One pointer, written by a touch-off. This never copies it, and reading it
    is not evidence that the physical tool was inspected.
    """
    try:
        import tool_spec
    except ImportError as exc:  # a checkout without the datasheet library
        return None, f"cannot read the fitted-tool pointer ({exc})"
    try:
        return tool_spec.active_tool_id(repo, arm), None
    except (OSError, ValueError) as exc:
        return None, f"the fitted-tool pointer is unreadable ({exc})"


def resolve_tool(repo: Path, stated: str | None, *, arm: str = "right") -> tuple[str | None, str, str | None]:
    """(tool_id, source, error) for the tool in the mount.

    Precedence, most explicit first: the ``--ee-tool`` flag, then an
    invocation's ``TATBOT_EE_TOOL``, then the execution owner's configured
    fitted tool. An explicit value that is wrong is a refusal, never a silent
    fall back to the default — the whole point of stating a tool is that a swap
    becomes visible. Resolving a default changes nothing on disk and does not
    establish that the tool was inspected; the datasheet-versus-calibration
    check in tool_spec.require_stated_tool still runs at connect.
    """
    known = known_tools(repo)
    listing = ", ".join(known) or "none"
    for value, source in ((stated, "flag"), (os.environ.get("TATBOT_EE_TOOL"), "environment")):
        if value:
            if value not in known:
                return None, source, f"unknown tool '{value}' from {TOOL_SOURCES[source]} (known: {listing})"
            return value, source, None
    configured, problem = configured_tool(repo, arm)
    if problem:
        return None, "configured", f"{problem}; {FLAG_HINT}"
    if not configured:
        return None, "configured", ("no tool stated and no fitted tool is configured; "
                                    f"{FLAG_HINT} (known: {listing})")
    if configured not in known:
        return None, "configured", (f"the configured fitted tool '{configured}' has no datasheet in "
                                    f"config/tools/ (known: {listing}); {FLAG_HINT}")
    return configured, "configured", None


def tag_error(tag: str | None) -> str | None:
    """--tag is optional; a given value must survive the launcher's `tr -cd 'A-Za-z0-9_-'` intact."""
    if tag is None:
        return None
    if not TAG_RE.match(tag):
        return "--tag must be 1-64 chars of [A-Za-z0-9_-]"
    return None


def short_hostname() -> str:
    host = socket.gethostname().split(".")[0].lower()
    return re.sub(r"[^A-Za-z0-9_-]", "", host)[:16] or "node"


def mint_launch_id(tag: str | None = None, *, now: float | None = None) -> str:
    """`<UTC %Y%m%dT%H%M%SZ>-<short hostname>-<4 hex>[-<tag>]`: the id every autonomous
    launch carries. arm_gate.sh ledgers and audits it (pid chain, SSH origin); a repeat is
    audited, never refused. Minted on the node that execs, right before the exec."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    parts = [stamp, short_hostname(), secrets.token_hex(2)]
    if tag:
        parts.append(tag)
    return "-".join(parts)


def write_launch_id(launch_id: str) -> None:
    """Exactly what `echo <id> > /tmp/tatbot-arm-token` does, immediately before exec."""
    ARM_TOKEN.write_text(launch_id + "\n")
    now = time.time()
    os.utime(ARM_TOKEN, (now, now))


def train_root() -> Path:
    return Path(os.environ.get("TATBOT_TRAIN_ROOT", "~/il-train")).expanduser()


def busy_reasons() -> list[str]:
    """Point-in-time hints; launchers own their actual lock and refusal codes."""
    from tatbot_cli.locks import held
    reasons = []
    root = train_root()
    if (root / "SWEEP_PAUSE").exists():
        reasons.append(f"SWEEP_PAUSE present at {root / 'SWEEP_PAUSE'}")
    try:
        if held(root / ".tatbot-training.lock"):
            reasons.append(f"training lock held at {root / '.tatbot-training.lock'}")
    except (OSError, ValueError) as exc:
        reasons.append(f"training lock ownership unknown: {exc}")
    return reasons
