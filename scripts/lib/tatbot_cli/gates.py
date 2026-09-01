"""Gates the CLI checks itself, before exec, so the exit code is distinguishable.

The launchers still run their own copies; this layer exists so an agent gets
exit 3 with a JSON reason instead of a bash usage message, and so `--dry-run`
can say which gate would have refused. Nothing here weakens a launcher gate —
the CLI has no flag that reaches a launcher's e-stop or arm-gate code path.
"""

from __future__ import annotations

import os
import re
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
NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


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


def resolve_tool(repo: Path, stated: str | None) -> tuple[str | None, str | None]:
    """(tool_id, error). Stated on the command line or in TATBOT_EE_TOOL — never inferred."""
    tool = stated or os.environ.get("TATBOT_EE_TOOL")
    known = ", ".join(known_tools(repo)) or "none"
    if not tool:
        return None, f"--ee-tool <id> is required: name the tool in the mount (known: {known})"
    if tool not in known_tools(repo):
        return None, f"unknown tool '{tool}' (known: {known})"
    return tool, None


def nonce_error(nonce: str | None) -> str | None:
    if not nonce:
        return (
            "--nonce <literal> is required for autonomous motion: a unique literal you type "
            "now (never $RANDOM or $(date), which a replayed shell re-evaluates); each nonce is "
            "single-use and ledgered by scripts/lib/arm_gate.sh"
        )
    if not NONCE_RE.match(nonce):
        return "nonce must be 1-128 chars of [A-Za-z0-9_-]"
    return None


def write_nonce(nonce: str) -> None:
    """Exactly what `echo <nonce> > /tmp/tatbot-arm-token` does, immediately before exec."""
    ARM_TOKEN.write_text(nonce + "\n")
    now = time.time()
    os.utime(ARM_TOKEN, (now, now))


def train_root() -> Path:
    return Path(os.environ.get("TATBOT_TRAIN_ROOT", "~/il-train")).expanduser()


def busy_reasons() -> list[str]:
    """What would make a hardware or training verb refuse with exit 6."""
    reasons = []
    root = train_root()
    if (root / "SWEEP_PAUSE").exists():
        reasons.append(f"SWEEP_PAUSE present at {root / 'SWEEP_PAUSE'}")
    if (root / ".tatbot-training.lock").exists():
        reasons.append(f"training lock held at {root / '.tatbot-training.lock'}")
    return reasons
