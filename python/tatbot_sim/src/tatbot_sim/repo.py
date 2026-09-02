"""Repo-root resolution for the simulator — TATBOT_REPO, else this checkout.

The simulator reads the URDF, tool registry, and inkmap lexicon from the
repo. TATBOT_REPO points elsewhere (tests, temporary clones); the fallback
is the editable-install location of this file (plan Phase 1).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


def repo_root() -> Path:
    env = os.environ.get("TATBOT_REPO")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def _repository_slug(remote_url: str) -> str:
    """Return an owner/repository label without credentials or local paths."""
    value = remote_url.strip()
    if not value:
        return "local-checkout"

    if "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme not in {"git", "http", "https", "ssh"} or not parsed.hostname:
            return "local-checkout"
        path = parsed.path
    elif ":" in value and not value.startswith(("/", "./", "../")):
        host, path = value.split(":", 1)
        if not host or "/" not in path:
            return "local-checkout"
    else:
        return "local-checkout"

    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) < 2 or any(part in {".", ".."} for part in parts[-2:]):
        return "local-checkout"
    owner, repository = parts[-2], parts[-1]
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository:
        return "local-checkout"
    return f"{owner}/{repository}"


def source_state(root: Path | None = None) -> dict[str, str | bool | None]:
    """Return the checkout revision and whether generated code is dirty.

    A dataset path may carry a convenient revision label, but the artifact
    itself has to remain attributable after it is moved. Failure to identify
    a Git checkout is recorded as unknown and left for the audit to reject
    when the current run-meta schema requires source provenance.
    """
    checkout = root or repo_root()

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(checkout), *args],
            capture_output=True, text=True, check=False)

    revision = git("rev-parse", "--verify", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    remote = git("remote", "get-url", "origin")
    return {
        "repository": _repository_slug(remote.stdout) if remote.returncode == 0 else "local-checkout",
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout) if status.returncode == 0 else None,
    }
