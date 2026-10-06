#!/usr/bin/env python3
"""Install the checkout's Tatbot launcher at the system command path."""
from __future__ import annotations

import argparse
import os
import subprocess
import uuid
from pathlib import Path

DESTINATION = Path("/usr/local/bin/tatbot")


def run(*argv: object) -> None:
    print("+", *(str(arg) for arg in argv), flush=True)
    subprocess.run([str(arg) for arg in argv], check=True)


def _link_target(link: Path) -> tuple[Path, Path]:
    raw = Path(os.readlink(link))
    absolute = raw if raw.is_absolute() else link.parent / raw
    return raw, absolute.resolve(strict=False)


def validate_destination(destination: Path, launcher: Path) -> None:
    """Preserve anything that is not recognizably an earlier Tatbot link."""
    if not os.path.lexists(destination):
        return
    if not destination.is_symlink():
        raise ValueError(f"refusing to replace non-symlink {destination}")
    raw, resolved = _link_target(destination)
    if resolved == launcher or (raw.name == "tatbot" and raw.parent.name == "scripts"):
        return
    raise ValueError(f"refusing to replace unrelated symlink {destination} -> {raw}")


def install_cli(repo: Path, destination: Path = DESTINATION, *, privileged: bool = True) -> None:
    repo = repo.resolve()
    launcher = repo / "scripts/tatbot"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ValueError(f"Tatbot launcher is missing or not executable: {launcher}")
    validate_destination(destination, launcher)
    if not destination.parent.is_dir():
        raise ValueError(f"command directory does not exist: {destination.parent}")

    prefix = ("sudo", "-n") if privileged else ()
    temporary = destination.with_name(f".{destination.name}.deploy-{uuid.uuid4().hex}")
    run(*prefix, "ln", "-s", str(launcher), str(temporary))
    try:
        run(*prefix, "mv", "-Tf", str(temporary), str(destination))
    except BaseException:
        subprocess.run([*prefix, "rm", "-f", str(temporary)], check=False)
        raise

    if _link_target(destination)[1] != launcher:
        raise RuntimeError(f"installed command does not resolve to {launcher}")
    run(str(destination), "--version")
    print(f"Tatbot CLI installed: {destination} -> {launcher}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    try:
        install_cli(args.repo)
    except ValueError as error:
        parser.exit(3, f"CLI install refused: {error}\n")


if __name__ == "__main__":
    main()
