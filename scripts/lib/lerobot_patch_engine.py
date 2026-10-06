"""Plan and validate a complete LeRobot patch set before changing installed files."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def write_breaking_hardlink(path: Path, content: str) -> None:
    """Replace the inode without modifying uv's shared dependency cache."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def plan_patches(patches):
    originals: dict[Path, str] = {}
    planned: dict[Path, str] = {}
    targets: dict[str, Path] = {}
    skipped: dict[str, str] = {}
    unresolved = []
    for module_name, candidates, replacement in patches:
        if module_name in skipped:
            continue
        if module_name not in targets:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as error:
                # Only an absent target is optional. A broken dependency inside
                # an installed target must fail before any source is changed.
                if error.name and (module_name == error.name or module_name.startswith(error.name + ".")):
                    skipped[module_name] = str(error)
                    continue
                raise
            if module.__file__ is None:
                raise ValueError(f"no file source: {module_name}")
            path = Path(module.__file__)
            targets[module_name] = path
            if path not in originals:
                originals[path] = path.read_text()
                planned[path] = originals[path]
        path = targets[module_name]
        source = planned[path]
        if replacement in source:
            continue
        for candidate in candidates:
            if candidate not in source:
                continue
            if source.count(candidate) != 1:
                raise ValueError(f"ambiguous patch site in {module_name}")
            source = source.replace(candidate, replacement, 1)
            for other in candidates:
                if other not in (replacement, candidate) and other in source.replace(replacement, ""):
                    raise ValueError(f"stale patch text remains in {module_name}; reinstall LeRobot")
            planned[path] = source
            break
        else:
            unresolved.append((module_name, path, replacement))
    for module_name, path, replacement in unresolved:
        if replacement not in planned[path]:
            raise ValueError(f"patch site not found in {module_name}; update the patch catalog")
    for path, source in planned.items():
        compile(source, str(path), "exec")
    return originals, planned, targets, skipped


def apply_patches(patches, *, receipt_path: Path | None = None) -> int:
    try:
        originals, planned, targets, skipped = plan_patches(patches)
        for path, source in originals.items():
            if path.read_text() != source:
                raise ValueError(f"source changed during patch planning: {path}")
        changed = []
        try:
            for path, source in planned.items():
                if source != originals[path]:
                    write_breaking_hardlink(path, source)
                    changed.append(path)
        except OSError:
            for path in reversed(changed):
                write_breaking_hardlink(path, originals[path])
            raise
        if receipt_path is not None:
            try:
                upstream_version = version("lerobot")
            except PackageNotFoundError:
                upstream_version = None
            receipt = {
                "schema": "tatbot.lerobot-patches/1",
                "catalog_sha256": digest(json.dumps(patches, separators=(",", ":"))),
                "lerobot_version": upstream_version,
                "modules": {name: digest(planned[path]) for name, path in sorted(targets.items())},
                "skipped_modules": sorted(skipped),
            }
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            write_breaking_hardlink(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        for name in sorted(targets):
            print(f"verified patched source: {name}")
        for name, reason in sorted(skipped.items()):
            print(f"skipped absent target: {name} ({reason})")
        return 0
    except (ImportError, OSError, ValueError, SyntaxError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
