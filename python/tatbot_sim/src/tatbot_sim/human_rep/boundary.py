"""Static boundaries for motion authority and frozen research dependencies."""

from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEARNED_PATH_MARKERS = {
    "generator",
    "learned",
    "optimizer",
    "policy",
    "research",
    "train",
    "training",
}
FORBIDDEN_IMPORT_PREFIXES = (
    "pen_path",
    "lerobot_robot_tatbot",
    "lerobot_robot_trossen",
    "lerobot_teleoperator_trossen",
    "tatbot_cli.launch",
    "tatbot_cli.verbs.hardware",
    "tatbot_cli.verbs.live",
    "tatbot_cli.verbs.rollout",
    "tatbot_cli.verbs.session",
    "tatbot_cli.verbs.vision",
    "tatbot_executor",
    "trossen_arm",
)
FORBIDDEN_IMPORT_FRAGMENTS = (
    "arm_gate",
    "pen_path",
    "estop_guard",
    "execute_draw",
    "motion_launcher",
    "write_samples_csv",
)


@dataclass(frozen=True)
class BoundaryViolation:
    path: Path
    line: int
    imported: str
    input_sha256: str
    detail: str

    code = "learned_execution_boundary_violation"

    @property
    def input_hashes(self) -> dict[str, str]:
        return {"source_file_sha256": self.input_sha256} if self.input_sha256 else {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "refused",
            "code": self.code,
            "path": self.path.as_posix(),
            "line": self.line,
            "imported": self.imported,
            "detail": self.detail,
            "input_hashes": self.input_hashes,
        }

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.code}: {self.detail} (sha256={self.input_sha256 or 'unavailable'})"


def _is_learned_path(path: Path) -> bool:
    for part in path.parts:
        normalized = part.lower().replace("-", "_")
        stem = normalized.removesuffix(".py")
        if stem in LEARNED_PATH_MARKERS or LEARNED_PATH_MARKERS.intersection(stem.split("_")):
            return True
    return False


def _imports(tree: ast.AST) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            result.extend((node.lineno, f"{module}.{alias.name}".strip(".")) for alias in node.names)
    return result


def _forbidden(name: str) -> bool:
    normalized = name.removeprefix("scripts.lib.")
    return normalized.startswith(FORBIDDEN_IMPORT_PREFIXES) or any(
        fragment in name for fragment in FORBIDDEN_IMPORT_FRAGMENTS
    )



def _runtime_sources(base: Path):
    # Prune dependency/build trees before walking them, rather than traversing
    # every installed file only to discard it at the leaf.
    excluded = {".git", ".venv", "build", "dist", "node_modules", "tests", "__pycache__"}
    for directory, children, files in os.walk(base):
        children[:] = sorted(name for name in children if name not in excluded)
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(directory) / name


def scan_forbidden_imports(root: str | Path) -> list[BoundaryViolation]:
    """Return violations under learned/research/policy path namespaces."""

    base = Path(root)
    violations: list[BoundaryViolation] = []
    for path in _runtime_sources(base):
        relative = path.relative_to(base)
        if not _is_learned_path(relative):
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            violations.append(
                BoundaryViolation(
                    path=relative,
                    line=0,
                    imported="<read-error>",
                    input_sha256="",
                    detail=f"cannot inspect learned source: {exc}",
                )
            )
            continue
        digest = hashlib.sha256(raw).hexdigest()
        try:
            source = raw.decode("utf-8")
            tree = ast.parse(source, filename=str(relative))
        except (UnicodeDecodeError, SyntaxError) as exc:
            violations.append(
                BoundaryViolation(
                    path=relative,
                    line=getattr(exc, "lineno", None) or 0,
                    imported="<parse-error>",
                    input_sha256=digest,
                    detail=f"cannot prove learned/exact boundary: {exc}",
                )
            )
            continue
        for line, imported in _imports(tree):
            if _forbidden(imported):
                violations.append(
                    BoundaryViolation(
                        path=relative,
                        line=line,
                        imported=imported,
                        input_sha256=digest,
                        detail=f"forbidden learned-to-exact import {imported}",
                    )
                )
    return violations


# These experiments retain tests for reproducibility, but have no runtime
# consumers. Reopening one requires a named consumer and measured experiment.
FROZEN_MODULES = frozenset({
    "tatbot_sim.human_rep.training",
    "tatbot_sim.human_rep.torch_patch",
    "tatbot_sim.human_rep.mechanics",
    "tatbot_sim.human_rep.coupled_mechanics",
    "tatbot_sim.human_rep.anatomy",
})


def scan_frozen_imports(root: str | Path) -> list[str]:
    """Reject new runtime consumers; permit frozen peers and historical tests.

    Checks direct, relative, and literal dynamic Python imports. This is an
    architectural check, not a sandbox for arbitrary Python execution.
    """
    base = Path(root)
    violations = []
    for path in _runtime_sources(base):
        relative = path.relative_to(base)
        if relative.parts[:2] == ("internal", "evidence"):
            continue
        parts = list(relative.with_suffix("").parts)
        package_start = max(i for i, part in enumerate(parts) if part == "tatbot_sim") if "tatbot_sim" in parts else None
        module = ".".join(parts[package_start:]) if package_start is not None else ""
        if module in FROZEN_MODULES:
            continue
        try:
            tree = ast.parse(path.read_bytes(), filename=str(relative))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            violations.append(f"{relative}: cannot inspect runtime imports: {exc}")
            continue
        imports = _imports(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and module:
                package = module.split(".")[:-node.level]
                prefix = ".".join([*package, *([node.module] if node.module else [])])
                imports.extend((node.lineno, f"{prefix}.{alias.name}") for alias in node.names)
            if (isinstance(node, ast.Call) and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and ((isinstance(node.func, ast.Name) and node.func.id == "__import__")
                         or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"))):
                imports.append((node.lineno, node.args[0].value))
        for line, imported in imports:
            if any(imported == frozen or imported.startswith(frozen + ".") for frozen in FROZEN_MODULES):
                violations.append(f"{relative}:{line}: frozen research dependency {imported}")
    return violations
