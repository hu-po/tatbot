"""The readers import with numpy and scipy only, and never reach the drawing stack."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import tatbot_surface

PACKAGE = Path(tatbot_surface.__file__).resolve().parent
MODULES = sorted(path.stem for path in PACKAGE.glob("*.py") if path.stem != "__init__")
FORBIDDEN = ("torch", "shapely", "mujoco", "warp", "rclpy")


def test_every_module_imports_without_torch_shapely_or_a_physics_engine():
    probe = f"""
import json, sys
before = list(sys.path)
for name in {MODULES!r}:
    __import__("tatbot_surface." + name)
print(json.dumps({{
    "loaded": sorted(m for m in {FORBIDDEN!r} if m in sys.modules),
    "added": [p for p in sys.path if p not in before],
    "ros": [p for p in sys.path if "/ros/" in p.replace("\\\\", "/")],
}}))
"""
    env = dict(os.environ)
    # A caller's own ROS paths are not the package's doing; start from none.
    env["PYTHONPATH"] = os.pathsep.join(
        entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry and "/ros/" not in entry)
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True,
                            check=True, timeout=120)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"loaded": [], "added": [], "ros": []}


def test_no_module_names_a_forbidden_package_even_lazily():
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            else:
                continue
            roots = {name.split(".")[0] for name in names}
            assert not roots & set(FORBIDDEN), f"{path.name}:{node.lineno} imports {sorted(roots)}"
