"""The stack configuration the orchestrator reads: the effective stack.yaml, the page of the print it
names, the fitted tool and its datasheet, and the run-log module. Pure Python (pyyaml); the share-dir
lookup is the only ROS touch.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import yaml
from tatbot_bridge import capture, stack


def stack_path(path: str | None = None) -> Path:
    return stack.stack_path(path, fallback=False)


def load_stack(path: str | None = None) -> dict:
    """The effective stack.yaml: the path stack.launch.py wrote into the ros-stack run (launch
    arguments applied), or the installed default."""
    return yaml.safe_load(stack_path(path).read_text())


def workspace_arm(repo: Path, arm: str) -> dict:
    """config/workspace.yaml's section for one arm (read, never written)."""
    return yaml.safe_load((Path(repo) / "config" / "workspace.yaml").read_text())[arm]


def fitted_tool(repo: Path, arm: str) -> str:
    return str(workspace_arm(repo, arm).get("tool_id") or "")


def tool_datasheet(repo: Path, tool_id: str):
    """config/tools/<tool_id>.yaml as scripts/lib/tool_spec loads and validates it (a ToolSpec)."""
    capture._lib(Path(repo))
    import tool_spec

    return tool_spec.load_tool(tool_id, repo)


def file_sha256(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()
    except OSError:
        return None


def revision(repo: Path) -> dict | None:
    """The deployed build: $TATBOT_REPO/REVISION (`tatbot ros deploy`: the sha, then `dirty=0|1`)."""
    try:
        lines = (Path(repo) / "REVISION").read_text().split()
    except OSError:
        return None
    return {"sha": lines[0] if lines else "", "dirty": "dirty=1" in lines}


def print_page(repo: Path, page: dict, root=None) -> dict:
    """stack.yaml's `page` with the geometry of the print it names: size_m, clear_m and inner_edges_m from the
    reference installed for page.pattern_id (scripts/lib/stencil_reference.installed_page), so a print of any
    size is drawn on as it was generated. Without one, stack.yaml's size_m and clear_m stand (the nominal
    page of a mock or a fixed bench page). `geometry` says which."""
    capture._lib(Path(repo))
    import stencil_reference

    found = stencil_reference.installed_page(str(page.get("pattern_id") or ""), root)
    if found is None:
        return {**page, "geometry": "stack.yaml"}
    out = {key: value for key, value in page.items() if key != "inner_edges_m"}
    out.update(size_m=found["size_m"], clear_m=found["clear_m"], geometry=f"print {found['pattern_id']}")
    if "inner_edges_m" in found:
        out["inner_edges_m"] = found["inner_edges_m"]
    return out


def page_trim(path: str | None) -> dict | None:
    """page.trim from the stack.yaml the launch was given (`config_path` in the effective one), read again
    at every goal so a trim stored there and deployed applies without a restart. None when unreadable."""
    if not path:
        return None
    try:
        return dict((yaml.safe_load(Path(path).expanduser().read_text()).get("page") or {}).get("trim") or {})
    except (OSError, AttributeError, yaml.YAMLError):
        return None


def runlog(repo: Path):
    """$TATBOT_REPO/scripts/lib/tatbot_runlog.py, imported by path (stdlib only by design)."""
    path = Path(repo) / "scripts" / "lib" / "tatbot_runlog.py"
    spec = importlib.util.spec_from_file_location("tatbot_runlog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
