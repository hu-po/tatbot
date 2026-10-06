"""Tests for the plugin's view of the tool registry.

No hardware and no lerobot imports: the shim is deliberately stdlib-only so it
can be exercised anywhere.

What matters here is the seam. ``scripts/lib/tool_spec.py`` is the one
implementation, reached through the ``tatbot-scriptlib`` dependency, so a venv
assembled without it breaks tool resolution at connect time on the robot — the
worst possible place to find out. These tests fail in CI instead.
"""

from pathlib import Path

from lerobot_robot_tatbot import tool_registry


def test_the_registry_is_reachable_from_inside_the_plugin():
    """The dependency resolves, and to the repo's own file rather than a
    vendored copy that could drift from what the scripts run."""
    registry = tool_registry.registry()
    assert registry is not None
    assert Path(registry.__file__) == tool_registry.REPO / "scripts" / "lib" / "tool_spec.py"


def test_the_stated_tool_has_a_mount():
    """What the arm asks the datasheet for at connect (2026-08-30): not a grip
    force any more, but that the tool has a mount on this arm and which
    URDF link that mount is."""
    calibrated = tool_registry.registry().active_tool_id(tool_registry.REPO)
    tool = tool_registry.stated_tool(calibrated)
    assert tool is not None, "config/workspace.yaml names no tool_id"
    assert tool.mounted
    assert tool.mount_frame("right") == "right/tool_mount"


def test_a_tool_without_a_mount_refuses(tmp_path):
    """mount: none means nobody has built an adapter for this tool — the
    laser pen today — and nothing may fit, calibrate or fly it."""
    reg = tool_registry.registry()
    (tmp_path / "config" / "tools").mkdir(parents=True)
    (tmp_path / "config" / "tools" / "blank.yaml").write_text(
        "schema_version: 2\ntool_id: blank\nkind: rotary_pen\n"
        'display_name: "Blank"\nprompt_phrase: "x"\nmount: none\n'
        "profile: [[-0.04, 0.010], [0.05, 0.001]]\n")
    blank = reg.load_tool("blank", tmp_path)
    assert not blank.mounted
    try:
        reg.require_stated_tool("blank", tmp_path, "right", {})
    except reg.ToolMountError as exc:
        assert "no mount" in str(exc)
    else:
        raise AssertionError("a tool with no mount must refuse, not default")
