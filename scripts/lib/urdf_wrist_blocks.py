"""Generated wrist-fiducial blocks in the shared URDF, as text: stdlib only.

One block per fiducial target sits in urdf/tatbot.urdf between marker
comments. export_wrist_tags.py renders a block from a layout; the arm-scoped
adopter splices one arm's block into whatever URDF the destination holds so
the other arm's block survives byte for byte. Both use these helpers.
"""
from __future__ import annotations

DEFAULT_TARGET = "wrist"
BEGIN_PREFIX = "  <!-- BEGIN GENERATED WRIST FIDUCIALS"
END_MARKER = "  <!-- END GENERATED WRIST FIDUCIALS -->"
LEGACY_BEGIN = "  <!-- Provisional geometry for the new three-fiducial wrist mount"
LEGACY_END = '  <joint name="right/realsense_depth_joint"'
# A block for a second target is inserted before this line the first time it
# is generated; afterwards its own markers locate it.
LEFT_INSERT_BEFORE = "  <!-- arm_r -->"


def block_begin(target: str) -> str:
    """The right arm's block predates targets and keeps its unqualified marker."""
    return BEGIN_PREFIX if target == DEFAULT_TARGET else f"{BEGIN_PREFIX} target={target}"


def find_urdf_block(text: str, target: str) -> tuple[int, int] | None:
    """Span of this target's generated block, or None when it was never generated."""
    marker = block_begin(target) + " layout_sha256="
    search = 0
    while (begin := text.find(BEGIN_PREFIX, search)) >= 0:
        head = text[begin:text.find("\n", begin)]
        # The untargeted right-arm marker must not match a `target=` block.
        if head.startswith(marker):
            end = text.find(END_MARKER, begin)
            if end < 0:
                raise ValueError("URDF generated wrist block has no end marker")
            end += len(END_MARKER)
            if end < len(text) and text[end] == "\n":
                end += 1
            return begin, end
        search = begin + len(BEGIN_PREFIX)
    return None


def replace_urdf_block(text: str, block: str, target: str = DEFAULT_TARGET) -> str:
    span = find_urdf_block(text, target)
    if span is not None:
        begin, end = span
        return text[:begin] + block + text[end:]
    if target == DEFAULT_TARGET:
        begin = text.find(LEGACY_BEGIN)
        end = text.find(LEGACY_END, begin)
        if begin < 0 or end < 0:
            raise ValueError("URDF has neither generated nor recognized legacy wrist block")
        return text[:begin] + block + text[end:]
    begin = text.find(LEFT_INSERT_BEFORE)
    if begin < 0:
        raise ValueError(f"URDF has no insertion point for the {target} block")
    return text[:begin] + block + text[begin:]
