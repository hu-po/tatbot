"""The staged/idle pose has ONE source: config/trossen/tatbot.yaml.

Until 2026-08-30 it lived in three places — the golden, scripts/il_recover_arm.sh
and tatbot_sim/planning.py — and nothing checked they agreed. Rolling the wrist
90 deg for the fixed EE mount would have moved one of them. Now the other two
READ the golden; this test fails if a literal copy comes back.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "config" / "trossen" / "tatbot.yaml"


def staged_from_golden():
    text = GOLDEN.read_text()
    m = re.search(r"^\s*staged_positions:\s*\[([^\]]+)\]", text, re.MULTILINE)
    assert m, "no staged_positions in tatbot.yaml"
    return [float(v) for v in m.group(1).split(",")]


def test_the_golden_staged_pose_is_seven_values_with_the_wrist_rolled():
    pose = staged_from_golden()
    assert len(pose) == 7
    # joint_5 ~ +pi/2: cube up, pen down (
    # phase 3). Captured on the arm 2026-08-30 (1.5566); the sign is the
    # FK-verified one, and a re-capture may move it a little.
    assert abs(pose[5] - 1.5707963267948966) < 0.15
    assert pose[6] == 0.0, "the 7th value is the carriage at rest"


def test_no_other_file_carries_a_literal_copy():
    """Any distinctive joint value from the golden appearing verbatim elsewhere
    is a copy waiting to drift."""
    pose = staged_from_golden()
    needles = {f"{v:.16g}" for v in pose if abs(v) > 0.1 and v != 1.5707963267948966}
    if not needles:
        # sleep pose + a rolled wrist (2026-08-30): nothing distinctive enough
        # to grep for; the readers are exercised by the other test instead
        return
    offenders = []
    for path in list(REPO.glob("scripts/**/*.py")) + list(REPO.glob("scripts/**/*.sh")) \
            + list(REPO.glob("python/**/src/**/*.py")):
        if "tests" in path.parts or path.name == "test_staged_pose_single_source.py":
            continue
        text = path.read_text(errors="ignore")
        for needle in needles:
            if needle in text:
                offenders.append(f"{path.relative_to(REPO)}: {needle}")
    assert not offenders, "staged pose copied outside tatbot.yaml:\n  " + "\n  ".join(offenders)
