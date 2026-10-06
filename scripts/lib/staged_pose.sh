#!/usr/bin/env bash
# The staged (= sleep) pose from config/trossen/tatbot.yaml, for wrappers that
# hand it to wxai_teleop as --staged-positions so a probe release lands
# both arms INSIDE the executor's live session (the fresh-session handover in
# il_recover_arm.sh sagged the follower at Enter on every 2026-09-02 draw).
#
#   source scripts/lib/staged_pose.sh
#   STAGED="$(staged_pose::csv "$REPO")" || exit 1
#   ... teleop_start.sh ... --staged-positions "$STAGED" ...

# Prints "a,b,c,d,e,f,g" (7 values) or fails. Stdlib only: a retained session
# source has no LeRobot environment, and the pose is one line of tatbot.yaml.
staged_pose::csv() {
  local repo="$1"
  PYTHONPATH="$repo/scripts/lib" python3 - "$repo/config/trossen/tatbot.yaml" <<'EOF'
import sys
from tool_spec import parse_simple_yaml
pose = parse_simple_yaml(open(sys.argv[1]).read())["follower"]["staged_positions"]
assert isinstance(pose, list) and len(pose) == 7, pose
print(",".join(repr(float(v)) for v in pose))
EOF
}
