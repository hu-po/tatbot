#!/usr/bin/env bash
# The staged (= sleep) pose from config/trossen/tatbot.yaml, for the wrappers
# that hand it to a native binary: wxai_teleop's --staged-positions (it lands
# both arms inside its live session) and arm_recover's --staged.
#
#   source scripts/lib/staged_pose.sh
#   STAGED="$(staged_pose::csv "$REPO")" || exit 1

# Prints "a,b,c,d,e,f,g" (7 values) or fails. Stdlib only, through the same
# tool_spec.staged_positions every other reader uses: one copy of the pose.
staged_pose::csv() {
  local repo="$1"
  PYTHONPATH="$repo/scripts/lib" python3 - "$repo" <<'PY'
import sys
from tool_spec import staged_positions
print(",".join(repr(v) for v in staged_positions(sys.argv[1])))
PY
}
