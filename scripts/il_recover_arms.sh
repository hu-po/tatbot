#!/usr/bin/env bash
# Recover both arms after a controller fault, in the same follower-then-leader
# order used by normal session landing. Each single-arm launcher takes position
# control where that arm stands, then moves slowly staged -> sleep -> idle.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot arm recover"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require || exit $?

if (($#)); then
  echo "usage: il_recover_arms.sh" >&2
  exit 2
fi

RC=0
"$REPO/scripts/il_recover_arm.sh" "$TATBOT_FOLLOWER_IP" follower || RC=$?
if ((RC)); then
  echo "Follower recovery failed or was interrupted; leader recovery was NOT started." >&2
  exit "$RC"
fi
"$REPO/scripts/il_recover_arm.sh" "$TATBOT_LEADER_IP" leader
