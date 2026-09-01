# shellcheck shell=bash
# Hardware-profile environment for launchers. Source after REPO is set:
#
#   source "$REPO/scripts/lib/profile_env.sh"
#   profile_env::require   # exports TATBOT_{LEADER_IP,FOLLOWER_IP,ESTOP_DEVICE}
#
# Launchers reached through the tatbot CLI already inherit these from the
# CLI's profile gate; this covers direct invocation. No profile, or a
# gate-incapable one, is a hard stop BEFORE anything touches an arm
# (plan Phase 2).

profile_env::require() {
  # ALL THREE or resolve: pre-set arm IPs alone must not skip resolution and
  # leave the safety-critical e-stop device unset (audit 2026-08-31,
  # finding: rollout launchers had no backstop for a missing device).
  if [[ -n "${TATBOT_FOLLOWER_IP:-}" && -n "${TATBOT_LEADER_IP:-}" \
        && -n "${TATBOT_ESTOP_DEVICE:-}" ]]; then
    return 0
  fi
  local repo exports
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  if exports="$(python3 "$repo/scripts/lib/tatbot_profile.py" --export)"; then
    eval "$exports"
  else
    echo "profile_env: no hardware-capable profile resolved — pass" \
         "TATBOT_PROFILE or run through the tatbot CLI. ($exports)" >&2
    return 3
  fi
  if [[ -z "${TATBOT_ESTOP_DEVICE:-}" ]]; then
    echo "profile_env: profile resolved but supplies no estop_device — refusing" >&2
    return 3
  fi
}
