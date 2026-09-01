# shellcheck shell=bash
# Shared root resolution for shell scripts — the shell face of
# scripts/lib/tatbot_paths.py. Source this; do not execute it.
#
#   source "$REPO/scripts/lib/paths.sh"
#   LOG_ROOT="$(tatbot_paths::log_root)"
#
# Resolution: TATBOT_LOG_ROOT > config/runlog.json log_root (via
# tatbot_runlog.py, which layers ~/.config/tatbot/runlog.json) > XDG state.
# No script may default to a literal $HOME/tatbot-logs (plan Phase 1).

tatbot_paths::repo_root() {
  if [[ -n "${TATBOT_REPO:-}" ]]; then echo "$TATBOT_REPO"; return; fi
  ( cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd )
}

tatbot_paths::log_root() {
  if [[ -n "${TATBOT_LOG_ROOT:-}" ]]; then echo "$TATBOT_LOG_ROOT"; return; fi
  local repo
  repo="$(tatbot_paths::repo_root)"
  python3 "$repo/scripts/lib/tatbot_runlog.py" root 2>/dev/null \
    || echo "${XDG_STATE_HOME:-$HOME/.local/state}/tatbot/logs"
}
