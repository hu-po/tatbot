#!/usr/bin/env bash
# The ink hook for launchers that put a tool on the skin.
#
#   source "$REPO/scripts/lib/ink_hook.sh"
#   ink_hook::strip "$@"; set -- "${INK_HOOK_ARGS[@]}"   # before positionals
#   ...
#   ink_hook::stamp                                     # once $RUN_DIR exists
#
# Two ways to run:
#
#   (nothing)  ink is left as it is: the open session (if any) is debited by
#              the post-run analysis, and the dataset stamp says which session
#              the run belonged to.
#   --no-ink   do not deal with ink at all (operator, 2026-08-29): no session,
#              no stroke debit; the run is stamped ink.tracking=false.
#              Exported as TATBOT_INK=0 for il_tool_meta.py / il_analyze_rollout.py.
#
# There is no scripted dip before a launch: palette dips belong to the ROS 2
# stack's ink program (ros/tatbot_ink).
#
# The decision is also written into the run directory as $RUN_DIR/ink.json
# ({tracking, hook, utc}), because an environment variable lives one shell:
# `tatbot rollout analyze <run>` re-run tomorrow from a fresh shell has no
# TATBOT_INK and would debit a --no-ink run, and a stale TATBOT_INK=0 in an
# interactive shell would silently skip a real one. The analysis reads the
# stamp first and falls back to the variable only when there is none.
ink_hook::strip() {
  INK_HOOK_NOINK=0
  INK_HOOK_ARGS=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --no-ink) INK_HOOK_NOINK=1; shift ;;
      *) INK_HOOK_ARGS+=("$1"); shift ;;
    esac
  done
  # The flag decides, never the caller's shell: il_tool_meta.py stamps the
  # dataset from this variable, and it must agree with ink.json.
  if [ "$INK_HOOK_NOINK" = 1 ]; then
    export TATBOT_INK=0
  else
    export TATBOT_INK=1
  fi
}

ink_hook::stamp() {
  # $RUN_DIR/ink.json — the run's own record of the decision (see header).
  [ -n "${RUN_DIR:-}" ] && [ -d "$RUN_DIR" ] || return 0
  local tracking=true hook=none
  if [ "${INK_HOOK_NOINK:-0}" = 1 ]; then
    tracking=false; hook="--no-ink"
    echo "ink: --no-ink — no session, no stroke debit for this run" >&2
  fi
  printf '{"tracking": %s, "hook": "%s", "utc": "%s"}\n' "$tracking" "$hook" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/ink.json"
}
