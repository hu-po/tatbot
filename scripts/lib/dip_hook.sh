#!/usr/bin/env bash
# The ink hook for launchers that put a tool on the skin.
#
#   source "$REPO/scripts/lib/dip_hook.sh"
#   dip_hook::strip "$@"; set -- "${DIP_HOOK_ARGS[@]}"   # before positionals
#   ...
#   dip_hook::run                                       # just before the arm session
#
# Three ways to run:
#
#   (nothing)  ink is left as it is: no dip, the open session (if any) is
#              debited by the post-run analysis, and the dataset stamp says
#              which session the run belonged to.
#   --dip      top up at the palette before the session starts, through
#              scripts/il_dip.sh --if-needed: opens an ink session if none is
#              open, dips only when the session's charge will not cover the
#              run (docs/ink.md). Any il_dip.py option can be forwarded as
#              --dip-arg=<option> (repeatable), e.g. --dip-arg=--program=<json>.
#   --no-ink   do not deal with ink at all (operator, 2026-08-29): no dip, no
#              session, no stroke debit; the run is stamped ink.tracking=false.
#              Exported as TATBOT_INK=0 for il_tool_meta.py / il_analyze_rollout.py.
#
# --dip and --no-ink together is a question, not a precedence rule: refused.
#
# The decision is also written into the run directory as $RUN_DIR/ink.json
# ({tracking, hook, utc}), because an environment variable lives one shell:
# `tatbot rollout analyze <run>` re-run tomorrow from a fresh shell has no
# TATBOT_INK and would debit a --no-ink run, and a stale TATBOT_INK=0 in an
# interactive shell would silently skip a real one. The analysis reads the
# stamp first and falls back to the variable only when there is none.
dip_hook::strip() {
  DIP_HOOK=0
  DIP_HOOK_NOINK=0
  DIP_HOOK_ARGS=()
  DIP_HOOK_EXTRA=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --dip) DIP_HOOK=1; shift ;;
      --no-ink) DIP_HOOK_NOINK=1; shift ;;
      --dip-arg=*) DIP_HOOK_EXTRA+=("${1#--dip-arg=}"); shift ;;
      *) DIP_HOOK_ARGS+=("$1"); shift ;;
    esac
  done
  if [ "$DIP_HOOK" = 1 ] && [ "$DIP_HOOK_NOINK" = 1 ]; then
    echo "--dip and --no-ink cannot both be given" >&2
    return 2
  fi
  if [ "$DIP_HOOK_NOINK" = 1 ]; then
    export TATBOT_INK=0
  else
    export TATBOT_INK="${TATBOT_INK:-1}"
  fi
}

dip_hook::stamp() {
  # $RUN_DIR/ink.json — the run's own record of the decision (see header).
  [ -n "${RUN_DIR:-}" ] && [ -d "$RUN_DIR" ] || return 0
  local tracking=true hook=none
  if [ "${DIP_HOOK_NOINK:-0}" = 1 ]; then tracking=false; hook="--no-ink"
  elif [ "${DIP_HOOK:-0}" = 1 ]; then hook="--dip"; fi
  printf '{"tracking": %s, "hook": "%s", "utc": "%s"}\n' "$tracking" "$hook" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/ink.json"
}

dip_hook::run() {
  dip_hook::stamp
  if [ "${DIP_HOOK_NOINK:-0}" = 1 ]; then
    echo "ink: --no-ink — no dip, no session, no stroke debit for this run" >&2
    return 0
  fi
  [ "${DIP_HOOK:-0}" = 1 ] || return 0
  local repo="${REPO:?dip_hook::run needs REPO}"
  local -a run_args=()
  [ -n "${RUN_DIR:-}" ] && run_args+=(--run-dir "$RUN_DIR")
  echo "dip: topping up ${EE_TOOL:?dip_hook::run needs EE_TOOL} at the palette before the session (if needed)" >&2
  "$repo/scripts/il_dip.sh" --ee-tool "$EE_TOOL" --yes --if-needed "${run_args[@]}" "${DIP_HOOK_EXTRA[@]}"
}
