#!/usr/bin/env bash
# Launch-id ledger and audit for AUTONOMOUS-motion launchers.
#
# Every autonomous launch carries a unique launch id,
# <utc %Y%m%dT%H%M%SZ>-<host>-<4 hex>[-<tag>]. The CLI mints it right before
# exec and writes it to /tmp/tatbot-arm-token; arm_gate::require reads it
# (minting one itself when the file is missing, stale or empty), appends it
# to the ledger, and audits every decision with the pid chain up to sshd and
# the SSH_CONNECTION. Nothing here refuses a launch: the trail exists because
# on 2026-08-24 three rollout launches fired that nobody could attribute
# (see the squiggle robot eval), and an attributable record is what was
# missing. A repeated id is audited and warned about, not refused.
#
# Scope: every launcher that moves the arm on its own (policy rollouts,
# scripted measurement moves).
# Ordinary teleop/record keep a human physically on the leader arm and carry
# no launch id.
#
# Nesting: a launcher may run another gated launcher as a child inside the
# same launch. A pass exports TATBOT_ARM_ARMED=<id> and TATBOT_ARM_ARMED_PID=$$; a child
# reuses them only when that pid is one of its own ancestors AND the id is
# the LAST line of the ledger — i.e. it was ledgered by the launch this
# process is running inside. Anything else (a stale export in an interactive
# shell) gets a fresh id of its own.

arm_gate::audit() {
  # Forensic trail for every decision: who asked — pid chain up to sshd and
  # the SSH_CONNECTION if any.
  local verdict="$1" id="$2"
  {
    printf '%s pid=%s verdict=%s id=%s ssh=[%s] chain=' \
      "$(date -u +%FT%T.%3NZ)" "$$" "$verdict" "$id" "${SSH_CONNECTION:-none}"
    local p=$$
    for _ in 1 2 3 4 5; do
      printf '%s:' "$(ps -o comm= -p "$p" 2>/dev/null | tr -d ' ')"
      p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
      [ -z "$p" ] || [ "$p" -le 1 ] && break
    done
    echo
  } >> /var/tmp/tatbot-arm-gate-audit.log 2>/dev/null || true
}

arm_gate::_is_ancestor() {
  # Is pid $1 an ancestor of this process (up to 12 levels)?
  local want="$1" p=$$
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -z "$p" ] || [ "$p" -le 1 ] && return 1
    [ "$p" = "$want" ] && return 0
  done
  return 1
}

arm_gate::mint() {
  # Same shape the CLI mints: <utc>-<host>-<4 hex>.
  local host hex
  host="${HOSTNAME:-$(hostname 2>/dev/null)}"; host="${host%%.*}"
  host="$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]' | tr -cd 'A-Za-z0-9_-' | head -c 16)"
  hex="$(od -An -N2 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n')"
  [ -n "$hex" ] || hex="$(printf '%04x' $((RANDOM % 65536)))"
  printf '%s-%s-%s' "$(date -u +%Y%m%dT%H%M%SZ)" "${host:-node}" "$hex"
}

arm_gate::require() {
  local token=/tmp/tatbot-arm-token
  local ledger=/var/tmp/tatbot-launch-ids
  local id=""
  if [ -n "${TATBOT_ARM_ARMED:-}" ]; then
    local inherited last
    inherited="$(printf '%s' "$TATBOT_ARM_ARMED" | head -c 128 | tr -cd 'A-Za-z0-9_-')"
    last="$(tail -n 1 "$ledger" 2>/dev/null || true)"
    if [ -n "$inherited" ] && [ "$inherited" = "$last" ] \
        && [ -n "${TATBOT_ARM_ARMED_PID:-}" ] && arm_gate::_is_ancestor "$TATBOT_ARM_ARMED_PID"; then
      arm_gate::audit pass-inherited "$inherited"
      return 0
    fi
    # A stale or mismatched export (not the last ledger line, or its launcher
    # is not an ancestor): this is its own launch, so it gets its own id.
    id="$(arm_gate::mint)"
    arm_gate::audit fresh-after-stale-inherit "$id"
    echo "arm_gate: TATBOT_ARM_ARMED was not this launch's id; minted launch id $id" >&2
  else
    if [ -f "$token" ] && [ $(( $(date +%s) - $(stat -c %Y "$token") )) -le 120 ]; then
      id="$(head -c 128 "$token" | tr -cd 'A-Za-z0-9_-')"
    fi
    rm -f "$token"
    if [ -z "$id" ]; then
      id="$(arm_gate::mint)"
      arm_gate::audit minted "$id"
      echo "arm_gate: no launch id from the CLI (token missing, stale or empty); minted launch id $id" >&2
    fi
  fi
  touch "$ledger"
  if grep -qx "$id" "$ledger"; then
    arm_gate::audit repeat "$id"
    echo "arm_gate: launch id '$id' is already in $ledger — a repeated id; audited, continuing." >&2
  else
    arm_gate::audit pass "$id"
  fi
  echo "$id" >> "$ledger"
  # Children that move the arm inside this launch
  # inherit the id instead of minting a second one.
  export TATBOT_ARM_ARMED="$id" TATBOT_ARM_ARMED_PID="$$"
}
