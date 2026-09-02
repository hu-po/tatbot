#!/usr/bin/env bash
# Single-use arming gate for AUTONOMOUS-motion launchers (policy rollouts,
# scripted dips, and the one-shot Cartesian square probe).
#
# Why: on 2026-08-24 three rollout launches fired that nobody initiated —
# each a byte-identical, non-interactive replay of a previously-executed
# command, ~7-8 s after the arm went idle (launcher never identified; see
# 2026-08-24 squiggle robot eval). Possessing a valid command
# string is therefore not proof of present intent. Every launch must be
# armed with a nonce the operator/session writes immediately before it:
#
#   echo <unique-literal-nonce> > /tmp/tatbot-arm-token && <launcher ...>
#
# The nonce must be a LITERAL (never $RANDOM or $(date ...): a shell replay
# re-evaluates substitutions and mints a fresh nonce). Consumed nonces are
# ledgered; a repeat refuses and points at the ancestry tripwire.
#
# Scope: every launcher that moves the arm on its own — policy rollouts, the
# scripted dip (il_dip.sh), and teleop_square.sh after its explicit handoff.
# Ordinary teleop/record keep a human physically on the leader arm and stay
# ungated, EXCEPT that their --dip runs il_dip.sh first, which is gated like
# any other autonomous motion.
#
# Nesting: a rollout's --dip runs il_dip.sh as a child AFTER the rollout has
# consumed its nonce. The child must not demand a second one (the operator
# armed this launch once, on purpose), and must not be fooled by a stale
# export in an interactive shell. So a pass exports TATBOT_ARM_ARMED=<nonce>
# and TATBOT_ARM_ARMED_PID=$$, and a child accepts them only when that pid is
# one of its own ancestors AND the nonce is the LAST line of the consumed
# ledger — i.e. it was consumed by the launch this process is running inside.

arm_gate::audit() {
  # Forensic trail for every gate decision. The 2026-08-24 phantom evaded
  # the gate invisibly because a PASS wrote nothing — now both verdicts
  # record who asked: pid chain up to sshd and the SSH_CONNECTION if any.
  local verdict="$1" nonce="$2"
  {
    printf '%s pid=%s verdict=%s nonce=%s ssh=[%s] chain=' \
      "$(date -u +%FT%T.%3NZ)" "$$" "$verdict" "$nonce" "${SSH_CONNECTION:-none}"
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

arm_gate::require() {
  local token=/tmp/tatbot-arm-token
  local ledger=/var/tmp/tatbot-consumed-nonces
  if [ -n "${TATBOT_ARM_ARMED:-}" ]; then
    local inherited last
    inherited="$(printf '%s' "$TATBOT_ARM_ARMED" | head -c 128 | tr -cd 'A-Za-z0-9_-')"
    last="$(tail -n 1 "$ledger" 2>/dev/null || true)"
    if [ -n "$inherited" ] && [ "$inherited" = "$last" ] \
        && [ -n "${TATBOT_ARM_ARMED_PID:-}" ] && arm_gate::_is_ancestor "$TATBOT_ARM_ARMED_PID"; then
      arm_gate::audit pass-inherited "$inherited"
      return 0
    fi
    arm_gate::audit refuse-stale-inherit "${inherited:--}"
    echo "REFUSING TO LAUNCH: TATBOT_ARM_ARMED is set but is not this launch's nonce" >&2
    echo "  (not the last consumed nonce, or its launcher is not an ancestor of this process)." >&2
    echo "  unset TATBOT_ARM_ARMED TATBOT_ARM_ARMED_PID and arm this launch with a fresh nonce." >&2
    return 2
  fi
  if [ ! -f "$token" ] || [ $(( $(date +%s) - $(stat -c %Y "$token") )) -gt 120 ]; then
    arm_gate::audit refuse-no-token "-"
    echo "REFUSING TO LAUNCH: no fresh arm token." >&2
    echo "  arm this launch (valid 120 s, single use):" >&2
    echo "    echo <unique-literal-nonce> > $token" >&2
    return 2
  fi
  local nonce
  nonce="$(head -c 128 "$token" | tr -cd 'A-Za-z0-9_-')"
  rm -f "$token"
  if [ -z "$nonce" ]; then
    arm_gate::audit refuse-empty-nonce "-"
    echo "REFUSING TO LAUNCH: arm token is empty — write a unique literal nonce into it." >&2
    return 2
  fi
  touch "$ledger"
  if grep -qx "$nonce" "$ledger"; then
    arm_gate::audit refuse-replayed-nonce "$nonce"
    echo "REFUSING TO LAUNCH: arm nonce '$nonce' already consumed — this is a REPLAYED command." >&2
    echo "  capture /tmp/phantom-ancestry.log NOW." >&2
    return 3
  fi
  echo "$nonce" >> "$ledger"
  arm_gate::audit pass "$nonce"
  # Children that move the arm inside this launch (dip_hook -> il_dip.sh)
  # inherit the arming instead of demanding a second nonce.
  export TATBOT_ARM_ARMED="$nonce" TATBOT_ARM_ARMED_PID="$$"
}
