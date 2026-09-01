#!/usr/bin/env bash
# Compare two policies on the robot, honestly.
#
#   il_compare_policies.sh <name>=<policy_dir> <name>=<policy_dir> [more...] [reps] [duration]
#
# e.g.
#   M=${TATBOT_SERVE_ROOT:-$HOME/il-serve}/models
#   il_compare_policies.sh \
#     h64=$M/multi_task_dit_draw_square_horizon64/checkpoints/last/pretrained_model \
#     h64flow=$M/multi_task_dit_draw_square_h64flow/checkpoints/last/pretrained_model 3 30
#
# WHY THIS EXISTS, AND WHY IT INTERLEAVES (docs/imitation_learning.md):
#
#   - Two or more contenders are interleaved in the order given. With three,
#     that is A B C A B C A B C.
#   - One rollout per policy cannot rank anything. Three runs of ONE checkpoint
#     spanned 78-876 mm^2 of drawn area on 2026-08-21 — an 11x spread from
#     identical weights, because stochastic chunk policies can sample a fresh
#     trajectory every horizon. Hence reps>=3, and the SPREAD is the result.
#   - The paper changes under you. A policy's willingness to descend falls off
#     as the sheet fills with marks (measured: 74% -> 50% -> 30% across one
#     session). A block design — all of A, then all of B — hands one policy a
#     clean sheet and the other a scribbled one, and that effect is large enough
#     to decide the comparison on its own. So the runs INTERLEAVE: A B A B A B.
#   - Start from a FRESH SHEET, or the first policy inherits the last session's
#     marks. This script cannot check that; the operator must.
#
# Each policy runs at its NATIVE chunk length, read from its own checkpoint —
# chunk length is part of the policy, not a knob. Pass TATBOT_N_ACTION_STEPS
# only if this host cannot reach the policy server to read config.json.
#
# Aborts if any client is ever left running: an orphaned client still driving
# the arm is the failure mode that produced the 2026-08-21 incident.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot rollout compare"

# Takes two or more policies: every leading name=path argument is a contender,
# and the optional reps and duration follow. Two-policy calls are unchanged.
NAMES=(); POLS=()
while [ $# -gt 0 ] && [[ "$1" == *=* ]]; do
  NAMES+=("${1%%=*}"); POLS+=("${1#*=}"); shift
done
[ "${#NAMES[@]}" -ge 2 ] || { echo "need at least two name=policy_dir args" >&2; exit 1; }

# reps and duration if given as numbers, then anything else is forwarded to the
# rollout (e.g. --robot.overforce_limit=4.0). Trailing arguments used to be
# accepted and silently dropped, which on 2026-08-21 meant a safety limit that
# was passed here never reached the robot and the guard stayed at its default.
REPS=3; DURATION=30
[[ "${1:-}" =~ ^[0-9]+$ ]] && { REPS="$1"; shift; }
[[ "${1:-}" =~ ^[0-9]+$ ]] && { DURATION="$1"; shift; }
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    -*) EXTRA+=("$1"); shift ;;
    *)  echo "unrecognised argument: $1" >&2
        echo "usage: name=dir name=dir [more...] [reps] [duration] [--robot.x=y ...]" >&2
        exit 1 ;;
  esac
done
[ "${#EXTRA[@]}" -gt 0 ] && echo "forwarding to each rollout: ${EXTRA[*]}"

echo "paired comparison: ${NAMES[*]}, ${REPS} reps each, ${DURATION}s per run"
echo "interleaved $(printf '%s ' "${NAMES[@]}")— repeating — so no policy gets a cleaner sheet"
echo "than another. Put a FRESH SHEET on the pad before starting."
echo

run_one() {
  local tag="$1" pol="$2"
  local policy_type use_depth state_size depth_encoding relative server ssh_target config
  echo "########## $tag ##########"
  if pgrep -f "[i]l_client_shield.py" >/dev/null 2>&1; then
    echo "ABORT: a rollout client is already running" >&2
    pgrep -af "[i]l_client_shield.py" >&2
    exit 1
  fi
  server="${TATBOT_POLICY_SERVER:-$(tatbot_nodes::target serve | sed "s/.*@//"):8080}"
  ssh_target="${TATBOT_POLICY_SSH:-$(tatbot_nodes::target serve | sed "s/@.*//")@${server%%:*}}"
  config="$(ssh -n -o BatchMode=yes -o ConnectTimeout=3 "$ssh_target" "cat '$pol/config.json'" 2>/dev/null)" || {
    echo "ABORT: cannot read checkpoint contract at $ssh_target:$pol/config.json" >&2
    exit 1
  }
  IFS='|' read -r policy_type use_depth state_size depth_encoding relative mask_ext_eff < <(
    python3 "$REPO/scripts/eval/checkpoint_contract.py" --format fields - <<<"$config"
  )
  [ "$state_size" = 7 ] || [ "$state_size" = 14 ] || {
    echo "ABORT: unsupported checkpoint state width $state_size" >&2
    exit 1
  }
  [ "$mask_ext_eff" = 1 ] && {
    echo "ABORT: $tag was trained with external effort masked to zero; the wire" >&2
    echo "  can only include or drop those channels, not zero them, so serving it" >&2
    echo "  live effort is train/serve skew. Teach the client to zero them first." >&2
    exit 1
  }
  echo "checkpoint contract: type=$policy_type depth=$use_depth state=$state_size relative=$relative"
  TATBOT_DEPTH="$use_depth" \
  TATBOT_DEPTH_ENCODING="$depth_encoding" \
  TATBOT_EXT_EFF=$([ "$state_size" = 14 ] && echo 1 || echo 0) \
  "$REPO/scripts/il_rollout_async.sh" "$DURATION" "$pol" "$policy_type" "${EXTRA[@]}" 2>&1 \
    | grep -E "tatbot-run (start|end)|geometry|motion|smoothness|dwell|timing|WARN"
  sleep 4
  if pgrep -f "[i]l_client_shield.py" >/dev/null 2>&1; then
    echo "ABORT: a client survived $tag — the arm may still be driven" >&2
    pgrep -af "[i]l_client_shield.py" >&2
    exit 1
  fi
  sleep 3
}

for i in $(seq 1 "$REPS"); do
  for k in "${!NAMES[@]}"; do
    run_one "${NAMES[$k]} $i/$REPS" "${POLS[$k]}"
  done
done

echo "########## complete — compare with: ##########"
echo "  scripts/il_analyze_rollout.py --compare ~/tatbot-logs/rollout_async/*/analysis.json"
echo "Read the SPREAD, not just the mean. And check the runs shared a control"
echo "rate and an observation window — --compare warns when they did not."
