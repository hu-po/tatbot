#!/usr/bin/env bash
# Which end-effector tool is in the gripper — stated, never inferred.
#
# Grip force, tip offset and reach all come from the fitted tool's datasheet.
# Until 2026-08-26 the launchers said nothing and the Python side read the tool
# out of config/workspace.yaml, which records the tool the last TOUCH-OFF used.
# After a physical swap that names the PREVIOUS tool and hands over a complete,
# plausible, wrong set of constants: a 130 mm laser pen driven with a 63.7 mm
# tip offset and a grip force chosen for a machined pen body. Stating the tool
# makes a swap visible; tool_spec.require_stated_tool then refuses when the
# statement and the calibration disagree.
#
#   source "$REPO/scripts/lib/ee_tool.sh"
#   ee_tool::strip "$@"; set -- "${EE_TOOL_ARGS[@]}"   # before positionals
#   ...usage / policy checks...
#   ee_tool::require                                    # before hardware
#   ... --robot.ee_tool="$EE_TOOL"
#
# Two steps on purpose. The strip has to happen before a launcher reads its
# positionals, or --ee-tool is swallowed as one of them. The REFUSAL belongs
# later, after the cheap argument and checkpoint validation a launcher does on
# its own: a bare invocation should answer with its usage, and a policy the
# stack cannot run should say so, before either is buried under a question
# about which pen is fitted. Tool identity is a hardware concern, so it gates
# where the other hardware concerns do — just before the arm gate.

ee_tool::_known() {
  local dir="$1" f names=""
  for f in "$dir"/*.yaml; do
    [ -e "$f" ] || continue
    names="${names:+$names, }$(basename "$f" .yaml)"
  done
  echo "${names:-none}"
}

ee_tool::strip() {
  local repo="${REPO:?ee_tool::strip needs REPO}"
  EE_TOOL=""
  EE_TOOL_ARGS=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --ee-tool)
        [ "$#" -ge 2 ] || { echo "--ee-tool needs a value" >&2; return 2; }
        EE_TOOL="$2"; shift 2 ;;
      --ee-tool=*) EE_TOOL="${1#--ee-tool=}"; shift ;;
      *) EE_TOOL_ARGS+=("$1"); shift ;;
    esac
  done
}

ee_tool::require() {
  local repo="${REPO:?ee_tool::require needs REPO}"
  local tools="$repo/config/tools"
  if [ -z "${EE_TOOL:-}" ]; then
    echo "--ee-tool <id> is required: name the tool in the gripper." >&2
    echo "  Grip force, tip offset and reach all come from its datasheet, and" >&2
    echo "  guessing means using the last tool's numbers on this one." >&2
    echo "  known tools: $(ee_tool::_known "$tools")" >&2
    return 2
  fi
  if [ ! -f "$tools/$EE_TOOL.yaml" ]; then
    echo "unknown --ee-tool '$EE_TOOL' — no $tools/$EE_TOOL.yaml" >&2
    echo "  known tools: $(ee_tool::_known "$tools")" >&2
    return 2
  fi
  # The deeper check — does this agree with what workspace.yaml was calibrated
  # with — belongs to tool_spec.require_stated_tool, which the follower runs at
  # connect. Doing it here too would be a second copy of the rule to drift.
}
