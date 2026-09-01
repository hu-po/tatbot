#!/usr/bin/env bash
# Production launchers fail closed. Hardware-free work must use the lower-level
# component command explicitly; pass-through arguments may not weaken safety.

estop_guard::reject_overrides() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --no-estop|--estop|--estop=*|--robot.estop_device*|--robot.estop_required*|--teleop.estop_device*|--teleop.estop_required*)
        echo "refusing E-stop override in a production launcher: $arg" >&2
        echo "use the lower-level component command for an intentional hardware-free bench run" >&2
        return 2
        ;;
    esac
  done
}
