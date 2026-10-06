#!/usr/bin/env bash
# LeRobot borrows the D405 owner for its lifetime. Never changes arm signals.
D405_OWNER_RESTORE=0
d405_owner::borrow() {
  local state
  state="$(systemctl show tatbot-visiond-d405.service -p LoadState --value 2>/dev/null)" || return 5
  [[ "$state" == not-found ]] && return 0
  if systemctl is-active --quiet tatbot-visiond-d405.service; then
    sudo -n systemctl stop tatbot-visiond-d405.service || return 5
    if systemctl is-active --quiet tatbot-visiond-d405.service; then
      echo 'D405 owner did not stop; refusing camera handoff' >&2
      return 6
    fi
    D405_OWNER_RESTORE=1
  fi
}
d405_owner::restore() {
  [[ "$D405_OWNER_RESTORE" == 1 ]] || return 0
  # If an interrupted wrapper still has a LeRobot child, leave the service
  # stopped. Starting a second librealsense owner would hide a failed cleanup.
  if pgrep -f '^([^ ]*/)?(lerobot-record([[:space:]]|$)|lerobot-rollout([[:space:]]|$)|python[^ ]* .*([/]il_client_shield[.]py|lerobot[.]))' >/dev/null; then
    echo 'D405 owner remains stopped: LeRobot process still alive; restore after its existing landing/cleanup completes' >&2
    return 6
  fi
  sudo -n systemctl start tatbot-visiond-d405.service || return 5
  D405_OWNER_RESTORE=0
}
# Called only by EXIT traps, after the launcher's existing signal/landing path.
d405_owner::finish() {
  local exit_status="$1"
  shift
  "$@" || exit_status=5
  runlog::finalize "$exit_status"
  trap - EXIT
  exit "$exit_status"
}
