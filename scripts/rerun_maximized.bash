# Open the Rerun viewer maximized — rerun-cli has no fullscreen/maximize flag.
#
# Setup (once per machine), add to ~/.bashrc:
#   source scripts/rerun_maximized.bash
#
# Defines a `rerun` shell function wrapping the real binary: it launches the
# viewer, waits for its window, lets the viewer finish restoring its persisted
# geometry (~2 s; maximizing earlier gets undone and flickers), then maximizes
# once. Works with any screen size. On Wayland/headless sessions or without
# xdotool it degrades to plain `rerun`. Non-GUI subcommands (e.g. `rerun rrd
# stats`) are unaffected. Maximizing prefers wmctrl (EWMH, any window
# manager) and falls back to GNOME's Super+Up keybinding via xdotool.

rerun() {
  # Maximize only when the viewer will open as an X11 window xdotool can
  # reach: no Wayland (the viewer would open a Wayland-native window), a
  # reachable X DISPLAY, and xdotool present. Otherwise run rerun plainly.
  if [ -n "${WAYLAND_DISPLAY:-}" ] || [ -z "${DISPLAY:-}" ] \
    || ! command -v xdotool >/dev/null 2>&1 \
    || ! xdotool getdisplaygeometry >/dev/null 2>&1; then
    command rerun "$@"
    return
  fi
  # set +m silences the "[1] pid" / "Done" job-control noise.
  set +m
  command rerun "$@" &
  local pid=$!
  trap 'kill '"$pid"' 2>/dev/null' INT TERM
  local wid="" i
  for i in $(seq 1 60); do
    wid=$(xdotool search --class rerun 2>/dev/null | head -n1)
    [ -n "$wid" ] && break
    # No window and the process exited (non-GUI subcommand): stop waiting.
    if ! kill -0 "$pid" 2>/dev/null && [ "$i" -gt 4 ]; then break; fi
    sleep 0.25
  done
  if [ -n "$wid" ]; then
    sleep 2
    _rerun_maximize "$wid"
  fi
  wait "$pid"
  local status=$?
  trap - INT TERM
  set -m
  return $status
}

_rerun_maximize() {
  local wid=$1
  xprop -id "$wid" _NET_WM_STATE 2>/dev/null | grep -q MAXIMIZED && return 0
  if command -v wmctrl >/dev/null 2>&1; then
    wmctrl -i -r "$wid" -b add,maximized_vert,maximized_horz 2>/dev/null && return 0
  fi
  # GNOME fallback: focus the window, then its maximize keybinding. The pause
  # after focusing is required for the keybinding to land on the new window.
  for _ in 1 2 3; do
    xdotool windowactivate --sync "$wid" 2>/dev/null
    sleep 0.4
    xdotool key --clearmodifiers super+Up 2>/dev/null
    sleep 0.6
    xprop -id "$wid" _NET_WM_STATE 2>/dev/null | grep -q MAXIMIZED && return 0
  done
}
