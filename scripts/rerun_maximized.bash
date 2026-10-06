# Interactive `rerun` for ~/.bashrc: every viewer window is capped and maximized.
#
#   source <checkout>/scripts/rerun_maximized.bash
#
# `rerun file.rrd` opens the file through rerun_session::view_file (memory cap
# VIEWER_MEMORY_LIMIT, default 1GB, maximized where an X11 window manager can
# be reached). A bare `rerun` attaches a capped window to the persistent
# viewer (`tatbot viewer open`). Subcommands (`rerun rrd ...`) and calls that
# already state --memory-limit pass straight through.

rerun() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --memory-limit|--memory-limit=*|--help|-h|--version|-V|rrd|reset|server|viewer-mcp)
        command rerun "$@"; return ;;
    esac
  done
  # shellcheck source=scripts/vision/rerun_session.sh
  source "${TATBOT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/vision/rerun_session.sh"
  if [ "$#" -eq 0 ]; then
    rerun_session::ensure_viewer
    return
  fi
  rerun_session::view_file "$@"
}
