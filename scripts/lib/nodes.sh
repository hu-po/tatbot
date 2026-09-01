# shellcheck shell=bash
# Fleet lookups for shell launchers — config/nodes.json is the only place
# node names and ssh targets live (plan Phase 5). Source after REPO is set:
#
#   source "$REPO/scripts/lib/nodes.sh"
#   POE="$(tatbot_nodes::target poe-cameras)"   # user@host, empty if no such node
#
# Empty output means "no node carries that role here" — callers decide
# whether that is fatal or a skipped optional step.

# target <role> [lan]  — "lan" prefers the node's ssh_lan (bulk transfers).
tatbot_nodes::target() {
  local role="$1" want="${2:-}" repo
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  python3 - "$repo" "$role" "$want" <<'PY'
import json, sys
from pathlib import Path
repo, role = Path(sys.argv[1]), sys.argv[2]
want = sys.argv[3] if len(sys.argv) > 3 else ""
try:
    data = json.loads((repo / "config" / "nodes.json").read_text())
except (OSError, json.JSONDecodeError):
    sys.exit(0)
for name, rec in data.items():
    if name.startswith(("//", "__")) or not isinstance(rec, dict):
        continue
    if role in rec.get("roles", []):
        print((rec.get("ssh_lan") if want == "lan" else "") or rec.get("ssh", ""))
        break
PY
}

tatbot_nodes::name() {
  local t; t="$(tatbot_nodes::target "$1")"; echo "${t%@*}"
}

# checkout <role>  — the node's own checkout path (config/nodes.json), for
# commands that run THERE. Never assume it: a scrub once rewrote a remote
# checkout path as if it were local, and PoE capture silently stopped starting.
tatbot_nodes::checkout() {
  local role="$1" repo
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  python3 - "$repo" "$role" <<'PY'
import json, sys
from pathlib import Path
repo, role = Path(sys.argv[1]), sys.argv[2]
try:
    data = json.loads((repo / "config" / "nodes.json").read_text())
except (OSError, json.JSONDecodeError):
    print("~/tatbot"); raise SystemExit(0)
for name, rec in data.items():
    if name.startswith(("//", "__")) or not isinstance(rec, dict):
        continue
    if role in rec.get("roles", []):
        print(rec.get("checkout") or "~/tatbot")
        break
else:
    print("~/tatbot")
PY
}
