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

# name <role>  — the node's KEY in config/nodes.json. Not derivable from the
# ssh target: `${target%%@*}` yields the LOGIN USER, which equals the node name
# only where a deployment happens to name them alike, and is wrong wherever one
# login is shared across nodes.
tatbot_nodes::name() {
  local role="$1" repo
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  python3 - "$repo" "$role" <<'PY'
import json, sys
from pathlib import Path
repo, role = Path(sys.argv[1]), sys.argv[2]
try:
    data = json.loads((repo / "config" / "nodes.json").read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(0)
for name, rec in data.items():
    if name.startswith(("//", "__")) or not isinstance(rec, dict):
        continue
    if role in rec.get("roles", []):
        print(name)
        break
PY
}

# checkout <role>  — the node's own checkout path (config/nodes.json), for
# commands that run THERE. Never assume it: a scrub once rewrote a remote
# checkout path as if it were local, and PoE capture silently stopped starting.
# The role owner's checkout as a remote shell sees it: `~/x` becomes `$HOME/x`
# so it expands inside a quoted command sent over ssh, where a tilde would not.
tatbot_nodes::remote_checkout() {
  local checkout
  checkout="$(tatbot_nodes::checkout "$1")"
  printf '%s\n' "${checkout/#\~/\$HOME}"
}

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

# hostname <role>  — what `hostname -s` reports on that node, which is not
# always the node name (config/nodes.json carries a `hostname` alias where they
# differ). Use it for "am I that node?" tests; `name` is for display and paths.
tatbot_nodes::hostname() {
  local role="$1" repo
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  python3 - "$repo" "$role" <<'PY'
import json, sys
from pathlib import Path
repo, role = Path(sys.argv[1]), sys.argv[2]
try:
    data = json.loads((repo / "config" / "nodes.json").read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(0)
for name, rec in data.items():
    if name.startswith(("//", "__")) or not isinstance(rec, dict):
        continue
    if role in rec.get("roles", []):
        print(rec.get("hostname") or name)
        break
PY
}

# this_node  — this machine's node NAME: TATBOT_NODE, else `hostname -s`
# mapped through any `hostname` alias in config/nodes.json, for deployments
# where they differ. Mirrors tatbot_cli.nodes.this_node so shell and Python agree.
tatbot_nodes::this_node() {
  local host repo
  if [ -n "${TATBOT_NODE:-}" ]; then echo "$TATBOT_NODE"; return; fi
  host="$(hostname -s)"
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  python3 - "$repo" "$host" <<'PY'
import json, sys
from pathlib import Path
repo, host = Path(sys.argv[1]), sys.argv[2]
try:
    data = json.loads((repo / "config" / "nodes.json").read_text())
except (OSError, json.JSONDecodeError):
    print(host); raise SystemExit(0)
for name, rec in data.items():
    if name.startswith(("//", "__")) or not isinstance(rec, dict):
        continue
    if rec.get("hostname") == host:
        print(name); break
else:
    print(host)
PY
}

