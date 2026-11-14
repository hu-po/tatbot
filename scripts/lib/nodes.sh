# shellcheck shell=bash
# Fleet lookups for shell launchers — config/nodes.json is the only place
# node names and ssh targets live. Source after REPO is set:
#
#   source "$REPO/scripts/lib/nodes.sh"
#   ARM="$(tatbot_nodes::target arm)"   # user@host, empty if no such node
#
# Empty output means "no node carries that role here" — callers decide
# whether that is fatal or a skipped optional step.

# _field <role> <fallback> <field>...  — the first non-empty field of the first
# node carrying the role, else the fallback. `name` is the node's key.
tatbot_nodes::_field() {
  local repo
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  python3 - "$repo" "$@" <<'PY'
import json, sys
from pathlib import Path
repo, role, fallback, *fields = Path(sys.argv[1]), *sys.argv[2:]
try:
    data = json.loads((repo / "config" / "nodes.json").read_text())
except (OSError, json.JSONDecodeError):
    data = {}
value = fallback
for name, rec in data.items():
    if name.startswith(("//", "__")) or not isinstance(rec, dict):
        continue
    if role in rec.get("roles", []):
        record = {**rec, "name": name}
        value = next((record[field] for field in fields if record.get(field)), fallback)
        break
if value:
    print(value)
PY
}

# target <role> [lan]  — "lan" prefers the node's ssh_lan (bulk transfers).
tatbot_nodes::target() {
  if [ "${2:-}" = lan ]; then
    tatbot_nodes::_field "$1" "" ssh_lan ssh
  else
    tatbot_nodes::_field "$1" "" ssh
  fi
}

# name <role>  — the node's KEY in config/nodes.json. Not derivable from the
# ssh target: `${target%%@*}` yields the LOGIN USER, which equals the node name
# only where a deployment happens to name them alike, and is wrong wherever one
# login is shared across nodes.
tatbot_nodes::name() {
  tatbot_nodes::_field "$1" "" name
}

# checkout <role>  — the node's own checkout path (config/nodes.json), for
# commands that run THERE. Never assume it: a scrub once rewrote a remote
# checkout path as if it were local, and PoE capture silently stopped starting.
tatbot_nodes::checkout() {
  tatbot_nodes::_field "$1" \~/tatbot checkout
}

# The role owner's checkout as a remote shell sees it: `~/x` becomes `$HOME/x`
# so it expands inside a quoted command sent over ssh, where a tilde would not.
tatbot_nodes::remote_checkout() {
  local checkout
  checkout="$(tatbot_nodes::checkout "$1")"
  printf '%s\n' "${checkout/#\~/\$HOME}"
}

# hostname <role>  — what `hostname -s` reports on that node, which is not
# always the node name (config/nodes.json carries a `hostname` alias where they
# differ). Use it for "am I that node?" tests; `name` is for display and paths.
tatbot_nodes::hostname() {
  tatbot_nodes::_field "$1" "" hostname name
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

