#!/usr/bin/env bash
# Independent material tracker; subscribes to the existing frame owner only.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot vision track-target"
# shellcheck source=scripts/lib/runlog.sh
source "$REPO/scripts/lib/runlog.sh"; runlog::init target-track
[[ -n "$RUN_DIR" ]] || { echo "target tracker requires a run log" >&2; exit 5; }
source_sha="$(python3 "$REPO/scripts/lib/fleet_source.py" --repo "$REPO" --require-clean)"
export TATBOT_SOURCE_COMMIT="$source_sha"
runlog::run cargo build --quiet --locked --manifest-path "$REPO/rust/Cargo.toml" \
  -p trackd --message-format=json-render-diagnostics > "$RUN_DIR/build-artifacts.jsonl"
python3 "$REPO/scripts/lib/fleet_source.py" --repo "$REPO" --require-clean --expected "$source_sha" >/dev/null
daemon="$(python3 -c 'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1])]; print(next(r["executable"] for r in reversed(rows) if r.get("reason")=="compiler-artifact" and r.get("target",{}).get("name")=="trackd" and r.get("executable")))' "$RUN_DIR/build-artifacts.jsonl")"
runlog::run "$daemon" "$@"
