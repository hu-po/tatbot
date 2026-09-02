#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/lib/cli_hint.sh
source "$repo/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot inkmap rig"
command -v blender >/dev/null || {
  echo "inkmap rig: Blender 4.0 or newer is required" >&2
  exit 1
}

log=$(mktemp)
trap 'rm -f "$log"' EXIT
set +e
blender --background --factory-startup \
  --python "$repo/web/inkmap/tools/rig-hbm.py" -- --repo "$repo" "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
if [[ $status -ne 0 ]] || ! grep -qx 'INKMAP_RIG_OK' "$log"; then
  echo "inkmap rig: generation failed; Blender's Python must provide NumPy 1.26" >&2
  echo "see docs/inkmap.md#named-body-poses for setup and numerical gates" >&2
  exit 1
fi
