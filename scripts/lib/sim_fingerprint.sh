# shellcheck shell=bash
# Content fingerprint of the simulator's guarded inputs, so a `scripts/check sim`
# that already passed can be recognized by local/CI callers. Source this; do not
# execute it.
#
#   source "$REPO/scripts/lib/sim_fingerprint.sh"
#   fp="$(sim_fingerprint::compute "$REPO")"
#   sim_fingerprint::is_fresh "$fp" && skip the run
#   sim_fingerprint::record "$fp"        # only after a PASS
#
# The key covers source content and the selected deployment/public profile.
# It does not cover installed packages, robot assets or render devices, so a
# source match alone never establishes that the current node has full coverage.
# The push hook does not run simulation checks.

sim_fingerprint::guarded() {
  printf '%s\n' \
    python/tatbot_sim \
    python/tatbot_contracts \
    scripts/lib \
    scripts/check \
    config \
    urdf \
    cpp/teleop \
    rust/tatbot-arm \
    web/inkmap/public/bodies \
    web/inkmap/src/core \
    web/inkmap/tools \
    config/inkmap \
    scripts/eval
}

sim_fingerprint::cache_dir() {
  echo "${XDG_CACHE_HOME:-$HOME/.cache}/tatbot/sim-pass"
}

# The scripts/tests suite reads more of the tree than the simulator does; the
# same cache serves it under its own scope with this wider guard.
sim_fingerprint::guarded_tests() {
  sim_fingerprint::guarded
  # The fast Python suite runs tatbot-arm-guide and compares its compiled
  # source record with the checkout. Include every Rust input in that record,
  # even the workspace lockfile outside tatbot-arm's own directory.
  printf '%s\n' scripts python cpp/teleop web/inkmap/src web/inkmap/tools docs/cli.md \
    rust/Cargo.toml rust/Cargo.lock rust/trossen-arm-sys
}

# sha256 over the working-tree contents of every tracked and every
# not-ignored-untracked file under the guarded paths, plus the profile the
# suite would run under (a checkout without config/workspace.yaml runs the
# smaller public profile, so its PASS must not satisfy a full checkout).
# A second argument names the function that lists the guarded paths.
sim_fingerprint::compute() {
  local repo="${1:-$PWD}" guard="${2:-sim_fingerprint::guarded}" profile=full
  [[ -f "$repo/config/workspace.yaml" && -f "$repo/config/trossen/tatbot.yaml" ]] || profile=public
  local -a paths=()
  mapfile -t paths < <("$guard")
  {
    printf 'profile=%s\n' "$profile"
    (
      cd "$repo" || exit 1
      { git ls-files -z -- "${paths[@]}"
        git ls-files -z --others --exclude-standard -- "${paths[@]}"
      } | sort -z | xargs -0 -r sha256sum 2>/dev/null
    )
  } | sha256sum | cut -d' ' -f1
}

# A scope suffix separates the full suite from the sim-fast subset. The full
# run is a superset, so a full PASS satisfies a subset question but not the
# reverse -- sim_fingerprint::is_fresh "$fp" fast accepts either marker.
sim_fingerprint::is_fresh() {
  local fp="${1:-}" scope="${2:-}" dir
  [[ -n "$fp" ]] || return 1
  dir="$(sim_fingerprint::cache_dir)"
  [[ -f "$dir/$fp" ]] && return 0
  [[ -n "$scope" && -f "$dir/$fp.$scope" ]]
}

# When the matching PASS was recorded, for the line that reports the cache hit.
sim_fingerprint::recorded_at() {
  local fp="${1:?fingerprint required}" scope="${2:-}" dir
  dir="$(sim_fingerprint::cache_dir)"
  if [[ -n "$scope" && -f "$dir/$fp.$scope" ]]; then cat "$dir/$fp.$scope"
  else cat "$dir/$fp" 2>/dev/null; fi
}

sim_fingerprint::record() {
  local fp="${1:?fingerprint required}" scope="${2:-}" dir marker
  dir="$(sim_fingerprint::cache_dir)"
  marker="$fp"; [[ -n "$scope" ]] && marker="$fp.$scope"
  mkdir -p "$dir" || return 0
  date -u +%Y-%m-%dT%H:%M:%SZ > "$dir/$marker" 2>/dev/null || return 0
  # Keep the directory from growing without bound across months of rebases.
  local -a old=()
  mapfile -t old < <(ls -1t "$dir" 2>/dev/null | tail -n +64)
  local f
  for f in "${old[@]}"; do rm -f -- "${dir:?}/$f"; done
  return 0
}
