# shellcheck shell=bash
# Content fingerprint of a check's guarded inputs, so a check that already
# passed can be recognized by local/CI callers: scripts/check's PASS cache and
# the sim 2.0 job pack (python/tatbot_sim2/check-jobs.sh). The name is older
# than that use. Source this; do not execute it.
#
#   source "$REPO/scripts/lib/sim_fingerprint.sh"
#   fp="$(sim_fingerprint::compute "$REPO" <guard function>)"
#   sim_fingerprint::is_fresh "$fp" && skip the run
#   sim_fingerprint::record "$fp"        # only after a PASS
#
# The key covers source content and the selected deployment/public profile.
# It does not cover installed packages, robot assets or render devices, so a
# source match alone never establishes that the current node has full coverage.

sim_fingerprint::guarded() {
  printf '%s\n' \
    python/tatbot_surface \
    python/tatbot_contracts \
    scripts/lib \
    scripts/check \
    config \
    urdf \
    ros/tatbot_ink \
    ros/tatbot_motion \
    ros/tatbot_description \
    ros/tatbot_session \
    ros/tatbot_bringup/config/stack.yaml \
    rust/tatbot-arm \
    web/inkmap/public/bodies \
    web/inkmap/src/core \
    web/inkmap/tools \
    config/inkmap \
    scripts/eval
}

# TATBOT_SIM_PASS_DIR moves the markers: a job that records many PASSes keeps
# them apart, so the 64-marker pruning below never evicts another job's.
sim_fingerprint::cache_dir() {
  echo "${TATBOT_SIM_PASS_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/tatbot/sim-pass}"
}

# The scripts/tests suite's guard: the shared inputs above plus everything
# else it reads.
sim_fingerprint::guarded_tests() {
  sim_fingerprint::guarded
  printf '%s\n' cpp/teleop
  # The fast Python suite runs tatbot-arm-guide and compares its compiled
  # source record with the checkout. Include every Rust input in that record,
  # even the workspace lockfile outside tatbot-arm's own directory.
  printf '%s\n' scripts python cpp/teleop web/inkmap/src web/inkmap/tools docs/cli.md \
    rust/Cargo.toml rust/Cargo.lock rust/trossen-arm-sys
  # A project that carries a scripts/check job pack runs its own tests/ there,
  # so editing those must not re-run this suite. The rest of the project stays
  # in the key: the CLI reads it (a verb's wrapped entry point, metadata a verb
  # scans), and this suite exercises the CLI.
  local pack
  for pack in python/*/check-jobs.sh; do
    [[ -f "$pack" ]] && printf ':(exclude)%s\n' "${pack%check-jobs.sh}tests"
  done
  return 0
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
  # In the checkout, so a guard that globs (a job pack) reads the repo it hashes.
  mapfile -t paths < <(cd "$repo" && "$guard")
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

# A scope suffix separates a full suite from a subset of it. The full
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
