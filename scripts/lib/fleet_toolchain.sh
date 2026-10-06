# shellcheck shell=bash
# Optional exact-version runtime supplied beside an immutable staged source.
# This changes only this process and its children; never shell startup files.

# The per-user tool bins. `uv`, `cargo` and `rerun` install under ~/.local/bin
# and ~/.cargo/bin, which only ~/.profile puts on PATH — so a non-login shell
# (a tmux pane, an `ssh host cmd`, a systemd unit, an editor's terminal) runs
# the CLI with neither, and a launcher dies with `uv: command not found` at
# exit 127 AFTER its gates ran. Appended, never prepended: a system or venv
# tool already on PATH still wins.
fleet_toolchain::user_bins() {
  local tool_bin node_bin
  for tool_bin in "$HOME/.cargo/bin" "$HOME/.local/bin"; do
    [[ -d "$tool_bin" && ":${PATH:-}:" != *":$tool_bin:"* ]] && export PATH="${PATH:+$PATH:}$tool_bin"
  done
  # An explicitly provisioned Tatbot runtime avoids changing the system Node.
  # A release-specific runtime still takes precedence in ::use below.
  node_bin="$HOME/.local/share/tatbot/toolchain/node/bin"
  if [[ -x "$node_bin/node" && ":${PATH:-}:" != *":$node_bin:"* ]]; then
    export PATH="$node_bin${PATH:+:$PATH}"
  fi
  return 0
}

# A node-provisioned RealSense SDK built with DDS, as static archives, wins
# release builds' pkg-config lookup there. The distribution package has no DDS,
# which the Ethernet D555 needs; static linking keeps the SDK inside the
# release's binaries, so nothing at run time can bind the package instead.
fleet_toolchain::realsense_dds() {
  local pc_dir="$HOME/.local/share/tatbot/toolchain/librealsense-dds/lib/pkgconfig"
  [[ -f "$pc_dir/realsense2.pc" ]] || return 0
  [[ ":${PKG_CONFIG_PATH:-}:" == *":$pc_dir:"* ]] && return 0
  export PKG_CONFIG_PATH="$pc_dir${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
}

fleet_toolchain::use() {
  local source_root node_bin assets
  source_root="$1"
  [[ "${source_root##*/}" == source && ! -e "$source_root/.git" ]] || return 0
  node_bin="$(dirname "$source_root")/toolchain/node/bin"
  assets="$(dirname "$source_root")/toolchain/maniskill"
  if [[ -d "$assets/data/robots/widowxai" && -z "${MS_ASSET_DIR:-}" ]]; then
    export MS_ASSET_DIR="$assets"
  fi
  fleet_toolchain::user_bins
  fleet_toolchain::realsense_dds
  [[ -x "$node_bin/node" ]] || return 0
  if [[ "${PATH:-}" != "$node_bin" && "${PATH:-}" != "$node_bin:"* ]]; then
    export PATH="$node_bin${PATH:+:$PATH}"
  fi
}
