"""A staged Node runtime, and the per-user tool bins, change only the invoking process."""
import os
import subprocess
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / 'lib/fleet_toolchain.sh'


def invoke(source, path, home):
    """PATH after two `use` calls. HOME is a bare tmp dir so the user bins are the
    test's to create — otherwise the result depends on whether whoever ran the
    suite happens to have ~/.local/bin."""
    return subprocess.check_output(
        ['/bin/bash', '-c', 'source "$1"; fleet_toolchain::use "$2"; fleet_toolchain::use "$2"; printf "%s" "$PATH"',
         'toolchain-test', str(HELPER), str(source)],
        env={**os.environ, 'PATH': path, 'HOME': str(home)}, text=True)


def user_bins(path, home):
    return subprocess.check_output(
        ['/bin/bash', '-c', 'source "$1"; fleet_toolchain::user_bins; fleet_toolchain::user_bins; printf "%s" "$PATH"',
         'toolchain-test', str(HELPER)],
        env={**os.environ, 'PATH': path, 'HOME': str(home)}, text=True)


def test_exact_staged_node_is_preferred_once_without_changing_parent(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    source = tmp_path / 'release/source'
    source.mkdir(parents=True)
    runtime = source.parent / 'toolchain/node/bin'
    runtime.mkdir(parents=True)
    node = runtime / 'node'
    node.write_text('#!/bin/sh\nexit 0\n')
    node.chmod(0o755)
    original = os.environ['PATH']
    assert invoke(source, '/usr/bin:/bin', home) == f'{runtime}:/usr/bin:/bin'
    assert os.environ['PATH'] == original
    (source / '.git').mkdir()
    assert invoke(source, '/usr/bin:/bin', home) == '/usr/bin:/bin'


def test_checkout_or_missing_staged_node_preserves_path(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    assert invoke(tmp_path / 'checkout', '/usr/bin:/bin', home) == '/usr/bin:/bin'
    source = tmp_path / 'release/source'
    source.mkdir(parents=True)
    assert invoke(source, '/usr/bin:/bin', home) == '/usr/bin:/bin'


def test_user_bins_appends_each_existing_bin_once(tmp_path):
    """`uv` lives in ~/.local/bin, which only a login shell puts on PATH: scripts/tatbot
    adds it for every launcher it execs, or they die with `uv: command not found`."""
    home = tmp_path / 'home'
    (home / '.local/bin').mkdir(parents=True)
    assert user_bins('/usr/bin:/bin', home) == f'/usr/bin:/bin:{home}/.local/bin'
    (home / '.cargo/bin').mkdir(parents=True)
    # appended, never prepended: a system or venv tool already on PATH still wins
    assert user_bins('/usr/bin:/bin', home) == f'/usr/bin:/bin:{home}/.cargo/bin:{home}/.local/bin'
    # already there (a login shell got here first): no duplicate
    assert user_bins(f'{home}/.local/bin:/usr/bin', home) == f'{home}/.local/bin:/usr/bin:{home}/.cargo/bin'


def test_shared_tatbot_node_is_used_but_release_runtime_stays_first(tmp_path):
    home = tmp_path / 'home'
    shared = home / '.local/share/tatbot/toolchain/node/bin'
    source = tmp_path / 'release/source'
    source.mkdir(parents=True)
    release = source.parent / 'toolchain/node/bin'
    for directory in (shared, release):
        directory.mkdir(parents=True)
        (directory / 'node').write_text('#!/bin/sh\nexit 0\n')
        (directory / 'node').chmod(0o755)
    original = os.environ['PATH']
    assert user_bins('/usr/bin:/bin', home) == f'{shared}:/usr/bin:/bin'
    assert invoke(source, '/usr/bin:/bin', home) == f'{release}:{shared}:/usr/bin:/bin'
    assert os.environ['PATH'] == original


def pkg_config_path(source, home, inherited=None):
    env = {**os.environ, 'PATH': '/usr/bin:/bin', 'HOME': str(home)}
    env.pop('PKG_CONFIG_PATH', None)
    if inherited is not None:
        env['PKG_CONFIG_PATH'] = inherited
    return subprocess.check_output(
        ['/bin/bash', '-c', 'source "$1"; fleet_toolchain::use "$2"; fleet_toolchain::use "$2"; printf "%s" "${PKG_CONFIG_PATH:-}"',
         'toolchain-test', str(HELPER), str(source)], env=env, text=True)


def test_provisioned_dds_realsense_wins_release_builds_only(tmp_path):
    """The D555 needs a DDS-enabled SDK; where a node provisions one, a release
    build links it ahead of the distribution package, and a checkout is untouched."""
    home = tmp_path / 'home'
    home.mkdir()
    source = tmp_path / 'release/source'
    source.mkdir(parents=True)
    assert pkg_config_path(source, home) == ''
    pc_dir = home / '.local/share/tatbot/toolchain/librealsense-dds/lib/pkgconfig'
    pc_dir.mkdir(parents=True)
    (pc_dir / 'realsense2.pc').write_text('Name: realsense2\nVersion: 2.58.4\n')
    assert pkg_config_path(source, home) == str(pc_dir)
    assert pkg_config_path(source, home, '/opt/other') == f'{pc_dir}:/opt/other'
    assert pkg_config_path(tmp_path / 'checkout', home) == ''
