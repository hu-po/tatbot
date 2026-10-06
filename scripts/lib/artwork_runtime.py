"""The checkout's artwork runtime: the pinned Node dependencies the shared SVG
adapter (`web/inkmap/tools/artwork.ts`) imports, brought to the checkout's own
package lock before the design stage runs it.

`web/inkmap/node_modules` is gitignored, so a fresh checkout has none: a deploy
fast-forwards the command path (`fleet_install.update_checkout`) and a clone
starts empty, and the first design compile on either failed
`artwork_runtime_unavailable ... Cannot find package '@xmldom/xmldom'` until an
operator ran `npm ci` by hand. `prepare` runs that install once per lock
digest — under a lock outside the tree, so concurrent sessions serialize on
one install — and skips when the receipt matches and the adapter's packages
are present. The receipt lives inside `node_modules`, which `npm ci` replaces
wholesale: a failed or foreign install leaves no receipt to trust, and nothing
here is ever an untracked path `git status` reports.

Stdlib only, Python >= 3.10, like the rest of scripts/lib: the deploy's
install (`fleet_install`) runs it from a bare checkout.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tatbot_digest import sha256_file

PROJECT = 'web/inkmap'
LOCK_FILE = 'package-lock.json'
# What the adapter imports: named so a lock that stops pinning one refuses
# here, with the package, instead of inside the adapter as a module-not-found.
MODULES = ('@xmldom/xmldom', 'ajv')
RECEIPT = '.tatbot-artwork-runtime.sha256'
NPM_CI = ('ci', '--ignore-scripts', '--omit=dev', '--no-audit', '--no-fund')
REASON = 'artwork_runtime_unavailable'
# Where `fleet_toolchain.sh` puts a provisioned Node runtime: an explicitly
# installed one under the user's data directory, ahead of the system runtime.
USER_TOOLCHAIN = '.local/share/tatbot/toolchain/node/bin'


def command() -> str:
    return 'npm ' + ' '.join(NPM_CI) + ' in ' + PROJECT


def find_npm(prefer=()) -> str | None:
    """`npm` the way the CLI and the deploy resolve Node (`fleet_toolchain::use`):
    a release's private runtime (`prefer`), then the user's provisioned one,
    then PATH — which a `scripts/tatbot` process has already extended."""
    for directory in (*prefer, Path.home() / USER_TOOLCHAIN):
        candidate = Path(directory) / 'npm'
        if os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which('npm')


def lock_path(project: Path) -> Path:
    """One lock per project path, beside the user's other caches, never in the tree."""
    cache = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
    name = hashlib.sha256(str(project.resolve()).encode()).hexdigest()[:16]
    return cache / 'tatbot' / 'artwork-runtime' / (name + '.lock')


def _missing(project: Path) -> list[str]:
    return [name for name in MODULES if not (project / 'node_modules' / name / 'package.json').is_file()]


def current(project: Path, expected: str) -> bool:
    """The receipt names this lock and the adapter's packages are installed."""
    receipt = project / 'node_modules' / RECEIPT
    try:
        return receipt.read_text().strip() == expected and not _missing(project)
    except OSError:
        return False


def prepare(repo, npm: str | None = None, log=None) -> bool:
    """Bring `repo/web/inkmap/node_modules` to its package lock. True when an
    install ran, False when the checkout was already current (no npm runs
    then). A missing or failing npm raises ValueError naming the command and
    `artwork_runtime_unavailable`, the refusal the adapter's own failure
    carries — never a traceback."""
    log = sys.stderr if log is None else log
    project = Path(repo) / PROJECT
    try:
        expected = sha256_file(project / LOCK_FILE)
    except OSError as error:
        raise ValueError(f'{REASON}: no {PROJECT}/{LOCK_FILE} to install from: {error}') from error
    if current(project, expected):
        return False
    lock = lock_path(project)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if current(project, expected):
            return False
        executable = npm or find_npm()
        if executable is None:
            raise ValueError(f'{REASON}: npm not found; the shared SVG adapter needs Node.js 22 and `{command()}`')
        print(f"Preparing the checkout's artwork runtime from its package lock ({command()})…", file=log, flush=True)
        try:
            # npm is a node script (`#!/usr/bin/env node`): the toolchain's bin
            # must lead PATH for the subprocess, as it does for scripts/tatbot,
            # or a deploy-time install over a bare ssh exits 127 (2026-09-20).
            env = dict(os.environ)
            # The directory npm was found in, not its symlink target's (`bin/npm`
            # points into `lib/node_modules/npm/bin`, where there is no node).
            env['PATH'] = os.pathsep.join([str(Path(executable).parent), env.get('PATH', '')])
            result = subprocess.run([executable, *NPM_CI], cwd=project, stdout=log, stderr=log, check=False, env=env)
        except OSError as error:
            raise ValueError(f'{REASON}: `{command()}` could not run: {error}') from error
        if result.returncode:
            raise ValueError(f'{REASON}: `{command()}` failed (exit {result.returncode}); '
                             'retry after resolving the logged error')
        missing = _missing(project)
        if missing:
            raise ValueError(f'{REASON}: {PROJECT}/node_modules lacks {", ".join(missing)} after `{command()}`; '
                             f'the package lock no longer pins the adapter\'s dependencies')
        (project / 'node_modules' / RECEIPT).write_text(expected + '\n')
        return True
