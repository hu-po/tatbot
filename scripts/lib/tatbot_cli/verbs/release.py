"""release — publish the public export of main to hu-po/tatbot (internal/export/README.md)."""

from __future__ import annotations

from tatbot_cli.registry import OFFLINE, REMOTE, verb
from tatbot_cli.verbs._common import py

BACKEND = "scripts/export_scan.py"
DOC = "internal/export/README.md"


def _ref(p):
    p.add_argument("--ref", default="origin/main", help="the committed revision to export (default origin/main)")


def _preview_args(p):
    _ref(p)
    p.add_argument("--files", action="store_true", help="list every added, changed and removed path")


def _publish_args(p):
    _ref(p)
    p.add_argument("version", nargs="?", help="tag the release, e.g. 0.10.0 (pushed as v0.10.0); omit for an untagged export")
    p.add_argument("--yes", action="store_true", help="publish without the y/N prompt")


@verb(effects=("read_files", "network", "start_process"), noun="release", verb="preview", tier=OFFLINE,
      output="text", args=_preview_args, wraps=(BACKEND,), example=(), doc=DOC,
      summary="build and scan the export of origin/main and list what it would change on the public repository",
      invariants=("Builds from `git archive` of a committed revision, never a working tree.",
                  "Refuses (exit 3) on a disclosure error or a public edit the export would overwrite."))
def release_preview(ctx, ns, rest):
    return py(ctx, BACKEND, "release", "--ref", ns.ref, *(["--files"] if ns.files else []), *rest)


@verb(effects=("read_files", "network", "start_process", "remote_write"), noun="release", verb="publish", tier=REMOTE,
      output="text", args=_publish_args, wraps=(BACKEND,), example=("0.10.0",), doc=DOC,
      summary="the preview, then one export commit (and tag) pushed to the public repository after a y/N prompt",
      invariants=("Whoever answers the prompt approves the release; there is no other approval record.",
                  "Never force-pushes: main and the tag are pushed atomically, and an existing tag refuses.",
                  "Tests run in the public repository's CI after the push, not before it."))
def release_publish(ctx, ns, rest):
    return py(ctx, BACKEND, "release", "--publish", "--ref", ns.ref,
              *(["--tag", ns.version] if ns.version else []), *(["--yes"] if ns.yes else []), *rest,
              notes=["target: main of the public repository hu-po/tatbot"])
