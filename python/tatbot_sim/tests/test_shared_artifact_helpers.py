"""The helpers that replaced ten hand-rolled copies must not change any bytes.

Nine evidence and audit generators each carried an identical `_json_bytes`, ten
an identical `_sha256`, and seven an identical `_git`. Consolidating them is only
safe if the shared versions are byte-for-byte what the copies produced, because
their output is hashed into manifests that are compared across runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from tatbot_contracts.canonical import ContractError, canonical_bytes, indented_bytes
from tatbot_contracts.digest import sha256_file
from tatbot_sim.repo import git_output

VALUES = [
    {"b": 1, "a": [1, 2.5, "x"], "u": "café ✓", "n": None, "t": True},
    {"nested": {"z": {"y": [{"k": 1}]}}, "empty": {}, "el": []},
    {"neg": -0.0, "big": 10**18, "ns_timestamp": 1788908476123456789},
    [], {}, "plain", 42, 0, -1.5,
]


def legacy_json_bytes(value):
    """Verbatim the body the nine copies shared."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"


def legacy_sha256(path):
    """Verbatim the body the ten copies shared."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.parametrize("value", VALUES)
def test_indented_bytes_reproduces_the_copies_it_replaced(value):
    assert indented_bytes(value) == legacy_json_bytes(value)


@pytest.mark.parametrize("bad", [
    {"x": float("nan")},
    {"y": float("inf")},
    {"z": [{"w": float("-inf")}]},   # nested, to prove the walk descends
    [float("nan")],
])
def test_indented_bytes_refuses_what_no_json_parser_would_read_back(bad):
    """The copies emitted a bare `NaN`; an empty sample set was enough."""
    assert b"NaN" in legacy_json_bytes(bad) or b"Infinity" in legacy_json_bytes(bad)
    with pytest.raises(ContractError):
        indented_bytes(bad)


@pytest.mark.parametrize("value", [{"neg": -0.0}, {"big": 10**18}])
def test_the_artifact_writer_is_deliberately_laxer_than_the_digest(value):
    """-0.0 and a nanosecond timestamp break a digest, not a file someone reads.

    Reusing the canonical guard wholesale would have made these an error in
    generators that legitimately produce them.
    """
    assert indented_bytes(value) == legacy_json_bytes(value)
    with pytest.raises(ContractError):
        canonical_bytes(value)


@pytest.mark.parametrize("size", [0, 1, 1024, 1024 * 1024, 1024 * 1024 + 7, 3 * 1024 * 1024])
def test_sha256_file_matches_the_copies_across_chunk_boundaries(size):
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(os.urandom(size))
        name = handle.name
    try:
        assert sha256_file(name) == legacy_sha256(name)
    finally:
        os.unlink(name)


def test_sha256_file_accepts_str_and_path_alike():
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(b"tatbot")
        name = handle.name
    try:
        assert sha256_file(name) == sha256_file(Path(name))
    finally:
        os.unlink(name)


def test_git_output_returns_the_answer_in_the_repo():
    assert git_output("rev-parse", "--verify", "HEAD") != "unknown"


def test_git_output_reports_unknown_rather_than_raising():
    """A generator must still produce its artifact where git cannot answer.

    The audit rejects an unknown revision downstream, where the schema requires
    one; that is a better place to fail than mid-render.
    """
    assert git_output("no-such-subcommand") == "unknown"


def test_no_module_still_carries_a_private_copy_of_these_helpers():
    """The point of the change: the copies are gone, not merely joined.

    Two same-named functions survive on purpose and are not copies of anything:
    human_rep/review.py's `_sha256(value, path)` validates that a string is a
    lowercase hex digest -- it hashes nothing -- and inkmap/sampler.py's
    `_json_bytes` is the compact digest form, not the indented artifact form.
    """
    root = Path(__file__).resolve().parents[3]
    allowed = {
        ("python/tatbot_sim/src/tatbot_sim/human_rep/review.py",
         "def _sha256(value: Any, path: str) -> str:"),
        ("python/tatbot_sim/src/tatbot_sim/inkmap/sampler.py",
         "def _json_bytes(value: object) -> bytes:"),
        # Not a copy either, and unreachable from here: scripts/lib is stdlib-only
        # and cannot import tatbot_sim at all.
        ("scripts/lib/tatbot_cli/status.py",
         "def _git(repo: Path, timeout_s: float) -> dict:"),
    }
    tracked = subprocess.run(
        ["git", "ls-files", "internal/evidence/*.py", "python/tatbot_sim/**/*.py", "scripts/**/*.py"],
        cwd=root, capture_output=True, text=True, check=True).stdout.split()

    offenders = []
    for path in tracked:
        text = (root / path).read_text()
        for line in text.splitlines():
            if not line.startswith(("def _sha256(", "def _json_bytes(", "def _git(")):
                continue
            if (path, line) in allowed:
                continue
            offenders.append(f"{path}: {line}")
    assert offenders == [], offenders
