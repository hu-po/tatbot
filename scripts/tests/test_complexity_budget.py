"""The complexity ratchet only ever lets the recorded numbers go down."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

import complexity_budget as cb  # noqa: E402


def run(monkeypatch, tmp_path, baseline_rows, measured):
    """Drive main() against a synthetic baseline and a synthetic measurement."""
    baseline = tmp_path / "complexity-debt.tsv"
    baseline.write_text(cb.HEADER + "".join(f"{p}\t{n}\t{c}\n" for (p, n), c in baseline_rows.items()))
    monkeypatch.setattr(cb, "BASELINE", baseline)
    monkeypatch.setattr(cb, "tracked_python", lambda: ["anything.py"])
    monkeypatch.setattr(cb, "measure", lambda paths: measured)
    return cb.main([])


def test_unchanged_tree_passes(monkeypatch, tmp_path, capsys):
    rows = {("a.py", "f"): 20}
    assert run(monkeypatch, tmp_path, rows, dict(rows)) == 0
    assert "complexity held" in capsys.readouterr().out


def test_a_function_getting_worse_fails(monkeypatch, tmp_path, capsys):
    assert run(monkeypatch, tmp_path, {("a.py", "f"): 20}, {("a.py", "f"): 21}) == 1
    assert "WORSE  a.py::f 20 -> 21" in capsys.readouterr().out


def test_a_newly_complex_function_fails(monkeypatch, tmp_path, capsys):
    assert run(monkeypatch, tmp_path, {}, {("a.py", "f"): 11}) == 1
    assert "NEW    a.py::f 11 > 10" in capsys.readouterr().out


def test_improvement_is_reported_but_does_not_fail(monkeypatch, tmp_path, capsys):
    """Improvement is the outcome this gate wants, so it must not go red for it.

    At this repository's commit rate a gate that failed whenever a function got
    simpler would be red almost always, and a gate expected to be red stops
    being read. A stale-high baseline still refuses regressions, which is the
    ratchet.
    """
    assert run(monkeypatch, tmp_path, {("a.py", "f"): 20}, {("a.py", "f"): 12}) == 0
    out = capsys.readouterr().out
    assert "BETTER a.py::f 20 -> 12" in out
    assert "--update" in out


def test_a_removed_function_is_reported_but_does_not_fail(monkeypatch, tmp_path, capsys):
    assert run(monkeypatch, tmp_path, {("a.py", "f"): 20}, {}) == 0
    assert "GONE   a.py::f (was 20)" in capsys.readouterr().out


def test_a_regression_still_fails_alongside_an_improvement(monkeypatch, tmp_path, capsys):
    """One file getting better must never mask another getting worse."""
    assert run(monkeypatch, tmp_path, {("a.py", "f"): 20, ("b.py", "g"): 15},
               {("a.py", "f"): 12, ("b.py", "g"): 16}) == 1
    out = capsys.readouterr().out
    assert "WORSE  b.py::g 15 -> 16" in out and "BETTER a.py::f 20 -> 12" in out


def test_only_the_files_in_scope_are_judged(monkeypatch, tmp_path, capsys):
    """The pre-commit path: a neighbour's dirty file must not fail my commit."""
    baseline = tmp_path / "complexity-debt.tsv"
    baseline.write_text(cb.HEADER + "mine.py\tf\t20\ntheirs.py\tg\t15\n")
    monkeypatch.setattr(cb, "BASELINE", baseline)
    monkeypatch.setattr(cb, "measure", lambda paths: {("mine.py", "f"): 20})
    assert cb.main(["mine.py"]) == 0
    assert "GONE" not in capsys.readouterr().out  # theirs.py is unseen, not gone


def test_update_rewrites_the_baseline(monkeypatch, tmp_path):
    baseline = tmp_path / "complexity-debt.tsv"
    monkeypatch.setattr(cb, "BASELINE", baseline)
    monkeypatch.setattr(cb, "tracked_python", lambda: ["anything.py"])
    monkeypatch.setattr(cb, "measure", lambda paths: {("b.py", "g"): 30, ("a.py", "f"): 12})
    assert cb.main(["--update"]) == 0
    rows = [line for line in baseline.read_text().splitlines() if not line.startswith("#")]
    assert rows == ["a.py\tf\t12", "b.py\tg\t30"]  # sorted, so diffs stay readable


def test_update_refuses_to_raise_or_add_an_entry(monkeypatch, tmp_path, capsys):
    """A bulk refresh once raised twenty entries in one commit; --update is the
    tightening step and must not be the escape hatch."""
    baseline = tmp_path / "complexity-debt.tsv"
    baseline.write_text(cb.HEADER + "a.py\tf\t20\nc.py\th\t15\n")
    monkeypatch.setattr(cb, "BASELINE", baseline)
    monkeypatch.setattr(cb, "tracked_python", lambda: ["anything.py"])
    monkeypatch.setattr(cb, "measure", lambda paths: {("a.py", "f"): 25, ("b.py", "g"): 12, ("c.py", "h"): 11})
    assert cb.main(["--update"]) == 1
    out = capsys.readouterr().out
    assert "WORSE  a.py::f 20 -> 25" in out
    assert "NEW    b.py::g 12 > 10" in out
    assert "--accept a.py::f" in out and "--accept b.py::g" in out
    assert baseline.read_text() == cb.HEADER + "a.py\tf\t20\nc.py\th\t15\n"  # untouched


def test_update_records_a_raise_only_when_named(monkeypatch, tmp_path, capsys):
    baseline = tmp_path / "complexity-debt.tsv"
    baseline.write_text(cb.HEADER + "a.py\tf\t20\n")
    monkeypatch.setattr(cb, "BASELINE", baseline)
    monkeypatch.setattr(cb, "tracked_python", lambda: ["anything.py"])
    monkeypatch.setattr(cb, "measure", lambda paths: {("a.py", "f"): 25, ("b.py", "g"): 12})
    assert cb.main(["--update", "--accept", "a.py::f"]) == 1  # b.py::g still refused
    assert cb.main(["--update", "--accept", "a.py::f", "--accept", "b.py::g"]) == 0
    assert "2 accepted by name" in capsys.readouterr().out
    rows = [line for line in baseline.read_text().splitlines() if not line.startswith("#")]
    assert rows == ["a.py\tf\t25", "b.py\tg\t12"]


def test_update_lowers_and_removes_without_ceremony(monkeypatch, tmp_path):
    baseline = tmp_path / "complexity-debt.tsv"
    baseline.write_text(cb.HEADER + "a.py\tf\t20\nb.py\tg\t30\n")
    monkeypatch.setattr(cb, "BASELINE", baseline)
    monkeypatch.setattr(cb, "tracked_python", lambda: ["anything.py"])
    monkeypatch.setattr(cb, "measure", lambda paths: {("a.py", "f"): 12})
    assert cb.main(["--update"]) == 0
    rows = [line for line in baseline.read_text().splitlines() if not line.startswith("#")]
    assert rows == ["a.py\tf\t12"]


def test_accept_must_name_a_function_over_the_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(cb, "BASELINE", tmp_path / "complexity-debt.tsv")
    monkeypatch.setattr(cb, "tracked_python", lambda: ["anything.py"])
    monkeypatch.setattr(cb, "measure", lambda paths: {("a.py", "f"): 12})
    with pytest.raises(SystemExit):
        cb.main(["--update", "--accept", "nope.py::f"])


def test_the_checked_in_baseline_matches_the_committed_tree():
    """The real ratchet over the real repo, minus whatever is uncommitted.

    Scoped to files that match HEAD on purpose. This is a shared checkout: a
    neighbour mid-edit would otherwise fail this test for a regression that is
    not committed, not this test's subject, and not anyone's to fix yet.
    """
    dirty = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", "*.py"],
                           cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split()
    clean = [path for path in tracked
             if path not in set(dirty) and not path.startswith(cb.EXCLUDED_PREFIX)]
    assert clean, "no committed Python to check"

    proc = subprocess.run([sys.executable, str(REPO / "scripts/complexity_budget.py"), *clean],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_an_unparsable_ruff_message_fails_loudly(monkeypatch):
    """A ruff message-format change must not read as 'no complex functions'."""
    monkeypatch.setattr(cb.shutil, "which", lambda name: "/usr/bin/ruff")

    class Proc:
        returncode = 1
        stdout = '[{"filename": "/x/a.py", "message": "something else entirely"}]'
        stderr = ""

    monkeypatch.setattr(cb.subprocess, "run", lambda *a, **k: Proc())
    try:
        cb.measure(["a.py"])
    except SystemExit as exc:
        assert "unparsed C901 message" in str(exc)
    else:
        raise AssertionError("a changed message format passed silently")


def test_the_baseline_never_names_a_private_path():
    """internal/** is private-only, and its paths carry a node name.

    Listing one in a public config file is what `scripts/check export` refuses;
    the ratchet skips the tree rather than earning an export-debt exception.
    """
    assert not any(path.startswith(cb.EXCLUDED_PREFIX) for path in cb.tracked_python())
    baseline = (REPO / "config/complexity-debt.tsv").read_text().splitlines()
    named = [line for line in baseline
             if line and not line.startswith("#") and line.startswith(cb.EXCLUDED_PREFIX)]
    assert named == [], named
