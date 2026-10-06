from __future__ import annotations

import hashlib
from pathlib import Path

from tatbot_sim.human_rep.boundary import scan_forbidden_imports, scan_frozen_imports

REPO = Path(__file__).resolve().parents[3]


def test_repository_learned_exact_import_boundary_is_clean():
    assert scan_forbidden_imports(REPO) == []


def test_learned_exact_import_boundary_deliberate_canary(tmp_path):
    source = tmp_path / "policy_writer.py"
    source.write_text(
        "from scripts.lib import pen_path\n",
        encoding="utf-8",
    )
    violations = scan_forbidden_imports(tmp_path)
    assert len(violations) == 1
    violation = violations[0]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert violation.imported == "scripts.lib.pen_path"
    assert violation.code == "learned_execution_boundary_violation"
    assert violation.input_hashes == {"source_file_sha256": digest}
    assert violation.as_dict() == {
        "status": "refused",
        "code": "learned_execution_boundary_violation",
        "path": "policy_writer.py",
        "line": 1,
        "imported": "scripts.lib.pen_path",
        "detail": "forbidden learned-to-exact import scripts.lib.pen_path",
        "input_hashes": {"source_file_sha256": digest},
    }


def test_learned_exact_boundary_catches_namespaced_motion_verbs(tmp_path):
    source = tmp_path / "research_policy.py"
    source.write_text(
        "import scripts.lib.tatbot_cli.verbs.rollout\n"
        "from scripts.lib.tatbot_cli.verbs import vision\n",
        encoding="utf-8",
    )

    violations = scan_forbidden_imports(tmp_path)
    assert [violation.imported for violation in violations] == [
        "scripts.lib.tatbot_cli.verbs.rollout",
        "scripts.lib.tatbot_cli.verbs.vision",
    ]
    assert all(violation.code == "learned_execution_boundary_violation" for violation in violations)


def test_learned_exact_boundary_fails_closed_on_unparseable_source(tmp_path):
    source = tmp_path / "research_policy.py"
    source.write_text("def incomplete(:\n", encoding="utf-8")

    violations = scan_forbidden_imports(tmp_path)
    assert len(violations) == 1
    violation = violations[0]
    assert violation.code == "learned_execution_boundary_violation"
    assert violation.imported == "<parse-error>"
    assert violation.line == 1
    assert violation.input_hashes == {"source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def test_runtime_has_no_frozen_research_dependencies():
    assert scan_frozen_imports(REPO) == []


def test_frozen_boundary_catches_indirect_and_relative_consumers(tmp_path):
    package = tmp_path / "python/tatbot_sim/src/tatbot_sim/human_rep"
    package.mkdir(parents=True)
    (package / "consumer.py").write_text(
        "from tatbot_sim.human_rep import training\n"
        "from .mechanics import fit\n"
        "importlib.import_module('tatbot_sim.human_rep.anatomy')\n"
    )
    assert len(scan_frozen_imports(tmp_path)) == 3
    # Frozen experiments and their historical reproducers remain inspectable.
    (package / "training.py").write_text("from .mechanics import fit\n")
    evidence = tmp_path / "internal/evidence"
    evidence.mkdir(parents=True)
    (evidence / "experiment.py").write_text("from tatbot_sim.human_rep import training\n")
    assert len(scan_frozen_imports(tmp_path)) == 3
