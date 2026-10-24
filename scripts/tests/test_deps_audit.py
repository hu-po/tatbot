"""scripts/deps_audit.py offline: what it reads from each lockfile and pin, and how it judges OSV records."""
import datetime
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("deps_audit", REPO / "scripts" / "deps_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_it_reads_registry_versions_from_every_lock_and_exact_pins_only(tmp_path):
    (tmp_path / "rust").mkdir()
    (tmp_path / "rust/Cargo.lock").write_text(
        'version = 4\n[[package]]\nname = "tokio"\nversion = "1.53.3"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        '[[package]]\nname = "tatbot-bus"\nversion = "0.1.0"\n')
    (tmp_path / "python/lerobot_robot_tatbot").mkdir(parents=True)
    (tmp_path / "python/lerobot_robot_tatbot/uv.lock").write_text(
        'version = 1\n[[package]]\nname = "Pillow"\nversion = "12.3.0"\nsource = { registry = "https://pypi.org/simple" }\n'
        '[[package]]\nname = "tatbot-contracts"\nversion = "0.1.0"\nsource = { editable = "../tatbot_contracts" }\n')
    (tmp_path / "web/inkmap").mkdir(parents=True)
    (tmp_path / "web/inkmap/package-lock.json").write_text(json.dumps({"packages": {
        "": {"name": "inkmap"}, "node_modules/vite": {"version": "7.3.7"},
        "node_modules/a/node_modules/fast-uri": {"version": "3.1.9"}, "node_modules/self": {"link": True}}}))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/requirements.txt").write_text(
        "# a comment\ncertifi==2026.10.19 \\\n    --hash=sha256:abc\nhuggingface_hub[cli]==1.28.0 ; python_version >= '3.10'\n"
        "sphinx\nnumpy>=1.26\n")
    assert audit.packages(tmp_path) == {
        ("crates.io", "tokio", "1.53.3"): {"rust/Cargo.lock"},
        ("PyPI", "pillow", "12.3.0"): {"python/lerobot_robot_tatbot/uv.lock"},
        ("npm", "vite", "7.3.7"): {"web/inkmap/package-lock.json"},
        ("npm", "fast-uri", "3.1.9"): {"web/inkmap/package-lock.json"},
        ("PyPI", "certifi", "2026.10.19"): {"docs/requirements.txt"},
        ("PyPI", "huggingface-hub", "1.28.0"): {"docs/requirements.txt"},
    }


def test_an_allowance_holds_for_its_package_until_its_date_and_unmaintained_is_only_a_notice(tmp_path):
    allow = tmp_path / "allow.tsv"
    allow.write_text("# id\tpackage\treview_by\treason\nGHSA-aaaa\tpillow\t2026-12-01\tcapped upstream\n")
    key = ("PyPI", "pillow", "12.3.0")
    found = {key: {"python/x/uv.lock"}, ("crates.io", "paste", "1.0.15"): {"rust/Cargo.lock"}}
    records = {
        key: [{"id": "GHSA-aaaa", "summary": "a decoder bug", "database_specific": {"severity": "HIGH"}},
              {"id": "PYSEC-1", "aliases": ["GHSA-aaaa"], "summary": "the same bug"},
              {"id": "GHSA-bbbb", "summary": "a new one", "database_specific": {"severity": "CRITICAL"}}],
        ("crates.io", "paste", "1.0.15"): [
            {"id": "RUSTSEC-2024-0436", "summary": "unmaintained",
             "affected": [{"database_specific": {"informational": "unmaintained"}}]}],
    }
    allowed = audit.allowances(allow)
    verdicts = lambda day: [(r["id"], r["verdict"]) for r in audit.judge(found, records, allowed, day)]  # noqa: E731
    assert verdicts(datetime.date(2026, 11, 1)) == [
        ("GHSA-aaaa", "accepted"), ("PYSEC-1", "accepted"), ("GHSA-bbbb", "finding"), ("RUSTSEC-2024-0436", "notice")]
    assert ("GHSA-aaaa", "expired") in verdicts(datetime.date(2026, 12, 2))


def test_the_committed_allowlist_parses_and_every_row_says_why():
    rows = audit.allowances()
    assert rows and all(row["reason"].strip() and row["package"] for row in rows)
