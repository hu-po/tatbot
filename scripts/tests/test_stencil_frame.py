"""Seed reproducibility, printable geometry, and offline CLI integration."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

import stencil_frame  # noqa: E402
import stencil_reference  # noqa: E402


def options(output, *extra):
    parser = argparse.ArgumentParser()
    stencil_frame.add_arguments(parser)
    ns = parser.parse_args(["--output", str(output), *extra])
    stencil_frame.validate(ns)
    ns.output = output
    return ns


def test_seed_reproduces_vector_and_pixels_with_distinct_alternatives(tmp_path):
    pytest.importorskip("PIL")
    outputs = []
    for folder, seed in (("first", "tatbot-42"), ("repeat", "tatbot-42"),
                         ("second", "tatbot-43"), ("third", "tatbot-44")):
        ns = options(tmp_path / folder, "--seed", seed,
                     "--instance-id", "0123456789abcdef01234567")
        stencil_frame.render(ns, stencil_frame.build(ns))
        outputs.append({p.name: p.read_bytes() for p in ns.output.iterdir()})
    assert outputs[0] == outputs[1]
    for name in ("stencil.png", "stencil.svg"):
        assert len({item[name] for item in (outputs[0], *outputs[2:])}) == 3
    # The floral artwork hash stays stable while the co-printed mark varies.
    assert json.loads(outputs[0]["settings.json"])["artwork_svg_sha256"] == (
        "b75816e8a0e1a873d35893bdb59785a9acc137e303d021d975d9fd308ed67e67"
    )


def test_checked_in_frames_have_blank_centers_margins_and_valid_hashes():
    image = pytest.importorskip("PIL.Image")
    samples = REPO / "docs/assets/stencil-frames"
    assert sorted(p.name for p in samples.iterdir()) == ["tatbot-42", "tatbot-43", "tatbot-44"]
    for folder in samples.iterdir():
        settings = json.loads((folder / "settings.json").read_text())
        assert settings["seed"] == folder.name
        assert settings["clear_center_mm"] == [62, 112]
        for name, digest in settings["files"].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
        with image.open(folder / "stencil.png") as im:
            assert im.mode == "1" and im.size == (1181, 1772)
            assert im.info["dpi"][0] == pytest.approx(300, abs=0.01)
            scale = 300 / 25.4
            assert im.crop(tuple(round(n * scale) for n in (19, 19, 81, 131))).getextrema() == (255, 255)
            x0, y0, x1, y1 = (round(n * scale) for n in (5, 5, 95, 145))
            for box in ((0, 0, im.width, y0), (0, y1, im.width, im.height),
                        (0, 0, x0, im.height), (x1, 0, im.width, im.height)):
                assert im.crop(box).getextrema() == (255, 255)
            assert 0.08 < im.histogram()[0] / (im.width * im.height) < 0.25
        vector = ET.parse(folder / "stencil.svg").getroot()
        assert vector.attrib["width"] == "100mm"
        assert vector.attrib["height"] == "150mm"
        assert len(vector.findall(".//{*}clipPath/{*}rect")) == 4


def test_tracking_manifest_identity_dimensions_and_hash_validation(tmp_path):
    pytest.importorskip("PIL")
    ns = options(tmp_path/"stencil")
    settings = stencil_frame.render(ns, stencil_frame.build(ns))
    manifest, image_path = stencil_reference.load(ns.output/"tracking.json")
    assert manifest["page_mm"] == [100, 150]
    assert manifest["coordinates"]["uv_to_print_mm"] == [[100, 0, 0], [0, 150, 0], [0, 0, 1]]
    assert manifest["physical_instance_encoded"]
    assert manifest["physical_instance_id"] == settings["physical_instance_id"]
    assert manifest["instance_mark"]["scheme"] == "tatbot.stencil-instance-grid/1"
    assert manifest["generator"]["svg_sha256"] == settings["artwork_svg_sha256"]
    assert manifest["image"]["sha256"] == settings["files"]["stencil.png"]
    changed = dict(manifest, page_mm=[1, 2])
    (ns.output/"tracking.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="metadata hash"):
        stencil_reference.load(ns.output/"tracking.json")
    (ns.output/"tracking.json").write_text(json.dumps(manifest))
    image_path.write_bytes(b"changed artwork")
    with pytest.raises(ValueError, match="hash"):
        stencil_reference.load(ns.output/"tracking.json")


def test_same_pattern_gets_fresh_print_instance_and_legacy_stays_unqualified(tmp_path):
    pytest.importorskip("PIL")
    marked = []
    for name, instance_id in (("a", "0123456789abcdef01234567"),
                              ("b", "fedcba9876543210fedcba98")):
        ns = options(tmp_path/name, "--instance-id", instance_id)
        stencil_frame.render(ns, stencil_frame.build(ns))
        marked.append(stencil_reference.load(ns.output/"tracking.json")[0])
    assert marked[0]['pattern_id'] == marked[1]['pattern_id']
    assert marked[0]['reference_id'] != marked[1]['reference_id']
    assert marked[0]['image']['sha256'] != marked[1]['image']['sha256']
    legacy = stencil_reference.load(REPO/'docs/assets/stencil-frames/tatbot-42/tracking.json')[0]
    assert legacy['pattern_id'] == marked[0]['pattern_id']
    assert legacy['physical_instance_encoded'] is False
    assert 'physical_instance_id' not in legacy
    ns = options(tmp_path/'fresh')
    stencil_frame.render(ns, stencil_frame.build(ns))
    fresh = stencil_reference.load(ns.output/'tracking.json')[0]
    assert fresh['physical_instance_encoded'] and fresh['physical_instance_id'] not in {
        row['physical_instance_id'] for row in marked}


def test_export_refuses_rehashed_svg_with_a_different_print_code(tmp_path):
    pytest.importorskip('PIL')
    ns = options(tmp_path/'marked', '--instance-id', '0123456789abcdef01234567')
    settings = stencil_frame.render(ns, stencil_frame.build(ns))
    svg_path = ns.output/'stencil.svg'
    prefix, marker, tail = svg_path.read_text().partition('<g id="tatbot-instance-mark">')
    svg_path.write_text(prefix + marker + tail.replace('fill="black"', 'fill="white"', 1))
    settings['files']['stencil.svg'] = hashlib.sha256(svg_path.read_bytes()).hexdigest()
    (ns.output/'settings.json').write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='printed SVG instance mark'):
        stencil_reference.export(ns.output/'settings.json')


def test_install_places_the_manifest_last_where_the_observer_reads_references(tmp_path, monkeypatch):
    """`stencil reference --install` puts the pair under this node's log root
    for the fleet observer's mtime scan: image first, manifest last, a
    same-content install untouched, a changed one withdrawn before replaced."""
    pytest.importorskip("PIL")
    ns = options(tmp_path/"stencil")
    settings = stencil_frame.render(ns, stencil_frame.build(ns))
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path/"logs"))
    reference = json.loads((ns.output/"tracking.json").read_text())
    installed = stencil_reference.install(ns.output/"tracking.json")
    assert installed == tmp_path/"logs/stencils/references"/reference["pattern_id"]
    manifest, image = stencil_reference.load(installed/"tracking.json")
    assert manifest == reference and image == installed/"stencil.png"
    assert sorted(path.name for path in installed.iterdir()) == ["settings.json", "stencil.png", "tracking.json"]
    stamp = (installed/"tracking.json").stat().st_mtime_ns
    assert stencil_reference.install(ns.output/"tracking.json") == installed
    assert (installed/"tracking.json").stat().st_mtime_ns == stamp, "an unchanged install is not a new bank"
    # Re-rendered artwork under the same pattern replaces the pair; the
    # manifest is the last file to change and describes the new image.
    (ns.output/"stencil.png").write_bytes((ns.output/"stencil.png").read_bytes() + b"\n")
    settings["files"]["stencil.png"] = hashlib.sha256((ns.output/"stencil.png").read_bytes()).hexdigest()
    (ns.output/"settings.json").write_text(json.dumps(settings))
    changed = stencil_reference.export(ns.output/"settings.json")
    assert changed["pattern_id"] == reference["pattern_id"] and changed["reference_id"] != reference["reference_id"]
    assert stencil_reference.install(ns.output/"tracking.json") == installed
    assert stencil_reference.load(installed/"tracking.json")[0] == changed
    assert (installed/"tracking.json").stat().st_mtime_ns >= (installed/"stencil.png").stat().st_mtime_ns
    # The script and the verb carry the flag; a plain export installs nothing.
    result = subprocess.run([sys.executable, str(REPO/"scripts/lib/stencil_reference.py"),
                             "--settings", str(ns.output/"settings.json"), "--install"],
                            capture_output=True, text=True, timeout=60, env={**os.environ, "TATBOT_LOG_ROOT": str(tmp_path/"other")})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["installed"] == str(tmp_path/"other/stencils/references"/reference["pattern_id"])
    plan = subprocess.run([sys.executable, "-S", str(REPO/"scripts/lib/tatbot_cli"), "--json", "--dry-run",
                           "vision", "stencil", "reference", "--settings", str(ns.output/"settings.json"), "--install"],
                          capture_output=True, text=True, timeout=30)
    assert plan.returncode == 0, plan.stderr
    assert json.loads(plan.stdout)["argv"][-1] == "--install"


@pytest.mark.parametrize("extra", [
    ["--frame-mm", "80"], ["--frame-mm", "nan"], ["--margin-mm", "-1"],
    ["--gap-rate", "nan"], ["--fill-rate", "1.1"], ["--stroke-mm", "0"],
    ["--width-mm", "inf"], ["--spacing-mm", "3"], ["--dpi", "0"],
    ["--frame-mm", "3"], ["--instance-id", "not-hex"],
    ["--unmarked", "--instance-id", "0123456789abcdef01234567"],
    ["--width-mm", "500", "--height-mm", "500", "--dpi", "1200"],
])
def test_invalid_settings_create_no_files(tmp_path, extra):
    target = tmp_path / "untouched"
    with pytest.raises(ValueError):
        options(target, *extra)
    assert not target.exists()


def test_bare_clone_dry_run_forwards_options_without_rendering(tmp_path):
    output = tmp_path / "no files here"
    command = [sys.executable, "-S", str(REPO / "scripts/lib/tatbot_cli"),
               "--json", "--dry-run", "vision", "stencil", "generate", "--design", "flower-of-life",
               "--seed", "flowers 43", "--frame-mm", "10", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["argv"][:9] == ["uv", "run", "--no-project", "--python", "3.12",
                                "--with", "numpy==2.5.3", "--with", "Pillow==12.1.1"]
    assert plan["argv"][plan["argv"].index("--design") + 1] == "flower-of-life"
    assert plan["argv"][plan["argv"].index("--seed") + 1] == "flowers 43"
    assert plan["argv"][plan["argv"].index("--frame-mm") + 1] == "10.0"
    assert plan["argv"][plan["argv"].index("--output") + 1] == str(output)
    assert not ({"sensor_read", "autonomous_motion", "remote_exec"} & set(plan["effects"]))
    assert {"environment_setup", "network", "write_files"} <= set(plan["effects"])
    assert not output.exists()
    invalid = subprocess.run([*command, "--frame-mm", "100"], capture_output=True,
                             text=True, timeout=30)
    assert invalid.returncode == 2
    assert not output.exists()
