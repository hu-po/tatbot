"""`vision stencil generate`: the coded default design, the legacy design, and the docs examples."""

import hashlib
import json

import pytest
import stencil_coded
import stencil_frame
import stencil_generate
import stencil_reference
from cli_runner import REPO, tatbot

EXAMPLES = REPO/"docs/assets/stencil-coded"
# The first coded pages printed for a test: seed -> pattern ID prefix.
PRINTED = {"1": "stencil-db77aa43fb37", "2": "stencil-e7c8cf3cae20", "3": "stencil-9a7d03d10ffc"}
INSTANCE = "0123456789abcdef01234567"


def generate(capsys, *argv):
    assert stencil_generate.main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def plan(tmp_path, *argv):
    return tatbot("--json", "--dry-run", "vision", "stencil", "generate", *argv,
                  "--output", str(tmp_path/"out"), isolated=True)


def test_the_coded_setting_keys_are_the_generators():
    assert set(stencil_coded.DEFAULTS) == stencil_generate.CODED_SETTINGS


def test_checked_in_coded_examples_load_with_their_code():
    assert sorted(path.name for path in EXAMPLES.iterdir()) == sorted(PRINTED)
    for seed, prefix in PRINTED.items():
        reference, _ = stencil_reference.load(EXAMPLES/seed/"tracking.json")
        assert reference["pattern_id"].startswith(prefix) and reference["seed"] == seed
        assert reference["physical_instance_id"] == reference["coded"]["print_id"] == stencil_coded.print_id_for(seed)
        settings = json.loads((EXAMPLES/seed/"settings.json").read_text())
        for name, digest in settings["files"].items():
            assert hashlib.sha256((EXAMPLES/seed/name).read_bytes()).hexdigest() == digest
        # Knots reach under 0.9 mm into the nominal 62 x 112 mm clear centre (19..81 x 19..131 mm).
        assert settings["clear_center_mm"] == [62, 112]
        left, top, right, bottom = settings["border_inner_mm"]
        assert max(left-19, top-19, 81-right, 131-bottom) < .9


def test_seed_1_regenerates_its_checked_in_example(tmp_path, capsys):
    result = generate(capsys, "--seed", "1", "--output", str(tmp_path), "--no-sheets")
    example = EXAMPLES/"1"
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(path.name for path in example.iterdir())
    assert (tmp_path/"coded.json").read_bytes() == (example/"coded.json").read_bytes()
    assert hashlib.sha256((tmp_path/"stencil.svg").read_bytes()).hexdigest() == \
        json.loads((example/"settings.json").read_text())["artwork_svg_sha256"]
    assert result["pattern_id"] == json.loads((example/"tracking.json").read_text())["pattern_id"]
    assert result["design"] == "coded-flower-of-life" and result["seed"] == "1" and result["sheets"] is None
    assert result["print_id"] == stencil_coded.print_id_for("1") == result["reference"]["coded"]["print_id"]


def test_an_omitted_seed_mints_a_distinct_print_with_sheets(tmp_path, capsys):
    runs = [generate(capsys, "--output", str(tmp_path/name)) for name in "ab"]
    assert runs[0]["seed"] != runs[1]["seed"] and runs[0]["pattern_id"] != runs[1]["pattern_id"]
    assert runs[0]["print_id"] == stencil_coded.print_id_for(runs[0]["seed"])
    sheets = runs[0]["sheets"]
    assert sheets["pattern_id"] == runs[0]["pattern_id"] and sheets["label"].startswith(
        f"pattern {runs[0]['pattern_id'][8:20]}")
    assert sorted(sheets["files"]) == ["paper-a4.pdf", "paper-letter.pdf", "sheet.svg", "stencil-app.png"]
    assert (tmp_path/"a/print/sheet.json").is_file() and (tmp_path/"a/coded.json").is_file()


def test_the_legacy_design_keeps_its_options_and_artwork(tmp_path, capsys):
    result = generate(capsys, "--design", "flower-of-life", "--unmarked", "--output", str(tmp_path/"42"),
                      "--fit-area", "203.2x269.4")
    legacy = REPO/"docs/assets/stencil-frames/tatbot-42"
    assert (tmp_path/"42/stencil.svg").read_bytes() == (legacy/"stencil.svg").read_bytes()
    assert result["seed"] == "tatbot-42" and result["print_id"] is None
    assert result["pattern_id"] == json.loads((legacy/"tracking.json").read_text())["pattern_id"]
    assert "paper-fit.pdf" in result["sheets"]["files"] and not (tmp_path/"42/coded.json").exists()
    marked = generate(capsys, "--design", "flower-of-life", "--instance-id", INSTANCE, "--frame-mm", "10",
                      "--output", str(tmp_path/"marked"), "--no-sheets")
    assert marked["print_id"] == INSTANCE and marked["sheets"] is None
    assert json.loads((tmp_path/"marked/settings.json").read_text())["frame_mm"] == 10


@pytest.mark.parametrize("argv", [["--set", "wobble=1"], ["--set", "knot_mm"], ["--frame-mm", "10"],
                                  ["--design", "flower-of-life", "--set", "knot_mm=2"]])
def test_the_backend_refuses_before_writing(tmp_path, argv):
    with pytest.raises(SystemExit) as refused:
        stencil_generate.main([*argv, "--output", str(tmp_path/"out")])
    assert refused.value.code == 2 and not (tmp_path/"out").exists()


def test_cli_plans_the_coded_design_by_default(tmp_path):
    result = plan(tmp_path, "--seed", "7", "--set", "knot-mm=2.4")
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["argv"]
    assert argv[:9] == ["uv", "run", "--no-project", "--python", "3.12", "--with", "numpy==2.5.3",
                        "--with", stencil_frame.PILLOW_REQUIREMENT]
    assert argv[argv.index("python")+1].endswith("scripts/lib/stencil_generate.py")
    assert argv[argv.index("--design")+1] == "coded-flower-of-life" and argv[argv.index("--seed")+1] == "7"
    assert argv[argv.index("--set")+1] == "knot-mm=2.4"
    assert not {"--frame-mm", "--dpi", "--unmarked", "--no-sheets"} & set(argv)
    assert not (tmp_path/"out").exists()
    unseeded = json.loads(plan(tmp_path).stdout)["argv"]
    assert "--seed" not in unseeded, "the backend mints a fresh seed"


@pytest.mark.parametrize("argv, message", [
    (["--frame-mm", "10"], "--frame-mm: flower-of-life only"),
    (["--unmarked", "--dpi", "600"], "the coded design takes --set key=value"),
    (["--set", "frame-mm=10", "--set", "wobble=1"], "unknown coded setting 'wobble'"),
    (["--design", "flower-of-life", "--set", "knot_mm=2"], "--set is for the coded design"),
    (["--design", "flower-of-life", "--frame-mm", "80"], "frame must leave a nonempty center"),
    (["--fit-area", "203.2x269.4", "--no-sheets"], "drop --no-sheets"),
])
def test_cli_refuses_what_the_design_does_not_take(tmp_path, argv, message):
    result = plan(tmp_path, *argv)
    assert result.returncode == 2 and message in result.stderr, result.stderr


def test_cli_forwards_the_legacy_options(tmp_path):
    result = plan(tmp_path, "--design", "flower-of-life", "--unmarked", "--frame-mm", "10", "--no-sheets")
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["argv"]
    assert "--unmarked" in argv and "--no-sheets" in argv and "--seed" not in argv
    assert argv[argv.index("--frame-mm")+1] == "10.0"
    marked = json.loads(plan(tmp_path, "--design", "flower-of-life", "--instance-id", INSTANCE).stdout)["argv"]
    assert marked[marked.index("--instance-id")+1] == INSTANCE and "--unmarked" not in marked
