"""python/tatbot_art shade: whip and pendulum shading from a tone image, with deposition along each stroke, laid by
each cartridge's own mark: a round liner's line, a magnum's swept row."""
import json
import math

import numpy as np
import pytest
from tatbot_art import cartridge, cli
from tatbot_art.color import to_8bit
from tatbot_art.png import write_png
from tatbot_art.render import render_program
from tatbot_art.shade import (
    Shading,
    descent_angle,
    generate,
    limit_rise,
    pendulums,
    preview_program,
    simplify,
    start_in,
    tone_from_image,
)

PX_PER_MM = 10.0


def _block():
    """A 30 x 15 mm block on a 40 x 20 mm page, full tone at its left, fading to none at its right."""
    tone = np.zeros((200, 400))
    tone[25:175, 50:350] = np.linspace(1.0, 0.0, 300)[None]
    return tone


def _strokes(technique, tone=None, tool="lutin-3rl-bugpin"):
    tone = _block() if tone is None else tone
    strokes, info = generate(tone, PX_PER_MM, Shading(technique=technique), cartridge.load(tool))
    program = preview_program(strokes, (tone.shape[1] / PX_PER_MM, tone.shape[0] / PX_PER_MM), "black", info["mark"])
    return strokes, info, program


def _coverage(program):
    return render_program(program, {"black": {"rgb": [0, 0, 0]}}, "paper", PX_PER_MM, fade=False).coverage["black"]


def test_each_cartridge_lays_its_own_mark():
    liner, magnum = cartridge.load("lutin-3rl-bugpin"), cartridge.load("lutin-9m-ex")
    widths = [liner.swath_mm(h) for h in range(0, 180, 10)]
    assert liner.kind == "round" and 0.5 < min(widths) < max(widths) < 0.6   # about the same whichever way
    assert cartridge.load("lutin-11rl-wormhole").swath_mm(0) == pytest.approx(1.16, abs=0.05)
    assert magnum.kind == "magnum" and magnum.swath_mm(90) == pytest.approx(9 * 0.35)   # across its row: n wires
    assert magnum.swath_mm(0) == pytest.approx(0.5, abs=0.02)                            # along it: a thin line
    row = magnum.magnum()
    assert row["row_mm"] + row["thickness_mm"] == pytest.approx(magnum.swath_mm(90))
    assert cartridge.load("lutin-9m-ex", row_deg=30).swath_mm(120) == pytest.approx(magnum.swath_mm(90))
    with pytest.raises(ValueError, match="lays no mark"):
        cartridge.load("picosecond-laser-pen")


def test_the_tone_gives_the_direction_from_dark_to_light():
    assert descent_angle(_block()) == pytest.approx(0.0, abs=1.0)
    assert math.cos(math.radians(descent_angle(_block()[:, ::-1]))) == pytest.approx(-1.0, abs=1e-3)
    assert descent_angle(_block().T.copy()) == pytest.approx(-90.0, abs=1.0)   # dark at the top: strokes run down
    assert descent_angle(np.zeros((50, 50))) == 0.0


def test_whips_begin_dark_fade_out_and_follow_the_tone():
    strokes, info, program = _strokes("whip")
    assert info["angle_deg"] == pytest.approx(0.0, abs=1.0) and info["pitch_mm"] == info["mark"]["width_mm"]
    assert len(strokes) == pytest.approx(15 / info["pitch_mm"], abs=1.5)   # 15 mm at the 3RL's own width
    for st in strokes:
        assert st["points_mm"][0, 0] < st["points_mm"][-1, 0]
        assert st["deposition"][0] > 0.95 and st["deposition"][-1] < 0.1
        assert len(st["points_mm"]) <= 4   # a straight ramp simplifies to its ends
    coverage, tone = _coverage(program), _block()
    assert np.abs(coverage - tone)[tone > 0.05].mean() < 0.05
    assert coverage[25:175, 60:100].mean() > 0.8 > 0.2 > coverage[25:175, 300:340].mean()


def test_a_pendulum_is_one_zigzag_lifted_at_its_light_turns_and_faster_than_whips():
    strokes, _, program = _strokes("pendulum")
    assert len(strokes) == 1
    pts, dep = strokes[0]["points_mm"], strokes[0]["deposition"]
    turns = np.flatnonzero(np.abs(np.diff(np.sign(np.diff(pts[:, 0])))) > 0) + 1
    assert len(turns) >= 40   # two per pair of its 26 swings
    light = turns[pts[turns, 0] > 0]
    assert len(light) >= 20 and np.all(dep[light] == 0.0)
    rise = np.diff(dep) / np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert rise.max() <= 1 / Shading().rise_mm + 1e-9
    whip_s = _strokes("whip")[2]["stats"]["time_estimate"]["modeled_s"]
    assert program["stats"]["time_estimate"]["modeled_s"] < 0.8 * whip_s


def test_a_pendulum_joins_through_tone_or_lifted_out_never_through_paper():
    first = (np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([1.0, 1.0]))
    second = (np.array([[11.0, 0.6], [20.0, 0.6]]), np.array([1.0, 1.0]))
    s = Shading(join_mm=3.0)
    paper, toned = (lambda pts: np.zeros(len(pts))), (lambda pts: np.ones(len(pts)))
    assert len(pendulums([[first], [second]], paper, s)) == 2
    assert len(pendulums([[first], [second]], toned, s)) == 1
    faint = [(p, np.array([1.0, 0.05])) for p, _ in (first,)], [(second[0], np.array([0.05, 1.0]))]
    (pts, dep), = pendulums(list(faint), paper, s)
    assert dep.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_a_magnum_shades_across_its_row_in_square_swaths():
    down = _block().T.copy()   # 15 mm wide, 30 mm tall from row 50, fading downward: across the rig's row
    strokes, info, program = _strokes("whip", down, "lutin-9m-ex")
    assert info["angle_deg"] == pytest.approx(-90.0) and info["off_descent_deg"] == pytest.approx(0.0, abs=1.0)
    assert info["pitch_mm"] == pytest.approx(3.15) and len(strokes) == pytest.approx(15 / 3.15, abs=1)
    coverage = _coverage(program)
    assert coverage[60:100, 25:175].mean() > 0.8 and np.abs(coverage - down)[down > 0.05].mean() < 0.05
    assert coverage[40:48, 25:175].max() < 0.05   # square at the top: a 3 mm round cap would reach 1.6 mm over it
    across = _strokes("whip", tool="lutin-9m-ex")[1]   # the block fades along the row: no whip can follow it
    assert abs(across["angle_deg"]) == pytest.approx(90.0) and across["off_descent_deg"] == pytest.approx(90.0, abs=1)


def test_the_render_sweeps_a_magnum_row():
    ops = [{"op": "stroke", "resource_id": "black", "points_m": [[-0.01, 0.0], [0.01, 0.0]]}]
    def drawn(row_deg):
        tool = {"magnum": {"row_deg": row_deg, "row_mm": 2.65, "thickness_mm": 0.5}}
        return _coverage({"page": {"size_m": [0.04, 0.02]}, "resources": [{"id": "black", "ink_id": "black",
                                                                             "tool": tool}], "ops": ops})
    across, along = drawn(90.0), drawn(0.0)   # the row across the stroke, then along it
    assert across[:, 200].sum() / PX_PER_MM == pytest.approx(3.15, abs=0.1)
    assert along[:, 200].sum() / PX_PER_MM == pytest.approx(0.5, abs=0.1)
    # square at x = -10 mm (column 100) but for the needles' 0.25 mm radius; a 3 mm round cap would reach 1.6 mm out
    assert across[:, 84:97].max() < 0.05 and across[88:112, 101:105].min() > 0.9
    assert along[98:102, 88:98].min() > 0.9   # along the row it reaches half the row past its ends


def test_deposition_rises_no_faster_than_rise_mm_and_ramps_simplify():
    pts = np.column_stack([np.arange(0.0, 2.01, 0.25), np.zeros(9)])
    step = np.r_[np.zeros(4), np.ones(5)]
    assert limit_rise(pts, step, 4.0)[-1] == pytest.approx(0.0625 * 5)
    assert limit_rise(pts, step[::-1], 4.0).tolist() == step[::-1].tolist()   # lifting out is never limited
    edge = np.r_[0.1, 0.6, np.ones(7)]   # an antialiased dark edge: the descent goes straight to depth
    assert start_in(pts, edge, 4.0)[1].tolist() == [1.0] * 7
    kept_pts, kept_dep = simplify(pts, np.linspace(1.0, 0.0, 9), 0.02)
    assert len(kept_pts) == 2 and kept_dep.tolist() == [1.0, 0.0]


def test_cli_shade_writes_a_render_a_preview_program_and_a_report(tmp_path):
    image = tmp_path / "tone.png"
    write_png(image, to_8bit(np.repeat((1.0 - _block())[..., None], 3, axis=2)))
    out = tmp_path / "shade"
    assert cli.main(["shade", str(image), "--width-mm", "40", "--technique", "pendulum", "--ink", "true_black",
                     "--out", str(out)]) == 0
    report = json.loads((out / "shade.json").read_text())
    assert report["strokes"] == 1 and report["shading"]["technique"] == "pendulum"
    assert report["render"]["inks"]["true_black"]["partial_mm"] > 0
    assert json.loads((out / "preview-program.json").read_text())["ops"][0]["deposition"]
    assert (out / "render.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    blank = tmp_path / "blank.png"
    write_png(blank, np.full((20, 20, 3), 255, np.uint8))
    assert cli.main(["shade", str(blank), "--width-mm", "10", "--out", str(tmp_path / "x")]) == 2
    assert cli.main(["shade", str(image), "--width-mm", "40", "--tool", "lutin-9m-ex", "--out", str(out)]) == 0
    report = json.loads((out / "shade.json").read_text())
    assert report["shading"]["mark"]["kind"] == "magnum" and report["shading"]["off_descent_deg"] == 90.0
    assert json.loads((out / "preview-program.json").read_text())["resources"][0]["tool"]["magnum"]["row_mm"] > 2


def _repo_with_a_fading_3rl(root, repo):
    """The checkout, linked, with one more datasheet: the 3RL with a 0.8 mm fade lift. No shipped tool records one
    yet; the ballpoint's paper ladder retired with it (2026-11-02), and a needle's on skin is unmeasured."""
    (root / "config" / "tools").mkdir(parents=True)
    for entry in repo.iterdir():
        if entry.name not in ("config", ".git"):
            (root / entry.name).symlink_to(entry)
    for entry in (repo / "config").iterdir():
        if entry.name != "tools":
            (root / "config" / entry.name).symlink_to(entry)
    for entry in (repo / "config" / "tools").iterdir():
        (root / "config" / "tools" / entry.name).symlink_to(entry)
    sheet = (repo / "config" / "tools" / "lutin-3rl-bugpin.yaml").read_text()
    (root / "config" / "tools" / "fading-3rl.yaml").write_text(
        sheet.replace("\nstroke_mm:", "\nfade_lift_mm: 0.8\nstroke_mm:", 1))
    return root


def test_a_shade_artwork_compiles_to_strokes_that_rise_as_they_fade(tmp_path):
    """`tatbot art shade` writes an acquired artwork (tatbot-art-shade/1) that `tatbot ros compile` turns into a
    program for a tool with a fade lift, whose points ride (1 - deposition) of it over the pen's path."""
    from pathlib import Path

    import tatbot_ink
    from tatbot_motion.plan import page_points

    image = tmp_path / "tone.png"
    write_png(image, to_8bit(np.repeat((1.0 - _block())[..., None], 3, axis=2)))
    out = tmp_path / "shade"
    assert cli.main(["shade", str(image), "--width-mm", "40", "--out", str(out)]) == 0
    repo = _repo_with_a_fading_3rl(tmp_path / "repo", Path(__file__).resolve().parents[2])
    result = tatbot_ink.compile(out / "artwork.json", repo=repo, tool_id="fading-3rl")
    strokes = [op for op in result["ops"] if op["op"] == "stroke"]
    assert len(strokes) == json.loads((out / "shade.json").read_text())["strokes"]
    assert all("deposition" in op for op in strokes)
    whip = strokes[len(strokes) // 2]
    heights = page_points(whip, np.eye(4), 0.0, 0.0008)[:, 2]
    np.testing.assert_allclose(heights, 0.0008 * (1.0 - np.asarray(whip["deposition"])), atol=1e-12)
    assert heights[0] < 1e-4 and heights[-1] > 0.0007   # in at its dark start, lifted out where the tone ends


def test_a_turned_photo_is_shaded_upright_as_separate_reads_it(tmp_path):
    from PIL import Image

    stored = Image.new("L", (40, 20))
    exif = stored.getexif()
    exif[0x0112] = 6   # stored on its side: shown turned 90 degrees, 20 wide and 40 tall
    stored.save(tmp_path / "photo.jpg", exif=exif)
    assert tone_from_image(tmp_path / "photo.jpg", 2.0, 10.0).shape == (40, 20)
