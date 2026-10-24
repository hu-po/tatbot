"""The fiducial-tile skin stencil: code separation, the printed seed-0 sheets, and that every marker reads back."""

import itertools
import json

import cv2
import numpy as np
import pytest
import stencil_tile

# The sheets printed (forearm and silicone 2026-10-17, forearm18 2026-10-23): layout -> marker ids in place order.
# A change here is a new print.
PRINTED = {
    "forearm": [8, 22, 12, 7, 15, 23, 4, 0, 20, 26, 27, 21],
    "forearm18": [6, 3, 9, 16, 17, 1],
    "silicone": [8, 20, 14, 11, 16, 12, 19, 17, 7, 27, 18, 6, 28, 26, 1, 9, 23, 21, 5, 22, 0, 15, 10],
}


def bits(code):
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5), code, 6)
    return tag[1:5, 1:5] > 127


def distance(a, b):
    """Bits apart under every turn of b, and a code against its own turns."""
    turns = range(1, 4) if a == b else range(4)
    return min(int((bits(a) != np.rot90(bits(b), k)).sum()) for k in turns)


def test_forearm_codes_are_six_bits_apart():
    codes = stencil_tile.FOREARM_CODES
    assert min(distance(a, b) for a, b in itertools.combinations_with_replacement(codes, 2)) >= 6


def test_forearm18_codes_are_six_bits_apart_and_none_of_the_forearm_prints():
    """Both prints may sit on one forearm: none of forearm18's codes is the forearm sheet's, each 5 bits from all."""
    codes = stencil_tile.FOREARM18_CODES
    assert min(distance(a, b) for a, b in itertools.combinations_with_replacement(codes, 2)) >= 6
    assert min(distance(a, b) for a in codes for b in PRINTED["forearm"]) >= 5


@pytest.mark.parametrize("layout", sorted(stencil_tile.LAYOUTS))
def test_seed_zero_is_the_printed_sheet(tmp_path, layout):
    out = stencil_tile.generate(layout, 0, tmp_path)
    record = json.loads((tmp_path/f"{layout}-seed0.json").read_text())
    assert [m["id"] for m in record["markers"]] == PRINTED[layout]
    assert len(set(PRINTED[layout])) == len(PRINTED[layout])
    assert out["record"] == record


def test_silicone_anchors_fill_the_corners():
    record = stencil_tile.build("silicone", 0)[1]
    anchors = [m for m in record["markers"] if m["role"] == "anchor"]
    tw, th = record["tile_mm"]
    assert {(round(m["corners_mm"][0][0], 6), round(m["corners_mm"][0][1], 6)) for m in anchors} == {
        (0, 0), (tw - 40, 0), (0, th - 40), (tw - 40, th - 40)}
    assert {m["id"] for m in anchors} <= set(stencil_tile.FOREARM_CODES)
    assert record["stencil_mm"][0] <= 128 and record["stencil_mm"][1] <= 172


@pytest.mark.parametrize("layout", sorted(stencil_tile.LAYOUTS))
def test_every_marker_reads_back_where_the_record_puts_it(layout):
    img, record = stencil_tile.build(layout, 0)
    ppm = 300 / 25.4
    sheet = stencil_tile.to_dpi(img, 300)
    sheet = cv2.erode(sheet, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # a transfer's ~0.4 mm bleed
    sheet = cv2.copyMakeBorder(sheet, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5))
    corners, ids, _ = detector.detectMarkers(sheet)
    seen = {int(i): (c.reshape(4, 2) - 40) / ppm for i, c in zip(ids.flatten(), corners, strict=True)}
    origin = np.array(record["tile_origin_in_stencil_mm"])
    for marker in record["markers"]:
        want = np.array(marker["corners_mm"]) + origin
        assert np.abs(seen[marker["id"]] - want).max() < 0.4, marker  # a few 300 dpi px of contour-corner bias


def test_generate_writes_an_installable_tile_reference(tmp_path, capsys):
    import stencil_generate
    import stencil_reference
    from fiducials import stencil

    assert {stencil_tile.GENERATOR} == stencil_reference.TILE_GENERATORS
    argv = ["--design", "fiducial-tile", "--preset", "silicone", "--seed", "0", "--output", str(tmp_path / "art")]
    assert stencil_generate.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    installed = stencil_reference.install(tmp_path / "art" / "tracking.json", root=tmp_path / "logs")
    page = stencil_reference.installed_page(out["pattern_id"], root=tmp_path / "logs")
    assert page["size_m"] == page["clear_m"] == [0.111, 0.1575]
    tile = stencil.load(installed)
    assert tile.ids == set(PRINTED["silicone"]) and len(tile.anchors) == 4
    assert (tmp_path / "art" / "print" / "silicone-seed0-300dpi.png").is_file()
    (installed / "tile.json").write_text("{}")   # a layout that is not the one the manifest names
    with pytest.raises(ValueError, match="tile.json"):
        stencil_reference.load(installed / "tracking.json")


def test_a_tile_seed_is_an_integer(capsys):
    import stencil_generate

    with pytest.raises(SystemExit):
        stencil_generate.main(["--seed", "tatbot-42", "--output", "x"])
    assert "seed is a non-negative integer" in capsys.readouterr().err


def test_a_retired_coded_reference_is_refused_by_name(tmp_path, capsys):
    """An old coded print's manifest left installed loads as nothing: the observer and the stack read tiles only."""
    import hashlib

    import stencil_generate
    import stencil_reference

    assert stencil_generate.main(["--seed", "0", "--output", str(tmp_path)]) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "tracking.json").read_text())
    manifest.pop("tile")
    manifest["generator"] = {"generator_version": "coded-fol-1", "svg_sha256": "0" * 64}
    manifest["pattern_id"] = "stencil-" + hashlib.sha256(json.dumps(manifest["generator"], sort_keys=True)
                                                         .encode()).hexdigest()
    manifest["reference_id"] = stencil_reference.reference_digest(manifest)
    (tmp_path / "tracking.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not a fiducial-tile print"):
        stencil_reference.load(tmp_path / "tracking.json")


def test_the_stack_draws_on_the_silicone_tile_seed_0(tmp_path, capsys):
    """stack.yaml's page.pattern_id is the silicone tile printed on 2026-10-17, regenerated."""
    import stencil_generate
    from cli_runner import REPO

    argv = ["--design", "fiducial-tile", "--preset", "silicone", "--seed", "0", "--output", str(tmp_path)]
    assert stencil_generate.main(argv) == 0
    pattern = json.loads(capsys.readouterr().out)["pattern_id"]
    assert f"pattern_id: {pattern}" in (REPO/"ros/tatbot_bringup/config/stack.yaml").read_text()
