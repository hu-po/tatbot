"""python/tatbot_lab read: a drawn cell measured from before and after frames of a real tile print, rendered as a
wrist camera would see it (about 3.8 px/mm, turned and tilted, the after frame from a pose a few pixels off), with
ink of known darkness, width and run-out drawn into the after frame."""
import json

import cv2
import numpy as np
import pytest
import stencil_tile
from tatbot_lab import patterns, tile
from tatbot_lab.read import frames, read_cell

SKIN = np.array([185.0, 195.0, 225.0])     # BGR
PRINT = np.array([125.0, 55.0, 105.0])     # the violet transfer
INK = np.array([35.0, 30.0, 30.0])         # black ink
PPM = stencil_tile.PPM
CAMERA = (900, 700)


@pytest.fixture(scope="module")
def sheet(tmp_path_factory):
    img, record = stencil_tile.build("forearm18", 0)
    directory = tmp_path_factory.mktemp("print")
    (directory / "tile.json").write_text(json.dumps(record))
    (directory / "tracking.json").write_text(json.dumps({"pattern_id": "stencil-" + "ab" * 32,
                                                         "page_mm": record["tile_mm"], "clear_center_uv": [0, 0, 1, 1]}))
    (left, top), (tw, th) = record["tile_origin_in_stencil_mm"], record["tile_mm"]
    page_to_px = np.array([[PPM, 0, (left + tw / 2) * PPM], [0, -PPM, (top + th / 2) * PPM], [0, 0, 1]])
    colour = np.where((img == 0)[..., None], PRINT, SKIN)
    return {"colour": colour, "page_to_px": page_to_px, "t": frames.tile(directory), "print": tile.load(directory)}


def _camera(shift_px=(0.0, 0.0), px_per_mm=3.8) -> np.ndarray:
    """Page mm -> camera px: 3.8 px/mm (the wrist's) unless told, turned 8 degrees, a little perspective, the tile in
    the middle."""
    a, k = np.radians(8), px_per_mm
    h = np.array([[k * np.cos(a), -k * np.sin(a), CAMERA[0] / 2], [-k * np.sin(a), -k * np.cos(a), CAMERA[1] / 2],
                  [0.0004 * k / 3.8, 0.0002 * k / 3.8, 1.0]])
    return np.array([[1, 0, shift_px[0]], [0, 1, shift_px[1]], [0, 0, 1]]) @ h


def _ink(sheet, strokes, *, width_mm, opacity, ends_at=None):
    """The print with ink along `strokes` (page mm), `opacity` of the way to black, stopping `ends_at` mm into the
    drawing when given (a dip run out)."""
    mask = np.zeros(sheet["colour"].shape[:2], np.uint8)
    drawn = 0.0
    for s in strokes:
        length = float(np.linalg.norm(s[-1] - s[0]))
        if ends_at is not None and drawn >= ends_at:
            break
        end = s[-1] if ends_at is None or drawn + length <= ends_at else s[0] + (s[-1] - s[0]) * (ends_at - drawn) / length
        px = cv2.perspectiveTransform(np.array([[s[0], end]], float), sheet["page_to_px"])[0]
        cv2.line(mask, tuple(np.round(px[0]).astype(int)), tuple(np.round(px[1]).astype(int)), 255,
                 max(1, round(width_mm * PPM)))
        drawn += length
    m = (mask > 0)[..., None] * opacity   # coverage, mixed in linear light as ink and light mix
    return _srgb(_linear(sheet["colour"]) * (1 - m) + _linear(INK) * m)


def _linear(c):
    v = np.asarray(c, float) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb(lin):
    v = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, 1), 1 / 2.4) - 0.055)
    return v * 255.0


def _view(sheet, colour, shift_px=(0.0, 0.0), px_per_mm=3.8) -> np.ndarray:
    blurred = cv2.GaussianBlur(colour.astype(np.float32), (0, 0), 0.5 * PPM / px_per_mm)
    m = sheet["page_to_px"] @ np.linalg.inv(_camera(shift_px, px_per_mm))
    out = cv2.warpPerspective(blurred, m, CAMERA, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out, 0, 255).astype(np.uint8)


def _cell(sheet, technique, settings, dips_mm=(0.0,)):
    c = tile.cells(sheet["print"], 16.0)[2]
    strokes = [pts for pts, _ in patterns.place(technique, settings, c.box_mm)]
    return {"id": c.id, "trial": "t0001", "strokes_mm": [s.tolist() for s in strokes], "dips_mm": list(dips_mm)}, \
        strokes


def test_a_line_cell_reads_its_darkening_and_ink_width(sheet):
    settings = {"trim_mm": -2.0, "speed_mm_s": 4.0}
    cell, strokes = _cell(sheet, "line", settings)
    before = _view(sheet, sheet["colour"])
    after = _view(sheet, _ink(sheet, strokes, width_mm=0.6, opacity=0.8), shift_px=(2.5, -1.5))
    out = read_cell(cell, "line", settings, [(before, after)], sheet["t"])
    assert out["noise_dL"] < 2.0
    assert min(out["per_line"]) > 8 * out["noise_dL"]
    # ink width: an opaque line holding the same ink, 0.8 of 0.6 mm, the ink taking 98% of the skin's light
    assert np.median(out["ink_width_mm"]) == pytest.approx(0.6 * 0.8 * 0.98, abs=0.1)
    assert out["speckle"]["share"] < 0.02


def test_a_carry_cell_finds_where_the_dip_ran_out_and_what_outlasted_its_cycle(sheet):
    settings = {"trim_mm": -2.0, "speed_mm_s": 4.0}
    cell, strokes = _cell(sheet, "carry", settings, dips_mm=(0.0, 120.0))
    before = _view(sheet, sheet["colour"])
    after = _view(sheet, _ink(sheet, strokes[:10], width_mm=0.5, opacity=0.8, ends_at=78.0), shift_px=(-2.0, 1.0))
    out = read_cell(cell, "carry", settings, [(before, after)], sheet["t"])
    first, second = out["carry"]
    assert first["from_mm"] == 0 and not first["outlasted"]
    assert first["strong_mm"] == pytest.approx(78, abs=12)   # ink stops 78 mm in, 6.5 strokes of 12 mm
    assert second["from_mm"] == 120 and second["peak"] < 3 * out["noise_dL"]   # nothing inked after the second dip


def test_a_flooded_dip_reads_wide_at_its_first_strokes_start_and_a_clean_one_does_not(sheet):
    """Five 10 mm lines on one dip. Clean: every stroke the same from end to end, so the stroke after the dip
    starts as wide as the steady line and as the other strokes' starts. Flooded: its first 3 mm 1.4 mm wide where
    the line is 0.5, and three drops thrown ahead of where the pen came down."""
    settings = {"trim_mm": -2.0, "speed_mm_s": 4.0}
    cell, strokes = _cell(sheet, "line", settings)
    before = _view(sheet, sheet["colour"])
    lines = _ink(sheet, strokes, width_mm=0.5, opacity=0.8)
    clean = read_cell(cell, "line", settings, [(before, _view(sheet, lines, shift_px=(1.5, -1.0)))], sheet["t"])
    flood = clean["flood"]
    assert flood["window_mm"] == 3.0 and [d["from_mm"] for d in flood["dips"]] == [0.0]
    assert flood["steady_width_mm"] == pytest.approx(0.5 * 0.8 * 0.98, abs=0.1)
    assert flood["over_steady"] == pytest.approx(1.0, abs=0.15) and flood["over_other_starts"] == pytest.approx(1.0, abs=0.15)
    assert flood["ahead_speckle"] < 0.02

    first = strokes[0]
    along = (first[-1] - first[0]) / np.linalg.norm(first[-1] - first[0])
    blob = [np.array([first[0], first[0] + 3.0 * along])]
    drops = [np.array([first[0] - k * along, first[0] - (k + 0.3) * along]) for k in (1.0, 2.0, 3.0)]
    wet = {**sheet, "colour": _ink({**sheet, "colour": lines}, blob, width_mm=1.4, opacity=0.8)}
    flooded = _ink(wet, drops, width_mm=0.8, opacity=0.8)
    out = read_cell(cell, "line", settings, [(before, _view(sheet, flooded, shift_px=(1.5, -1.0)))], sheet["t"])
    flood = out["flood"]
    assert flood["over_steady"] > 2.0 and flood["over_other_starts"] > 2.0
    assert flood["dips"][0]["start_width_mm"] == pytest.approx(1.4 * 0.8 * 0.98, abs=0.25)
    assert flood["other_starts_width_mm"] == pytest.approx(clean["flood"]["other_starts_width_mm"], abs=0.05)
    assert flood["ahead_speckle"] > 0.1 > clean["flood"]["ahead_speckle"]
    # a stroke's own numbers leave its ends out, so the flood does not move the line's width much
    assert np.median(out["ink_width_mm"]) == pytest.approx(np.median(clean["ink_width_mm"]), abs=0.05)


def test_a_carry_cells_flood_reads_each_dips_first_stroke(sheet):
    settings = {"trim_mm": -2.0, "speed_mm_s": 4.0}
    cell, strokes = _cell(sheet, "carry", settings, dips_mm=(0.0, 60.0))
    before = _view(sheet, sheet["colour"])
    after = _view(sheet, _ink(sheet, strokes, width_mm=0.5, opacity=0.8), shift_px=(-2.0, 1.0))
    flood = read_cell(cell, "carry", settings, [(before, after)], sheet["t"])["flood"]
    assert [d["from_mm"] for d in flood["dips"]] == [0.0, 60.0]
    assert flood["over_other_starts"] == pytest.approx(1.0, abs=0.2)


def test_a_glint_in_one_view_of_three_is_dropped(sheet):
    settings = {"trim_mm": -2.0, "speed_mm_s": 4.0}
    cell, strokes = _cell(sheet, "line", settings)
    inked = _ink(sheet, strokes, width_mm=0.6, opacity=0.8)
    views = []
    for k, shift in enumerate(((0.0, 0.0), (3.0, 2.0), (-2.0, 3.0))):
        after = _view(sheet, inked, shift)
        if k == 1:   # a wet-ink glint over the middle stroke in one view only
            centre = cv2.perspectiveTransform(np.array([[strokes[2].mean(axis=0)]], float), _camera(shift))[0, 0]
            cv2.circle(after, tuple(np.round(centre).astype(int)), 6, (255, 255, 255), -1)
        views.append((_view(sheet, sheet["colour"], shift), after))
    fused = read_cell(cell, "line", settings, views, sheet["t"])
    glinted = read_cell(cell, "line", settings, [views[1]], sheet["t"])
    clean = read_cell(cell, "line", settings, [views[0]], sheet["t"])
    assert fused["views"] == 3
    assert abs(fused["per_line"][2] - clean["per_line"][2]) < abs(glinted["per_line"][2] - clean["per_line"][2])


def test_the_overheads_coarse_frames_read_a_cells_mean_darkening(sheet):
    """The overhead at ~0.9 px/mm: a 10 mm square of strokes is nine pixels, enough for its mean change."""
    settings = {"trim_mm": -2.0, "speed_mm_s": 4.0}
    cell, strokes = _cell(sheet, "solid", settings)
    before = _view(sheet, sheet["colour"], px_per_mm=0.9)
    after = _view(sheet, _ink(sheet, strokes, width_mm=0.5, opacity=0.9), shift_px=(0.6, -0.4), px_per_mm=0.9)
    out = read_cell(cell, "solid", settings, [(before, after)], sheet["t"], instrument="overhead")
    assert out["dL_mean"] > 10 * out["noise_dL"]


def test_a_region_reads_markers_only_inside_it(sheet):
    """Two sheets of one print in view read every marker twice; --roi keeps one sheet's."""
    settings = {"trim_mm": -2.0, "speed_mm_s": 4.0}
    cell, strokes = _cell(sheet, "line", settings)
    before = _view(sheet, sheet["colour"])
    after = _view(sheet, _ink(sheet, strokes, width_mm=0.6, opacity=0.8), shift_px=(1.0, 0.0))
    whole = read_cell(cell, "line", settings, [(before, after)], sheet["t"], roi=(0, 0, CAMERA[0], CAMERA[1]))
    assert min(whole["per_line"]) > 8 * whole["noise_dL"]
    with pytest.raises(ValueError, match="fewer than"):
        read_cell(cell, "line", settings, [(before, after)], sheet["t"], roi=(0, 0, 40, 40))


def _facing(x_m, z_m=1.0):
    """A page centred at (x_m, 0, z_m) in the overhead's colour optical frame, its face toward the camera."""
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[:3, 3] = [x_m, 0.0, z_m]
    return pose


def test_a_chosen_sheets_region_holds_its_markers_and_none_of_the_other_mounts(sheet):
    """Two mounts carrying one print in the overhead's view at ~0.9 px/mm: the chosen sheet's pose and the capture's
    intrinsics bound the markers `lab read` takes, so every one is that sheet's, whichever reads larger."""
    optics = {"fx": 900.0, "fy": 900.0, "ppx": 450.0, "ppy": 350.0, "width": CAMERA[0], "height": CAMERA[1],
              "distortion_coefficients": [0.0] * 5}
    k = np.array([[optics["fx"], 0, optics["ppx"]], [0, optics["fy"], optics["ppy"]], [0, 0, 1]])
    colour = cv2.GaussianBlur(sheet["colour"].astype(np.float32), (0, 0), 0.5 * PPM / 0.9)
    frame = np.full((CAMERA[1], CAMERA[0], 3), SKIN, np.float32)
    left, right = _facing(-0.12), _facing(0.10)
    for pose in (left, right):
        h = k @ np.column_stack([pose[:3, 0] * 1e-3, pose[:3, 1] * 1e-3, pose[:3, 3]])   # page mm -> px
        warped = cv2.warpPerspective(colour, sheet["page_to_px"] @ np.linalg.inv(h), CAMERA,
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=(-1, -1, -1))
        frame = np.where(warped >= 0, warped, frame)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    for pose, side in ((left, -1), (right, 1)):
        found = frames.markers(frame, sheet["t"], "overhead",
                               roi=frames.sheet_roi(pose, sheet["print"].tile_mm, optics))
        assert len(found) >= 4
        assert all(set(np.sign(c[:, 0] - optics["ppx"])) == {side} for c in found.values())
    assert frames.sheet_roi(_facing(3.0), sheet["print"].tile_mm, optics) is None   # out of the frame


def test_a_view_that_cannot_place_the_cell_is_left_out_of_its_read(sheet):
    """A run's holds aim one at each cell: the hold over a cell across the sheet sees none of this one's markers
    (the xArm's -3.25 A/A pair, 2026-11-09). It is left out; the cell reads from the views that place it."""
    settings = {"trim_mm": -2.0, "speed_mm_s": 4.0}
    cell, strokes = _cell(sheet, "line", settings)
    before = _view(sheet, sheet["colour"])
    after = _view(sheet, _ink(sheet, strokes, width_mm=0.6, opacity=0.8), shift_px=(2.5, -1.5))
    blank = np.full_like(before, 128)
    out = read_cell(cell, "line", settings, [(blank, blank), (before, after)], sheet["t"])
    alone = read_cell(cell, "line", settings, [(before, after)], sheet["t"])
    assert out["views"] == 1 and out["per_line"] == alone["per_line"]


def _dotted(sheet, strokes, where=(), *, width_mm=0.35, opacity=0.5, dot_mm=1.1):
    """The print with ink along every chord of `strokes` (page mm polylines) and a dark dot at each of the page
    points `where`."""
    line, dots = np.zeros(sheet["colour"].shape[:2], np.uint8), np.zeros(sheet["colour"].shape[:2], np.uint8)
    for s in strokes:
        px = cv2.perspectiveTransform(np.array([s], float), sheet["page_to_px"])[0]
        cv2.polylines(line, [np.round(px).astype(np.int32)], False, 255, max(1, round(width_mm * PPM)))
    if len(where):
        for px in cv2.perspectiveTransform(np.array([where], float), sheet["page_to_px"])[0]:
            cv2.circle(dots, tuple(np.round(px).astype(int)), max(1, round(dot_mm * PPM / 2)), 255, -1)
    m = np.maximum((line > 0) * opacity, (dots > 0) * 0.9)[..., None]
    return _srgb(_linear(sheet["colour"]) * (1 - m) + _linear(INK) * m)


def test_a_cells_starts_stops_and_turns_are_read_apart(sheet):
    """The pen dwells where it comes down, where it lifts and where it turns, each differently (the dose model): a
    dark dot at every start darkens the starts alone, one at every stop the stops alone, one on every turn the
    turns alone."""
    settings = {"trim_mm": -2.0, "speed_mm_s": 3.5}
    before = _view(sheet, sheet["colour"])

    def read(technique, cell, colour):
        out = read_cell(cell, technique, settings, [(before, _view(sheet, colour, (2.0, -1.5)))], sheet["t"])
        return out, out["stroke_ends"]

    cell, strokes = _cell(sheet, "line", settings)
    _, even = read("line", cell, _dotted(sheet, strokes))
    assert even["turns"] is None and even["turns_over_middles"] is None and even["middles"] > 4.0
    _, started = read("line", cell, _dotted(sheet, strokes, [s[0] for s in strokes]))
    assert started["starts"] > even["starts"] + 8.0 and abs(started["stops"] - even["stops"]) < 1.0
    assert started["starts_over_middles"] > even["starts_over_middles"] + 0.8
    _, stopped = read("line", cell, _dotted(sheet, strokes, [s[-1] for s in strokes]))
    assert stopped["stops"] > even["stops"] + 8.0 and abs(stopped["starts"] - even["starts"]) < 1.0
    assert abs(stopped["middles"] - even["middles"]) < 1.0
    # the corners cell: an L, a zigzag and three strokes too short to have a middle; the turns are the L's and the
    # zigzag's inner vertices
    cell, strokes = _cell(sheet, "corners", settings)
    plain, plain_ends = read("corners", cell, _dotted(sheet, strokes))
    turned, turned_ends = read("corners", cell, _dotted(sheet, strokes, [p for s in strokes for p in s[1:-1]]))
    assert plain_ends["turns"] is not None and turned_ends["turns"] > plain_ends["turns"] + 8.0
    assert abs(turned_ends["starts"] - plain_ends["starts"]) < 1.0 and abs(turned_ends["stops"] - plain_ends["stops"]) < 1.0
    assert turned_ends["turns_over_middles"] > plain_ends["turns_over_middles"] + 0.8
    assert turned["ends_over_middles"] > plain["ends_over_middles"] + 0.2   # the lumped reading sees them too
