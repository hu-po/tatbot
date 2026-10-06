"""The sheet cache rebuilds what no longer matches the substrate — worn
variants included.

The base sheets are checked against the record's size, pitch and extent on
every request. Worn variants used to be rebuilt only alongside their base, so
a caller asking for no variants (the substrate tests, a run with sheet DR off)
rebuilt the base at the pad's new size and left the old variants behind; the
next run that wanted them composited a 512-wide sheet over a 452-wide ink
field and stopped at the first frame (2026-09-12, every node with a cache
from before the pad was re-measured).

Needs cv2, no render device:

    uv run pytest -q tests/test_sheet_cache.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import cv2
import numpy as np
from tatbot_sim import textures, tools

REPO = Path(__file__).resolve().parents[3]


def _sub(name):
    return tools.registry().load_substrate(name, REPO)


def test_a_worn_variant_left_behind_by_a_base_rebuild_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(textures, "TEX_DIR", tmp_path)
    sub = _sub("paper_pad")
    size = (sub.texel_rows, sub.texel_cols)

    # The base sheet, rebuilt by a caller that wanted no variants...
    base = textures.grid_paper_sheets(1, sub, seed=1)[0]
    assert cv2.imread(base["png"]).shape[:2] == size
    # ...over a worn variant and quad from the letter-size pad the cache held before.
    stale = tmp_path / "grid_00_w1"
    cv2.imwrite(str(stale.with_suffix(".png")), np.full((662, 512, 3), 255, np.uint8))
    textures._write_quad(stale, 0.2159, 0.2794)

    sheets = textures.grid_paper_sheets(1, sub, seed=1, wear_variants=1)
    [worn] = [s for s in sheets if s["png"].endswith("_w1.png")]
    assert cv2.imread(worn["png"]).shape[:2] == size
    hx, hy = textures._quad_extent(Path(worn["obj"]))
    assert abs(hx - sub.width_m / 2) < 1e-6 and abs(hy - sub.height_m / 2) < 1e-6


def test_a_matching_worn_variant_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(textures, "TEX_DIR", tmp_path)
    sub = _sub("paper_pad")
    first = textures.grid_paper_sheets(1, sub, seed=1, wear_variants=1)
    stamp = {s["png"]: Path(s["png"]).stat().st_mtime_ns for s in first}
    again = textures.grid_paper_sheets(1, sub, seed=1, wear_variants=1)
    assert [s["png"] for s in again] == [s["png"] for s in first]
    assert {s["png"]: Path(s["png"]).stat().st_mtime_ns for s in again} == stamp


def test_an_unreadable_skin_cache_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(textures, "TEX_DIR", tmp_path)
    sub = _sub("silicon_skin")
    png = tmp_path / "skin_silicon_skin_00.png"
    png.write_bytes(b"incomplete PNG")
    [sheet] = textures.skin_sheets(1, sub)
    assert cv2.imread(sheet["png"]).shape[:2] == (sub.texel_rows, sub.texel_cols)


def test_a_texture_replacement_keeps_the_old_image_until_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(textures, "TEX_DIR", tmp_path)
    sub = _sub("silicon_skin")
    png = tmp_path / "skin_silicon_skin_00.png"
    old = np.full((3, 4, 3), 127, np.uint8)
    cv2.imwrite(str(png), old)
    replace = textures.os.replace
    observed = []

    def inspect(staged, destination):
        if Path(destination) == png:
            np.testing.assert_array_equal(cv2.imread(str(png)), old)
            assert cv2.imread(str(staged)).shape[:2] == (sub.texel_rows, sub.texel_cols)
            observed.append(True)
        replace(staged, destination)

    monkeypatch.setattr(textures.os, "replace", inspect)
    [sheet] = textures.skin_sheets(1, sub)
    assert observed == [True]
    assert cv2.imread(sheet["png"]).shape[:2] == (sub.texel_rows, sub.texel_cols)
    assert not list(tmp_path.glob(".skin_silicon_skin_00.png.*"))


def test_parallel_paper_requests_share_one_complete_ruling(tmp_path, monkeypatch):
    monkeypatch.setattr(textures, "TEX_DIR", tmp_path)
    start = Barrier(4)

    def request(seed):
        start.wait(timeout=5)
        return textures.grid_paper_sheets(1, seed=seed)[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        sheets = list(pool.map(request, range(4)))
    for sheet in sheets:
        assert sheet == sheets[0]
        assert cv2.imread(sheet["png"]).shape[:2] == (textures.SIZE_Y, textures.SIZE_X)
        assert textures._quad_extent(Path(sheet["obj"])) == (textures.SHEET_W_M/2, textures.SHEET_H_M/2)
