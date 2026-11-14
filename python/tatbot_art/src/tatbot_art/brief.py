"""`tatbot art separate --brief`: one placement of an Inkmap tattoo brief as the image to separate.

Inkmap places the source image and chooses its drawn width; its brief (`tatbot.inkmap-brief/1`) carries both.
This writes the placement's image under OUT/source/, with a `tatbot.art-generation/1` sidecar when the art was
generated, so `separate` reads it as it reads a corpus image. A mirrored placement is separated mirrored: the
compile places a design on the page but never turns or flips it. The rotation is recorded, not applied, because
it is measured against the body's surface, not the page.

The crop is checked against the brief's `art_px`, the rectangle Inkmap sized: the pipeline separates exactly that
rectangle at `width_mm`, so a crop that disagrees would draw the design at another size.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from tatbot_contracts.art import (
    BRIEF_SCHEMA,
    SCHEMA,
    image_bytes,
    make_render,
    validate_art_record,
    validate_brief,
)
from tatbot_contracts.canonical import canonical_digest

from .color import to_8bit
from .png import write_png

EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg"}
CROP_TOLERANCE_PX = 2   # Inkmap reads the image through the browser's decoder; JPEG decoders differ by a pixel


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "art"


def placement(brief: dict, placement_id: str | None) -> dict:
    """The named placement, or the brief's only one."""
    places = brief["placements"]
    ids = ", ".join(p["id"] for p in places)
    if placement_id is not None:
        for place in places:
            if place["id"] == placement_id:
                return place
        raise ValueError(f"--placement {placement_id}: the brief places {ids}")
    if len(places) > 1:
        raise ValueError(f"the brief places {len(places)} designs ({ids}); name one with --placement")
    return places[0]


def in_order(brief_path: Path) -> list[tuple[str, str]]:
    """Every placement's id and the folder it separates into, numbered in the brief's drawing order."""
    brief = validate_brief(json.loads(brief_path.read_text()))
    return [(p["id"], f"{n:02d}-{_slug(brief['art'][p['art_id']]['name'])}")
            for n, p in enumerate(brief["placements"], 1)]


def source(brief_path: Path, placement_id: str | None, out: Path) -> dict:
    """Write the placement's image and sidecar; return its path, the width, the art's inks and style, and the
    record the separation report keeps."""
    from PIL import Image, ImageOps

    from .image import art_box, flatten_paper, load, on_paper

    brief = validate_brief(json.loads(brief_path.read_text()))
    place = placement(brief, placement_id)
    art = brief["art"][place["art_id"]]
    data = image_bytes(art)
    folder = out / "source"
    folder.mkdir(parents=True, exist_ok=True)
    stem = _slug(art["name"])
    x, y, w, h = art["art_px"]
    if place["mirror"]:
        path = folder / f"{stem}-mirrored.png"
        with Image.open(io.BytesIO(data)) as im:
            ImageOps.mirror(on_paper(im)).save(path)
        x = art["image"]["width_px"] - x - w
    else:
        path = folder / f"{stem}{EXTENSIONS[art['image']['media_type']]}"
        path.write_bytes(data)
    if generation := art["source"]["generation"]:
        path.with_suffix(".json").write_text(json.dumps({
            "schema": "tatbot.art-generation/1", "prompt": generation["prompt"], "model": generation["model"],
            "generator": generation["generator"], "subject": art["name"]}, indent=1) + "\n")
    crop = art_box(flatten_paper(load(path)))
    if max(abs(a - b) for a, b in zip(crop, (x, y, w, h), strict=True)) > CROP_TOLERANCE_PX:
        raise ValueError(f"{art['name']}: the pipeline crops the image to {list(crop)} px but the brief sized "
                         f"{[x, y, w, h]}; place it again in Inkmap")
    palette = art["palette"] or {}
    return {"image": path, "width_mm": place["width_mm"], "inks": palette.get("inks"), "style": palette.get("style"),
            "report": {"schema": brief["schema"], "content_sha256": brief["content_sha256"], "name": brief["name"],
                       "placement": place["id"], "art_sha256": art["content_sha256"],
                       "width_mm": place["width_mm"], "height_mm": place["height_mm"],
                       "rotation_rad": place["rotation_rad"], "mirror": place["mirror"], "site": place["site"]}}


def write_render(out: Path, report: dict, picture: np.ndarray, lost_inks: list[str]) -> Path:
    """`inkmap-render.json` beside a brief's separation: what Inkmap shows on the body in place of the source art."""
    placed, time = report["brief"], report["time"]
    png = write_png(out / "ink-on-white.png", to_8bit(picture)).read_bytes()
    record = make_render(art_sha256=placed["art_sha256"], width_mm=placed["width_mm"], height_mm=placed["height_mm"],
                         mirror=placed["mirror"], png=png, minutes=time["minutes"], dips=time["dips"],
                         rinses=time["rinses"], lost_inks=lost_inks, weakest_source=report["weakest_source"])
    path = out / "inkmap-render.json"
    path.write_text(json.dumps(record))
    return path


def stock_art(entry: dict, image: bytes) -> dict:
    """The art record of a stock collection entry, as Inkmap builds it (web/inkmap/src/core/design-assets.ts stockRecord):
    the same digest, so a render made from it binds the art a visitor places."""
    record = {"schema": SCHEMA, "content_sha256": "", "name": entry["name"], "art_px": entry["art_px"],
              "source": entry["source"], "palette": entry["palette"],
              "image": {"media_type": "image/jpeg" if entry["image"].endswith(".jpg") else "image/png",
                        "sha256": hashlib.sha256(image).hexdigest(), "width_px": entry["width_px"],
                        "height_px": entry["height_px"], "data_base64": base64.b64encode(image).decode()}}
    record["content_sha256"] = canonical_digest(record)
    return validate_art_record(record)


def stock_brief(art_id: str, record: dict, width_mm: float) -> dict:
    """A one-placement brief for stock art at a width. It places the art on no body: only the image and size are read."""
    _, _, art_width, art_height = record["art_px"]
    doc = {"schema": BRIEF_SCHEMA, "content_sha256": "", "name": record["name"], "body": {}, "art": {art_id: record},
           "placements": [{"id": f"stock-{art_id}", "art_id": art_id, "width_mm": width_mm,
                           "height_mm": round(width_mm * art_height / art_width, 3), "rotation_rad": 0, "mirror": False,
                           "anchor": {"face": 0, "barycentric": [1, 0, 0]}, "site": None, "sentence": None}]}
    doc["content_sha256"] = canonical_digest(doc)
    return validate_brief(doc)
