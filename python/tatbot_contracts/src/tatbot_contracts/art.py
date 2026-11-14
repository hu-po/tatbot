"""Source art (`tatbot.inkmap-art/1`, config/inkmap/art.schema.json): a design as an image, what Inkmap places.

Stdlib only, the Python half of web/inkmap/src/core/art.ts. The art pipeline reads it from a tattoo brief
(`tatbot.inkmap-brief/1`) to separate the image at the width chosen in Inkmap; DrawingBot V3 makes the strokes
there. `art_px` is the art's rectangle by the pipeline's own rule (python/tatbot_art image.art_box).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import re
import struct
from typing import Any

from tatbot_contracts.canonical import canonical_digest

SCHEMA = "tatbot.inkmap-art/1"
BRIEF_SCHEMA = "tatbot.inkmap-brief/1"
RENDER_SCHEMA = "tatbot.inkmap-render/1"
MAX_ART_BYTES = 8_000_000
_SHA = re.compile(r"[0-9a-f]{64}")
_INK = re.compile(r"[a-z0-9_]+")
_FIELDS = {"schema", "content_sha256", "name", "image", "art_px", "source", "palette"}


def _keys(value: Any, expected: set[str], where: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{where}: expected fields {sorted(expected)}")
    return value


def _text(value: Any, where: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip() or len(value) > 10_000:
        raise ValueError(f"{where}: expected a bounded nonempty string")


def image_size(data: bytes, media_type: str) -> tuple[int, int]:
    """(width, height) from a PNG's IHDR or a JPEG's frame header, without decoding the image."""
    if media_type == "image/png":
        if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 24:
            raise ValueError("art image: not a PNG")
        return struct.unpack(">II", data[16:24])
    if data[:2] != b"\xff\xd8":
        raise ValueError("art image: not a JPEG")
    i = 2
    frames = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in frames:
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + length
    raise ValueError("art image: JPEG has no frame header")


def image_bytes(record: dict) -> bytes:
    try:
        return base64.b64decode(record["image"]["data_base64"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("art image: data_base64 is not base64") from exc


def _image(record: dict) -> None:
    image = _keys(record["image"], {"media_type", "sha256", "width_px", "height_px", "data_base64"}, "art.image")
    if image["media_type"] not in ("image/png", "image/jpeg"):
        raise ValueError("art.image.media_type: PNG or JPEG only")
    data = image_bytes(record)
    if len(data) > MAX_ART_BYTES:
        raise ValueError("art image exceeds 8 MB")
    if hashlib.sha256(data).hexdigest() != image["sha256"]:
        raise ValueError("art image bytes differ from their digest")
    if image_size(data, image["media_type"]) != (image["width_px"], image["height_px"]):
        raise ValueError("art image size differs from the record")


def _art_box(record: dict) -> None:
    box, image = record["art_px"], record["image"]
    if not (isinstance(box, list) and len(box) == 4 and all(type(v) is int and v >= 0 for v in box)):
        raise ValueError("art.art_px: expected four whole pixel values")
    x, y, w, h = box
    if w < 1 or h < 1 or x + w > image["width_px"] or y + h > image["height_px"]:
        raise ValueError("art.art_px: the art lies outside its image")


def _source(value: Any) -> None:
    source = _keys(value, {"kind", "identifier", "license", "attribution", "generation"}, "art.source")
    if source["kind"] not in ("stock", "imported", "generated"):
        raise ValueError("art.source.kind: unknown")
    for field in ("identifier", "license", "attribution"):
        _text(source[field], f"art.source.{field}", nullable=True)
    if source["generation"] is not None:
        generation = _keys(source["generation"], {"prompt", "model", "generator"}, "art.source.generation")
        _text(generation["prompt"], "art.source.generation.prompt")
        _text(generation["model"], "art.source.generation.model", nullable=True)
        _text(generation["generator"], "art.source.generation.generator", nullable=True)


def _palette(palette: Any) -> None:
    if palette is None:
        return
    _keys(palette, {"inks", "style"}, "art.palette")
    inks = palette["inks"]
    if not (isinstance(inks, list) and 1 <= len(inks) <= 5 and all(isinstance(i, str) and _INK.fullmatch(i) for i in inks)):
        raise ValueError("art.palette.inks: expected one to five ink ids")
    _text(palette["style"], "art.palette.style", nullable=True)


def validate_art_record(record: Any) -> dict:
    """A source-art record whose digest, image bytes, image size and art rectangle hold together."""
    _keys(record, _FIELDS, "art")
    if record["schema"] != SCHEMA:
        raise ValueError(f"art: expected {SCHEMA}; Inkmap places source art, not stroke artwork")
    _text(record["name"], "art.name")
    if not isinstance(record["content_sha256"], str) or not _SHA.fullmatch(record["content_sha256"]):
        raise ValueError("art.content_sha256: expected SHA-256")
    if canonical_digest(record) != record["content_sha256"]:
        raise ValueError("art: content digest differs")
    _image(record)
    _art_box(record)
    _source(record["source"])
    _palette(record["palette"])
    return record


def validate_brief(brief: Any) -> dict:
    """A tattoo brief whose digest holds, whose art is valid, and whose placements each name placed art."""
    _keys(brief, {"schema", "content_sha256", "name", "body", "art", "placements"}, "brief")
    if brief["schema"] != BRIEF_SCHEMA:
        raise ValueError(f"brief: expected {BRIEF_SCHEMA}")
    if canonical_digest(brief) != brief["content_sha256"]:
        raise ValueError("brief: content digest differs")
    if not isinstance(brief["placements"], list) or not brief["placements"]:
        raise ValueError("brief: needs at least one placement")
    for record in brief["art"].values():
        validate_art_record(record)
    used = {p["art_id"] for p in brief["placements"]}
    if used != set(brief["art"]):
        raise ValueError("brief: the art registry must hold exactly the placed art")
    ids = [p["id"] for p in brief["placements"]]
    if len(set(ids)) != len(ids):
        raise ValueError("brief: duplicate placement id")
    for p in brief["placements"]:
        width = p["width_mm"]
        if not (isinstance(width, (int, float)) and width > 0 and abs(width * 10 - round(width * 10)) < 1e-9):
            raise ValueError(f"brief placement {p['id']}: width_mm must be positive, to 0.1 mm")
    return brief


def make_render(*, art_sha256: str, width_mm: float, height_mm: float, mirror: bool, png: bytes, minutes: float,
                dips: int, rinses: int, lost_inks: list[str], weakest_source: str) -> dict:
    """What a separation gives back to Inkmap (`tatbot.inkmap-render/1`, config/inkmap/render.schema.json): a picture
    of the predicted ink on white, bound to the art, width and mirror it was made for. Pixels only, never strokes."""
    width_px, height_px = image_size(png, "image/png")
    record = {"schema": RENDER_SCHEMA, "content_sha256": "", "art_sha256": art_sha256, "width_mm": width_mm,
              "height_mm": height_mm, "mirror": mirror,
              "image": {"media_type": "image/png", "sha256": hashlib.sha256(png).hexdigest(), "width_px": width_px,
                        "height_px": height_px, "data_base64": base64.b64encode(png).decode()},
              "minutes": minutes, "dips": dips, "rinses": rinses, "lost_inks": lost_inks,
              "weakest_source": weakest_source}
    record["content_sha256"] = canonical_digest(record)
    return validate_render(record)


def validate_render(record: Any) -> dict:
    _keys(record, {"schema", "content_sha256", "art_sha256", "width_mm", "height_mm", "mirror", "image", "minutes",
                   "dips", "rinses", "lost_inks", "weakest_source"}, "render")
    if record["schema"] != RENDER_SCHEMA:
        raise ValueError(f"render: expected {RENDER_SCHEMA}")
    if canonical_digest(record) != record["content_sha256"]:
        raise ValueError("render: content digest differs")
    if not isinstance(record["art_sha256"], str) or not _SHA.fullmatch(record["art_sha256"]):
        raise ValueError("render.art_sha256: expected SHA-256")
    _image(record)
    return record
