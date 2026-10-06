"""Preserve DBV3 geometry while handing it to Inkmap and the ROS compiler."""
from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SVG_NS = "http://www.w3.org/2000/svg"
MAX_SVG_BYTES = 2_000_000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def normalize_svg(raw: bytes) -> tuple[str, dict]:
    """DBV3's plain Batik export has physical dimensions and no viewBox.

    SVG user units are CSS pixels (96/inch). Adding the corresponding viewBox
    preserves scale, transforms, paths and pen styles. Never fit to ink bounds.
    XML declarations, comments and an external SVG DTD are serialization only;
    entities, internal DTD subsets and other processing instructions are refused.
    The native path decoder subsequently validates admitted geometry and pen attributes.
    """
    if not raw or len(raw) > MAX_SVG_BYTES:
        raise ValueError("expected a DBV3 SVG of at most 2 MB")
    text = raw.decode("iso-8859-1")
    if re.search(r"<!ENTITY|<!DOCTYPE[^>]*\[", text, re.I):
        raise ValueError("SVG entities and internal DTD subsets are not supported")
    text = re.sub(r"^\s*<\?xml\s[^?]*\?>", "", text, count=1)
    text = re.sub(r"<!DOCTYPE\s+svg\s+PUBLIC\s+(['\"])-//W3C//DTD SVG 1.0//EN\1\s+"
                  r"(['\"])http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd\2\s*>", "", text)
    if "<!DOCTYPE" in text.upper() or "<?" in text:
        raise ValueError("unsupported XML declaration")
    root = ET.fromstring(text)
    if root.tag != f"{{{SVG_NS}}}svg":
        raise ValueError("expected SVG namespace")
    size = [_millimetres(root.get(key, "")) for key in ("width", "height")]
    if root.get("viewBox") is None:
        root.set("viewBox", f"0 0 {size[0] * 96 / 25.4:.12g} {size[1] * 96 / 25.4:.12g}")
    # ElementTree drops only comments/declarations and unused namespace bindings.
    # Unsupported elements and attributes remain for the native decoder to refuse.
    ET.register_namespace("", SVG_NS)
    svg = ET.tostring(root, encoding="unicode")
    return svg, {"size_mm": size, "view_box": root.get("viewBox"),
                 "raw_path_elements": sum(n.tag == f"{{{SVG_NS}}}path" for n in root.iter()),
                 "normalizer": "dbv3-batik-svg/1", "svg_user_units_per_inch": 96}


def _millimetres(value: str) -> float:
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?mm", value):
        raise ValueError("DBV3 export must state width and height in mm")
    number = float(value[:-2])
    if not math.isfinite(number) or not 0 < number <= 2000:
        raise ValueError("page dimension out of range")
    return number
