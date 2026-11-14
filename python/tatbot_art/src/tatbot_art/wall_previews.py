"""Recorded robot renditions and scale previews for the Flash Wall."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml
from PIL import Image, ImageChops, ImageOps

from .acquire import REPO
from .inks import look
from .loop import measured


@lru_cache(maxsize=32)
def cartridge_name(tool_id: str) -> str:
    path = REPO / "config" / "tools" / f"{Path(tool_id).name}.yaml"
    if path.is_file():
        label = yaml.safe_load(path.read_text()).get("cartridge")
        if label:
            return label.split(",", 1)[0]
    return tool_id


def recipe(report: dict, catalog: dict) -> dict:
    refused = {layer["job"] for layer in report.get("refused", [])}
    layers = [layer for layer in report.get("layers", []) if layer.get("job") not in refused]
    inks = report.get("line_mm", dict.fromkeys(layer["ink"] for layer in layers))
    tools = report.get("tools", {})
    return {
        "strokes": list(dict.fromkeys(layer["fill"] for layer in layers)),
        "inks": [{"name": look(catalog, ink).name, "cartridge": cartridge_name(tools[ink]) if ink in tools else None}
                 for ink in inks],
    }


@lru_cache(maxsize=8)
def read_print(path: Path) -> tuple[dict, Image.Image]:
    from stencil_reference import load

    folder = path if path.is_dir() else path.parent
    reference, image_path = load(folder / "tracking.json")
    with Image.open(image_path) as image:
        # Keep the full marker pattern visible behind dark ink.
        sheet = ImageOps.colorize(image.convert("L"), "#b9aec3", "#ffffff")
    sheet.thumbnail((800, 800), Image.Resampling.LANCZOS)
    return reference, sheet


def place_on_stencil(sheet: Image.Image, ink: Image.Image, page_mm, canvas_mm, at_mm) -> Image.Image:
    """Place white-backed ink using the drawing stack's centred, y-up page coordinates."""
    w, h = sheet.size
    width_mm, height_mm = canvas_mm
    x, y = at_mm
    sx, sy = w / page_mm[0], h / page_mm[1]
    bounds = (round(w / 2 + (x - width_mm / 2) * sx), round(h / 2 - (y + height_mm / 2) * sy),
              round(w / 2 + (x + width_mm / 2) * sx), round(h / 2 - (y - height_mm / 2) * sy))
    left, top, right, bottom = bounds
    result = sheet.copy()
    if right > left and bottom > top:
        layer = ink.convert("RGB").resize((right - left, bottom - top), Image.Resampling.LANCZOS)
        result.paste(ImageChops.multiply(sheet.crop(bounds), layer), (left, top))
    return result


def stencil_for(directory: Path, report: dict, assets: dict, out: Path | None) -> dict:
    page = report.get("page") or {}
    view = "drawn" if "time" in report else "ours"
    prefix = f"{directory.name}-stencil-{view}-"
    if not page.get("fits") or not page.get("stencil") or (out is None and not any(k.startswith(prefix) for k in assets)):
        return {}
    ink_path = directory / ("drawn.png" if view == "drawn" else "target.png")
    if not ink_path.is_file():
        return {}
    try:
        reference, sheet = read_print(Path(page["stencil"]))
    except (OSError, ValueError, KeyError) as error:
        print(f"art wall: {directory.name} stencil preview unavailable ({error})")
        return {}
    page_mm, canvas_mm, at_mm = reference["page_mm"], report["canvas_mm"], page["at_mm"]
    identity = hashlib.sha256(ink_path.read_bytes())
    identity.update(json.dumps([reference["reference_id"], canvas_mm, at_mm]).encode())
    identity.update(sheet.tobytes())
    key = prefix + identity.hexdigest()[:16]
    if out is not None:
        with Image.open(ink_path) as ink:
            image = place_on_stencil(sheet, ink, page_mm, canvas_mm, at_mm)
        out.mkdir(parents=True, exist_ok=True)
        image.save(out / f"{key}.png")
    return {"page_mm": page_mm, "view": view, "url": f"/_blob/{assets[key]}" if key in assets else None}


def read_report(directory: Path) -> dict:
    path = directory / "separation.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def describe(directory: Path, report: dict, catalog: dict, assets: dict, out: Path | None) -> dict:
    fields = {"recipe": recipe(report, catalog), "canvas_mm": report.get("canvas_mm")}
    if stencil := stencil_for(directory, report, assets, out):
        fields["stencil"] = stencil
    return fields


def enrich(designs: dict, corpus: Path, catalog: dict, assets: dict, out: Path | None = None) -> None:
    """Add rendition metadata and exported stencil overlays without creating new voting records."""
    root = corpus / "separations"
    reports = {did: read_report(root / did) for did in designs}
    for did, design in designs.items():
        if report := reports[did]:
            design.update(describe(root / did, report, catalog, assets, out))

    # Non-study alternates are opt-in via the uploaded asset and must come from the same original image.
    for directory in sorted(root.glob("*~*")):
        did, _, label = directory.name.partition("~")
        key = f"{directory.name}-drawn"
        if directory.name in designs or did not in designs or key not in assets or (directory / "study.json").is_file():
            continue
        report = read_report(directory)
        digest = reports[did].get("image_sha256")
        if not digest or digest != report.get("image_sha256") or "time" not in report:
            continue
        designs[did].setdefault("renditions", []).append({
            "id": directory.name, "label": label.replace("-", " "),
            "views": {"drawn": f"/_blob/{assets[key]}"}, **measured(report),
            **describe(directory, report, catalog, assets, out),
        })
