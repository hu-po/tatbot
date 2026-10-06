#!/usr/bin/env python3
"""Render inventory-driven replacement tags at 300 DPI.

Print at 100% scale and caliper the black square before use. Retain the white
quiet zone (one module on each side), remove old copies, and recalibrate every
remounted target. Labels identify intended targets, never measured orientations.
"""

import argparse
import json
import struct
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fiducials import load_inventory
from fiducials.detector import tag_dictionary
from PIL import Image

DPI = 300
MM = DPI / 25.4
VIEWER_TEXTURE_MODULE_PX = 100


def render_sheet(target, ids):
    dictionary = tag_dictionary(target.family)
    modules = dictionary.markerSize + 2
    tag_mm = target.edge_m * 1000
    tag_px = round(tag_mm * MM)
    margin_px = round(tag_px / modules)
    label_px = round(10 * MM)
    cell = tag_px + 2 * margin_px
    sheet = np.full((len(ids) * (cell + label_px) + label_px,
                     cell + 2 * label_px), 255, np.uint8)
    for row, tag_id in enumerate(ids):
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, modules * 120)
        marker = cv2.resize(marker, (tag_px, tag_px), interpolation=cv2.INTER_NEAREST)
        y = label_px + row * (cell + label_px) + margin_px
        x = label_px + margin_px
        sheet[y:y + tag_px, x:x + tag_px] = marker
        label = f"{target.family.removeprefix('apriltag_')} {tag_id} | black {tag_mm:g} mm | 100%"
        font_scale = min(1.6, (sheet.shape[1] - 2 * label_px) /
                         cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0])
        cv2.putText(sheet, label, (label_px, y + tag_px + margin_px + round(4 * MM)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, 2)
    return sheet


def viewer_asset_name(target, tag_id):
    family = target.family.removeprefix("apriltag_")
    edge_mm = target.edge_m * 1000
    if not float(edge_mm).is_integer():
        raise ValueError(f"viewer assets require an integer-mm black edge, got {edge_mm:g}")
    return f"{family}_{tag_id:03d}_{int(edge_mm)}mm"


def render_viewer_texture(target, tag_id):
    """Marker plus its one-module white quiet zone, matching the print contract."""
    dictionary = tag_dictionary(target.family)
    black_modules = dictionary.markerSize + 2
    marker_px = black_modules * VIEWER_TEXTURE_MODULE_PX
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_px)
    texture = np.full(
        (marker_px + 2 * VIEWER_TEXTURE_MODULE_PX,) * 2,
        255,
        np.uint8,
    )
    margin = VIEWER_TEXTURE_MODULE_PX
    texture[margin:-margin, margin:-margin] = marker
    return texture, black_modules


def write_viewer_asset(target, tag_id, assets_dir):
    texture, black_modules = render_viewer_texture(target, tag_id)
    output = assets_dir / viewer_asset_name(target, tag_id)
    output.mkdir(parents=True, exist_ok=True)
    full_edge_m = target.edge_m * (black_modules + 2) / black_modules
    half = full_edge_m / 2
    encoded = BytesIO()
    Image.fromarray(texture).save(encoded, format="PNG")
    png = encoded.getvalue()
    # The quad lies in the tag frame the detector solves (fiducials.geometry
    # TAG_CORNER_SIGNS): +x toward the pattern's right edge, +y toward its top
    # edge, +z out of the face. glTF puts texture coordinate (0, 0) at the
    # image's TOP-left (unlike OBJ's bottom-left), so the image's top row must
    # map to the +y vertices — otherwise the viewer shows every tag mirrored.
    positions = struct.pack(
        "<12f",
        -half, -half, 0, -half, half, 0,
        half, -half, 0, half, half, 0,
    )
    uvs = struct.pack("<8f", 0, 1, 0, 0, 1, 1, 1, 0)
    indices = struct.pack("<6H", 0, 2, 3, 0, 3, 1)
    png_offset = len(positions) + len(uvs) + len(indices)
    binary = positions + uvs + indices + png
    binary += b"\0" * (-len(binary) % 4)
    gltf = {
        "asset": {"version": "2.0", "generator": "tatbot generate_wrist_tags.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
            "indices": 2,
            "material": 0,
            "mode": 4,
        }]}],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0,
                "roughnessFactor": 1,
            },
            "doubleSided": True,
        }],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(uvs), "target": 34962},
            {"buffer": 0, "byteOffset": len(positions) + len(uvs), "byteLength": len(indices), "target": 34963},
            {"buffer": 0, "byteOffset": png_offset, "byteLength": len(png)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [-half, -half, 0], "max": [half, half, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2",
             "min": [0, 0], "max": [1, 1]},
            {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR",
             "min": [0], "max": [3]},
        ],
    }
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    glb = (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )
    (output / "tag.glb").write_bytes(glb)
    (output / "tag.png").write_bytes(png)
    print(f"wrote {output}: {target.family} id {tag_id}, black edge {target.edge_m * 1000:g} mm")


def main():
    inventory = load_inventory()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', choices=sorted(inventory.targets), default='wrist')
    ap.add_argument('--ids', help='comma-separated configured target or spare ids')
    ap.add_argument('--output', type=Path)
    ap.add_argument(
        '--viewer-assets',
        action='store_true',
        help='write textured glTF assets under urdf/meshes/tags instead of a print sheet',
    )
    args = ap.parse_args()
    target = inventory.target(args.target)
    ids = tuple(int(v) for v in args.ids.split(',')) if args.ids else target.ids
    spares = inventory.spare_ids if target.family == inventory.family else ()
    if len(set(ids)) != len(ids) or set(ids) - (set(target.ids) | set(spares)):
        ap.error('ids must be unique configured target or same-family spare ids')
    if args.viewer_assets:
        if args.output and args.output.suffix:
            ap.error('--viewer-assets --output must name a directory')
        assets = args.output or Path(__file__).resolve().parents[2] / 'urdf/meshes/tags'
        for tag_id in ids:
            write_viewer_asset(target, tag_id, assets)
        return
    output = args.output or Path(__file__).resolve().parents[2] / 'docs' / (
        f'{args.target}-tags-{target.family.removeprefix("apriltag_")}.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_sheet(target, ids)).save(output, dpi=(DPI, DPI))
    print(f'wrote {output}: 300 DPI / 100%, caliper black square {target.edge_m * 1000:g} mm')


if __name__ == '__main__':
    main()
