#!/usr/bin/env python3
"""Embed the installed printed-material color in the palette viewer GLB.

Rerun 0.36 preserves a GLB's material instead of applying Asset3D's fallback
albedo. The palette is a single-material PETG-CF print, so make that material
part of the self-contained asset rather than depending on viewer behavior.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

GLB_MAGIC = b"glTF"
JSON_CHUNK = b"JSON"
PRINTED_MOUNT_RGBA = (31, 31, 33, 255)
MATERIAL_NAME = "Tatbot printed mount black"


def _pad_four(data: bytes, pad: bytes) -> bytes:
    return data + pad * ((-len(data)) % 4)


def _glb_chunks(blob: bytes) -> list[tuple[bytes, bytes]]:
    if len(blob) < 20:
        raise ValueError("asset is too short to be a GLB")
    magic, version, total = struct.unpack_from("<4sII", blob)
    if magic != GLB_MAGIC or version != 2 or total != len(blob):
        raise ValueError("asset is not a complete glTF binary v2 file")

    chunks: list[tuple[bytes, bytes]] = []
    offset = 12
    while offset < len(blob):
        if offset + 8 > len(blob):
            raise ValueError("truncated GLB chunk header")
        length, kind = struct.unpack_from("<I4s", blob, offset)
        offset += 8
        end = offset + length
        if end > len(blob):
            raise ValueError("truncated GLB chunk")
        chunks.append((kind, blob[offset:end]))
        offset = end
    if offset != len(blob) or not chunks or chunks[0][0] != JSON_CHUNK:
        raise ValueError("GLB must begin with one JSON chunk")
    return chunks


def _add_printed_material(gltf: dict) -> None:
    if gltf.get("asset", {}).get("version") != "2.0":
        raise ValueError("embedded glTF is not version 2.0")
    if gltf.get("images") or gltf.get("textures"):
        raise ValueError("palette body asset unexpectedly contains textures")
    meshes = gltf.get("meshes", [])
    primitives = [primitive for mesh in meshes for primitive in mesh.get("primitives", [])]
    if not primitives:
        raise ValueError("palette body asset has no mesh primitives")

    red, green, blue, alpha = (channel / 255.0 for channel in PRINTED_MOUNT_RGBA)
    gltf["materials"] = [{
        "name": MATERIAL_NAME,
        "pbrMetallicRoughness": {
            "baseColorFactor": [red, green, blue, alpha],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.8,
        },
        "doubleSided": True,
    }]
    for primitive in primitives:
        primitive["material"] = 0


def prepare_glb(blob: bytes) -> bytes:
    """Return *blob* with one opaque near-black material on every primitive."""
    chunks = _glb_chunks(blob)
    gltf = json.loads(chunks[0][1].rstrip(b" \t\r\n\0"))
    _add_printed_material(gltf)

    encoded = _pad_four(
        json.dumps(gltf, sort_keys=True, separators=(",", ":")).encode(),
        b" ",
    )
    chunks[0] = (JSON_CHUNK, encoded)
    body = b"".join(struct.pack("<I4s", len(data), kind) + data for kind, data in chunks)
    return struct.pack("<4sII", GLB_MAGIC, 2, 12 + len(body)) + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that rewriting is already byte-for-byte stable",
    )
    args = parser.parse_args()
    original = args.asset.read_bytes()
    prepared = prepare_glb(original)
    if args.check:
        if prepared != original:
            raise SystemExit(f"palette viewer material is not canonical: {args.asset}")
        return 0
    temporary = args.asset.with_suffix(args.asset.suffix + ".next")
    temporary.write_bytes(prepared)
    temporary.replace(args.asset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
