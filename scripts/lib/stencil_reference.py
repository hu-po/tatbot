"""Versioned reference identity and page coordinates for generated stencils.

A pattern identity distinguishes artwork, not two physical copies of that
artwork. Dimensions describe the requested print and are not a measurement.

The stencil is the fiducial tile (`stencil_tile.py`), named by its layout:
`tile.json`, every marker's code and printed corners, bound by hash in the
manifest, so a reference loads only beside the layout it was exported with.
Its page is the tile, all of it clear: the design is drawn over the markers.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import tatbot_runlog
from tatbot_digest import sha256_file as digest

SCHEMA = "tatbot.stencil-reference/1"
# Where this node's stencil observer reads references, under the log root:
# one directory per pattern holding tracking.json beside its stencil.png and tile.json.
OBSERVER_REFERENCES = "stencils/references"
SETTINGS_FILE = "settings.json"
# stencil_tile.GENERATOR, repeated here so this module stays stdlib-only; a test pins them together.
TILE_GENERATORS = frozenset({"tile-16h5-1"})
TILE_FILE = "tile.json"


def reference_digest(value):
    payload = {key: item for key, item in value.items() if key != "reference_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def is_tile(value):
    """Whether a manifest or generator settings describe a fiducial-tile print."""
    return value.get("generator", value).get("generator_version") in TILE_GENERATORS


def build(settings):
    """The manifest for a generated tile print's settings."""
    if not is_tile(settings):
        raise ValueError("only a fiducial-tile print has a stencil reference")
    width, height = settings["width_mm"], settings["height_mm"]
    pixels = settings["pixels"]
    if not all(math.isfinite(v) and v > 0 for v in (width, height, *pixels)):
        raise ValueError("reference dimensions must be positive and finite")
    identity = {"generator_version": settings["generator_version"], "tile_sha256": settings["files"][TILE_FILE]}
    pattern = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    margin = settings["margin_mm"] + settings["frame_mm"]
    result = {
        "schema": SCHEMA, "pattern_id": "stencil-" + pattern,
        "seed": settings["seed"], "generator": identity,
        "image": {"file": "stencil.png", "sha256": settings["files"]["stencil.png"],
                  "width": pixels[0], "height": pixels[1]},
        "page_mm": [width, height], "dimensions_measured": False,
        "coordinates": {"origin": "page_top_left", "u_direction": "right", "v_direction": "down",
                        "domain": "whole_page", "pixel_centers": "uv=((x+0.5)/width,(y+0.5)/height)",
                        "uv_to_print_mm": [[width, 0, 0], [0, height, 0], [0, 0, 1]]},
        "clear_center_uv": [margin/width, margin/height, 1-margin/width, 1-margin/height],
        "physical_instance_encoded": False,
        "tile": {"file": TILE_FILE, "sha256": settings["files"][TILE_FILE], "family": settings["family"]},
    }
    result["reference_id"] = reference_digest(result)
    return result


def export(settings_path, output=None):
    settings_path = Path(settings_path).expanduser().resolve()
    settings = json.loads(settings_path.read_text())
    reference = build(settings)
    for name in ("stencil.png", TILE_FILE):
        if digest(settings_path.parent/name) != settings["files"][name]:
            raise ValueError(f"artwork hash mismatch: {name}")
    destination = Path(output) if output else settings_path.with_name("tracking.json")
    if destination.parent.resolve() != settings_path.parent:
        raise ValueError("tracking manifest must remain beside its artwork")
    destination.write_text(json.dumps(reference, sort_keys=True, indent=2) + "\n")
    return reference


def load(path):
    path = Path(path).expanduser().resolve()
    if path.stat().st_size > 64_000:
        raise ValueError("reference manifest exceeds size limit")
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported stencil reference schema")
    if value.get("reference_id") != reference_digest(value):
        raise ValueError("reference metadata hash mismatch")
    image = value["image"]
    if image["file"] != "stencil.png":
        raise ValueError("reference image must be the adjacent stencil.png")
    image_path = path.parent/image["file"]
    if image_path.stat().st_size > 32_000_000 or digest(image_path) != image["sha256"]:
        raise ValueError("reference image hash or size mismatch")
    identity = value["generator"]
    expected = "stencil-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if value["pattern_id"] != expected:
        raise ValueError("reference pattern identity mismatch")
    if not all(isinstance(v, int) and 32 <= v <= 12000 for v in (image["width"], image["height"])):
        raise ValueError("invalid reference raster dimensions")
    if not is_tile(value) or value.get("tile", {}).get("file") != TILE_FILE:
        raise ValueError("not a fiducial-tile print (the coded and floral prints are retired)")
    sha256 = value["tile"]["sha256"]
    tile_path = path.parent/TILE_FILE
    if sha256 != value["generator"]["tile_sha256"] or not tile_path.is_file() or digest(tile_path) != sha256:
        raise ValueError("tile.json hash mismatch")
    return value, image_path


def observer_references(root=None):
    """The observer's reference directory on this node."""
    return (Path(root) if root else tatbot_runlog.log_root()) / OBSERVER_REFERENCES


def page_geometry(manifest):
    """A print's page in the drawing stack's page frame (ros/README.md section 3: metres, origin at the page
    centre, x right, y toward the top of the print): `size_m` and the nominal `clear_m`."""
    width, height = (v / 1000 for v in manifest["page_mm"])
    u0, v0, u1, v1 = manifest["clear_center_uv"]
    return {"pattern_id": manifest["pattern_id"], "size_m": [width, height],
            "clear_m": [round((u1 - u0) * width, 9), round((v1 - v0) * height, 9)]}


def installed_page(pattern_id, root=None):
    """page_geometry of the print installed for the observer as `pattern_id`; None when none is."""
    try:
        manifest = json.loads((observer_references(root) / pattern_id / "tracking.json").read_text())
    except (OSError, ValueError):
        return None
    return page_geometry(manifest)


def _copy(source, destination, name, sha256=None):
    """One file into the installed directory, staged then replaced atomically, its hash checked when given."""
    staged = destination / f".{name}.{os.getpid()}"
    shutil.copyfile(source, staged)
    if sha256 is not None and digest(staged) != sha256:
        staged.unlink()
        raise ValueError(f"{name} changed while it was being installed")
    os.replace(staged, destination / name)


def install(manifest_path, root=None):
    """Install a validated reference where the observer reads it. The image, tile.json and the generator's
    settings.json land first and the manifest last, each atomically, so the observer's mtime scan never pairs a
    manifest with files it does not describe; a same-content install touches nothing, and a changed one withdraws
    the manifest before the other files change, so the print reads as absent, never as a mismatch. Returns the
    installed directory."""
    reference, image_path = load(manifest_path)
    source = Path(manifest_path).expanduser().resolve().parent
    destination = observer_references(root) / reference["pattern_id"]
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "tracking.json"
    if manifest.is_file():
        try:
            if load(manifest)[0]["reference_id"] == reference["reference_id"]:
                return destination
        except (ValueError, OSError, KeyError):
            pass
        manifest.unlink()
    _copy(image_path, destination, "stencil.png")
    _copy(source / TILE_FILE, destination, TILE_FILE, reference["tile"]["sha256"])
    if (source / SETTINGS_FILE).is_file():
        _copy(source / SETTINGS_FILE, destination, SETTINGS_FILE)
    staged = destination / f".tracking.json.{os.getpid()}"
    staged.write_text(json.dumps(reference, sort_keys=True, indent=2) + "\n")
    os.replace(staged, manifest)
    return destination


def add_arguments(parser):
    parser.add_argument("--settings", required=True, help="existing generated settings.json beside stencil.png and tile.json")
    parser.add_argument("--install", action="store_true",
                        help="also install the manifest, its artwork and tile.json where this node's stencil observer "
                             "reads references (<log root>/stencils/references/<pattern_id>/); the observer rescans "
                             "by mtime, so no service restarts")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        reference = export(args.settings)
        if args.install:
            installed = install(Path(args.settings).expanduser().resolve().with_name("tracking.json"))
            reference = {"reference": reference, "installed": str(installed)}
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(reference, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
