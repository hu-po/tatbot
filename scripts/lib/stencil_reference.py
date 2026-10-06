"""Versioned reference identity and page coordinates for generated stencils.

A pattern identity distinguishes artwork, not two physical copies of that
artwork. Dimensions describe the requested print and are not a measurement.

A coded flower-of-life print (`stencil_coded.py`) carries its print ID in its
own lattice bits; the code lives in `coded.json` beside the artwork. The
manifest binds that file by hash, so a coded reference loads only with the
code it was exported with, and never as plain artwork.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import stencil_instance
import tatbot_runlog
from tatbot_digest import sha256_file as digest

SCHEMA = "tatbot.stencil-reference/1"
# Where this node's stencil observer reads references, under the log root:
# one directory per pattern holding tracking.json beside its stencil.png
# (and a coded print's coded.json).
OBSERVER_REFERENCES = "stencils/references"
# The coded generator's version and scheme (stencil_coded.VERSION, .SCHEME),
# repeated here so this module stays stdlib-only; a test pins them together.
CODED_GENERATORS = frozenset({"coded-fol-1"})
CODED_SCHEME = "tatbot.stencil-coded-fol/1"
CODED_FILE = "coded.json"
SETTINGS_FILE = "settings.json"
MAX_CODED_BYTES = 256_000


def reference_digest(value):
    payload = {key: item for key, item in value.items() if key != "reference_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def is_coded(value):
    """Whether a manifest or generator settings describe a coded flower-of-life print."""
    generator = value.get("generator", value).get("generator_version")
    return generator in CODED_GENERATORS or "coded" in value or "coded_scheme" in value


def build(settings, coded=None):
    """The manifest for generated artwork; `coded` is `(code, coded.json sha256)` for a coded print."""
    width, height = settings["width_mm"], settings["height_mm"]
    pixels = settings["pixels"]
    if not all(math.isfinite(v) and v > 0 for v in (width, height, *pixels)):
        raise ValueError("reference dimensions must be positive and finite")
    if is_coded(settings) != (coded is not None):
        raise ValueError("a coded print's reference is built with its coded.json, and only then")
    identity = {"generator_version": settings["generator_version"],
                "svg_sha256": settings.get("artwork_svg_sha256", settings["files"]["stencil.svg"])}
    pattern = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result = {
        "schema": SCHEMA, "pattern_id": "stencil-" + pattern,
        "seed": settings["seed"], "generator": identity,
        "image": {"file": "stencil.png", "sha256": settings["files"]["stencil.png"],
                  "width": pixels[0], "height": pixels[1]},
        "page_mm": [width, height], "dimensions_measured": False,
        "coordinates": {"origin": "page_top_left", "u_direction": "right", "v_direction": "down",
                        "domain": "whole_page", "pixel_centers": "uv=((x+0.5)/width,(y+0.5)/height)",
                        "uv_to_print_mm": [[width, 0, 0], [0, height, 0], [0, 0, 1]]},
        "clear_center_uv": [(settings["margin_mm"] + settings["frame_mm"])/width,
                            (settings["margin_mm"] + settings["frame_mm"])/height,
                            1-(settings["margin_mm"] + settings["frame_mm"])/width,
                            1-(settings["margin_mm"] + settings["frame_mm"])/height],
        "physical_instance_encoded": settings.get("instance_mark") is not None or coded is not None,
    }
    if coded is not None:
        code, sha256 = coded
        result["physical_instance_id"] = code["print_id"]
        result["coded"] = {"file": CODED_FILE, "sha256": sha256, "scheme": code["scheme"],
                           "print_id": code["print_id"]}
    elif result["physical_instance_encoded"]:
        result["physical_instance_id"] = settings["physical_instance_id"]
        result["instance_mark"] = settings["instance_mark"]
    result["reference_id"] = reference_digest(result)
    return result


def _read_code(path):
    """A coded.json's bytes and parsed code, bounded and of the known scheme."""
    if path.stat().st_size > MAX_CODED_BYTES:
        raise ValueError("coded.json exceeds size limit")
    raw = path.read_bytes()
    code = json.loads(raw)
    if not isinstance(code, dict) or code.get("scheme") != CODED_SCHEME:
        raise ValueError("unsupported coded stencil scheme")
    if not stencil_instance.valid_id(code.get("print_id")):
        raise ValueError("coded.json carries an invalid print ID")
    return raw, code


def _exported_code(settings, directory):
    """(code, sha256) of the coded.json beside generated settings. Artwork generated before
    settings recorded the code's hash is bound by its print ID, geometry and lattice size."""
    raw, code = _read_code(directory/CODED_FILE)
    sha256 = hashlib.sha256(raw).hexdigest()
    recorded = settings["files"].get(CODED_FILE)
    if recorded is not None and recorded != sha256:
        raise ValueError("artwork hash mismatch: coded.json")
    geometry = code.get("geometry", {})
    if (code["scheme"] != settings.get("coded_scheme") or code["print_id"] != settings.get("coded_print_id")
            or any(geometry.get(key) != settings.get(key) for key in
                   ("width_mm", "height_mm", "margin_mm", "frame_mm", "spacing_mm", "stroke_mm", "knot_mm"))
            or len(code.get("nodes", ())) != settings.get("junctions")
            or len(code.get("edges", ())) != settings.get("coded_edges")):
        raise ValueError("coded.json does not describe this artwork")
    return code, sha256


def export(settings_path, output=None):
    settings_path = Path(settings_path).expanduser().resolve()
    settings = json.loads(settings_path.read_text())
    coded = _exported_code(settings, settings_path.parent) if is_coded(settings) else None
    reference = build(settings, coded)
    for name in ("stencil.png", "stencil.svg"):
        if digest(settings_path.parent/name) != settings["files"][name]:
            raise ValueError(f"artwork hash mismatch: {name}")
    if settings.get('instance_mark') is not None:
        svg = (settings_path.parent/'stencil.svg').read_text()
        prefix, separator, remainder = svg.partition('<g id="tatbot-instance-mark">\n')
        expected = ('\n'.join(stencil_instance.svg_elements(
            settings['instance_mark'], settings['physical_instance_id'])) + '\n</g>\n</svg>\n')
        if not separator or not prefix.endswith('</g>\n') or remainder != expected:
            raise ValueError('printed SVG instance mark is missing')
        artwork = prefix + '</svg>\n'
        if hashlib.sha256(artwork.encode()).hexdigest() != settings.get('artwork_svg_sha256'):
            raise ValueError('pre-mark artwork hash mismatch')
    elif ('artwork_svg_sha256' in settings
          and settings['artwork_svg_sha256'] != settings['files']['stencil.svg']):
        raise ValueError('unmarked artwork hash mismatch')
    destination = Path(output) if output else settings_path.with_name("tracking.json")
    if destination.parent.resolve() != settings_path.parent:
        raise ValueError("tracking manifest must remain beside its artwork")
    destination.write_text(json.dumps(reference, sort_keys=True, indent=2) + "\n")
    return reference


def _validate_instance(value):
    encoded = value.get('physical_instance_encoded')
    if type(encoded) is not bool:
        raise ValueError('invalid print-instance qualification')
    if 'coded' in value:
        coded = value['coded']
        if (not encoded or 'instance_mark' in value or not isinstance(coded, dict)
                or set(coded) != {'file', 'sha256', 'scheme', 'print_id'}
                or coded['file'] != CODED_FILE or coded['scheme'] != CODED_SCHEME
                or not stencil_instance.valid_id(value.get('physical_instance_id'))
                or coded['print_id'] != value['physical_instance_id']):
            raise ValueError('invalid coded print identity')
    elif encoded:
        if not stencil_instance.valid_id(value.get('physical_instance_id')):
            raise ValueError('invalid physical print-instance ID')
        stencil_instance.validate(value.get('instance_mark'), value['page_mm'])
    elif 'physical_instance_id' in value or 'instance_mark' in value:
        raise ValueError('unmarked reference cannot claim a print-instance ID')


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
    _validate_instance(value)
    if not all(isinstance(v, int) and 32 <= v <= 12000 for v in (image["width"], image["height"])):
        raise ValueError("invalid reference raster dimensions")
    if value["generator"].get("generator_version") in CODED_GENERATORS or "coded" in value:
        _check_code(value, path.parent/CODED_FILE)
    return value, image_path


def _check_code(value, code_path):
    """A coded reference loads only beside the coded.json it was exported with."""
    if "coded" not in value:
        raise ValueError("coded reference predates its coded.json binding; re-export it with "
                         "`tatbot vision stencil reference --settings <artwork>/settings.json`")
    if value["generator"].get("generator_version") not in CODED_GENERATORS:
        raise ValueError("a coded reference names an unknown coded generator")
    if not code_path.is_file():
        raise ValueError("coded reference without its coded.json")
    raw, code = _read_code(code_path)
    if hashlib.sha256(raw).hexdigest() != value["coded"]["sha256"]:
        raise ValueError("coded.json hash mismatch")
    geometry = code.get("geometry", {})
    if (code["print_id"] != value["coded"]["print_id"]
            or [geometry.get("width_mm"), geometry.get("height_mm")] != value["page_mm"]):
        raise ValueError("coded.json does not describe this reference")


def coded_path(manifest_path):
    """The coded.json a loaded coded reference was verified against."""
    return Path(manifest_path).expanduser().resolve().parent/CODED_FILE


def observer_references(root=None):
    """The observer's reference directory on this node."""
    return (Path(root) if root else tatbot_runlog.log_root()) / OBSERVER_REFERENCES


def page_geometry(manifest, settings=None):
    """A print's page in the drawing stack's page frame (ros/README.md section 3: metres, origin at the page
    centre, x right, y toward the top of the print): `size_m`, the nominal `clear_m` and, from the generator's
    settings.json, `inner_edges_m`, each side's innermost border ink (x of left and right, y of bottom and top).
    Settings that are not this print's artwork are ignored."""
    width, height = (v / 1000 for v in manifest["page_mm"])
    u0, v0, u1, v1 = manifest["clear_center_uv"]
    page = {"pattern_id": manifest["pattern_id"], "size_m": [width, height],
            "clear_m": [round((u1 - u0) * width, 9), round((v1 - v0) * height, 9)]}
    settings = settings or {}
    artwork = settings.get("artwork_svg_sha256") or (settings.get("files") or {}).get("stencil.svg")
    if settings.get("border_inner_mm") and artwork == manifest["generator"]["svg_sha256"]:
        x0, y0, x1, y1 = (v / 1000 for v in settings["border_inner_mm"])
        page["inner_edges_m"] = {side: round(value, 9) for side, value in (
            ("left", x0 - width / 2), ("right", x1 - width / 2), ("bottom", height / 2 - y1), ("top", height / 2 - y0))}
    return page


def installed_page(pattern_id, root=None):
    """page_geometry of the print installed for the observer as `pattern_id`; None when none is."""
    directory = observer_references(root) / pattern_id
    try:
        manifest = json.loads((directory / "tracking.json").read_text())
    except (OSError, ValueError):
        return None
    try:
        settings = json.loads((directory / SETTINGS_FILE).read_text())
    except (OSError, ValueError):
        settings = None
    return page_geometry(manifest, settings)


def install(manifest_path, root=None):
    """Install a validated reference where the observer reads it. The image,
    a coded print's coded.json and the generator's settings.json (the drawing
    stack's page geometry, the pen-tip fit's lattice) land first and the
    manifest last, each atomically, so the observer's mtime scan never pairs a
    manifest with files it does not describe; a same-content install touches
    nothing, and a changed one withdraws the manifest before the other files
    change, so the print reads as absent, never as a mismatch. Returns the
    installed directory."""
    reference, image_path = load(manifest_path)
    destination = observer_references(root) / reference["pattern_id"]
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "tracking.json"
    settings = Path(manifest_path).expanduser().resolve().parent / SETTINGS_FILE
    if manifest.is_file():
        try:
            if load(manifest)[0]["reference_id"] == reference["reference_id"] and (
                    not settings.is_file() or (destination / SETTINGS_FILE).is_file()):
                return destination
        except (ValueError, OSError, KeyError):
            pass
        manifest.unlink()
    staged = destination / f".stencil.png.{os.getpid()}"
    shutil.copyfile(image_path, staged)
    os.replace(staged, destination / "stencil.png")
    if "coded" in reference:
        raw = coded_path(manifest_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["coded"]["sha256"]:
            raise ValueError("coded.json changed while it was being installed")
        staged = destination / f".{CODED_FILE}.{os.getpid()}"
        staged.write_bytes(raw)
        os.replace(staged, destination / CODED_FILE)
    elif (destination / CODED_FILE).exists():
        (destination / CODED_FILE).unlink()
    if settings.is_file():
        staged = destination / f".{SETTINGS_FILE}.{os.getpid()}"
        shutil.copyfile(settings, staged)
        os.replace(staged, destination / SETTINGS_FILE)
    elif (destination / SETTINGS_FILE).exists():
        (destination / SETTINGS_FILE).unlink()
    staged = destination / f".tracking.json.{os.getpid()}"
    staged.write_text(json.dumps(reference, sort_keys=True, indent=2) + "\n")
    os.replace(staged, manifest)
    return destination


def add_arguments(parser):
    parser.add_argument("--settings", required=True, help="existing generated settings.json beside PNG and SVG")
    parser.add_argument("--install", action="store_true",
                        help="also install the manifest and its artwork (and a coded print's coded.json) where "
                             "this node's stencil observer reads references "
                             "(<log root>/stencils/references/<pattern_id>/); the observer rescans by mtime, "
                             "so no service restarts")


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
