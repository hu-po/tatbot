"""The shared artwork envelope for acquired drawing paths; no SVG reconstruction."""
from __future__ import annotations

import copy
import math
import re

from tatbot_contracts.canonical import canonical_digest
from tatbot_contracts.paths import validate_path_program

SCHEMA = "tatbot.inkmap-artwork/2"
FIELDS = {"schema", "content_sha256", "name", "source_sha256", "source", "conversion", "program"}


def _keys(value: dict, fields: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(value, dict) or set(value) - optional != fields:
        raise ValueError(f"artwork requires fields {sorted(fields)}")


def _text(value, *, nullable=False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip() or len(value) > 100_000:
        raise ValueError("artwork requires bounded nonempty text")


def _sha(value) -> None:
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError("artwork requires lowercase SHA-256")


def validate_source(source: dict) -> None:
    _keys(source, {"kind", "identifier", "license", "attribution", "generation"})
    if source["kind"] not in ("stock", "imported", "generated", "fixture"):
        raise ValueError("unknown artwork source kind")
    for key in ("identifier", "license", "attribution"):
        _text(source[key], nullable=True)
    generation = source["generation"]
    if source["kind"] != "generated":
        if generation is not None:
            raise ValueError("only generated artwork carries generation metadata")
        return
    _keys(generation, {"prompt", "model", "model_revision", "seed", "tracing"}, {"request_sha256", "png_sha256", "settings"})
    _reported_generation(generation)
    _text(generation["prompt"])
    for key in ("model", "model_revision", "tracing"):
        _text(generation[key], nullable=True)
    seed = generation["seed"]
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**53 - 1):
        raise ValueError("artwork generation seed must be a nonnegative safe integer or null")


def _reported_generation(generation: dict) -> None:
    for key in ("request_sha256", "png_sha256"):
        if key in generation:
            _sha(generation[key])
    if "settings" not in generation:
        return
    settings = generation["settings"]
    _keys(settings, {"model", "model_revision", "width", "height", "steps", "guidance"})
    _text(settings["model"])
    _text(settings["model_revision"], nullable=True)
    for key in ("width", "height", "steps", "guidance"):
        if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]):
            raise ValueError("reported generation settings must be finite numbers")


def validate_conversion(conversion: dict) -> None:
    _keys(conversion, {"adapter", "recipe_sha256", "chord_error_m"})
    _text(conversion["adapter"])
    if conversion["recipe_sha256"] is not None:
        _sha(conversion["recipe_sha256"])
    error = conversion["chord_error_m"]
    if type(error) not in (float, int) or not math.isfinite(error) or not 0 < error <= .0001:
        raise ValueError("artwork chord error must be in (0, 0.1 mm]")


def freeze_artwork(program: dict, *, name: str, source: dict, conversion: dict) -> dict:
    record = {"schema": SCHEMA, "name": name, "source_sha256": program["provenance"]["source_sha256"],
              "source": source, "conversion": conversion, "program": program}
    record["content_sha256"] = canonical_digest(record)
    return validate_path_artwork(record)


def validate_artwork_metadata(value: dict) -> None:
    """Check the shared envelope; callers also validate the program geometry they consume."""
    _keys(value, FIELDS)
    if value["schema"] != SCHEMA:
        raise ValueError("unsupported artwork schema; acquire a current artwork record")
    _text(value["name"])
    _sha(value["source_sha256"])
    validate_source(value["source"])
    validate_conversion(value["conversion"])
    if value["program"]["provenance"]["source_sha256"] != value["source_sha256"]:
        raise ValueError("artwork and program bind different source bytes")
    if value["content_sha256"] != canonical_digest(value):
        raise ValueError("artwork digest mismatch")
    canvas = value["program"]["canvas_m"]
    if any(not 0 < canvas[key] <= 2 for key in ("width", "height")):
        raise ValueError("artwork canvas exceeds 2 m")
    if max_width_m(value) > .02:
        raise ValueError("artwork planning width exceeds 20 mm")


def validate_path_artwork(value: dict) -> dict:
    _keys(value, FIELDS)
    validate_path_program(value["program"])
    validate_artwork_metadata(value)
    return copy.deepcopy(value)


def canvas_m(record: dict) -> list[float]:
    return [record["program"]["canvas_m"][key] for key in ("width", "height")]


def max_width_m(record: dict) -> float:
    return max(element["width_m"] for layer in record["program"]["layers"] for element in layer["elements"])


def require_acquired_artwork(record: dict) -> dict:
    """Production admission; generic source/fixture paint records are not finished artwork."""
    if (not isinstance(record, dict) or record.get("conversion", {}).get("adapter") != "dbv3-batik-paths/1"
            or record["conversion"].get("recipe_sha256") is None
            or record.get("program", {}).get("provenance", {}).get("producer") != "dbv3-batik-paths/1"):
        raise ValueError("Acquired DBV3 artwork is required. Generate with DrawingBot V3 and import artwork.json; legacy artwork requires regeneration")
    return validate_path_artwork(record)
