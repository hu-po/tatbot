"""One immutable, physical DBV3 acquisition; no artwork compiler or robot imports."""
from __future__ import annotations

import copy
import math
import platform
import re
import shutil
from pathlib import Path

from tatbot_contracts.artwork import freeze_artwork, validate_source
from tatbot_contracts.paths import freeze_program

from drawingbot.artifacts import REPO, digest, normalize_svg, write_json
from drawingbot.native_export import decode_export
from drawingbot.pens import validate_drawing_set
from drawingbot.recipe import DEFAULTS, read_json

SCHEMA = "tatbot.dbv3-job/3"


def validate_job(job: dict) -> dict:
    required = {"schema", "id", "source", "size_mm", "pen_width_mm", "drawing_set", "pfm", "settings", "state"}
    if not isinstance(job, dict) or set(job) != required or job["schema"] != SCHEMA:
        raise ValueError(f"expected {SCHEMA} with fields {sorted(required)}")
    if not isinstance(job["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", job["id"]):
        raise ValueError("job id must be a bounded lowercase identifier")
    _source(job["source"])
    _dimensions(job)
    _settings(job)
    _state(job["state"])
    validate_drawing_set(job["drawing_set"], job["pen_width_mm"])
    return copy.deepcopy(job)


def _source(source: dict) -> None:
    if not isinstance(source, dict) or set(source) != {"file", "sha256", "provenance"}:
        raise ValueError("source requires file, sha256 and provenance")
    if not isinstance(source["file"], str) or not source["file"] or not isinstance(source["provenance"], dict):
        raise ValueError("source file and provenance are required")
    if not isinstance(source["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
        raise ValueError("source sha256 must identify the exact input bytes")
    validate_source(source["provenance"])


def _dimensions(job: dict) -> None:
    size = job["size_mm"]
    if not isinstance(size, list) or len(size) != 2 or not all(_number(v, 2000) for v in size):
        raise ValueError("size_mm requires two finite dimensions in (0, 2000]")
    if not _number(job["pen_width_mm"], 20):
        raise ValueError("pen_width_mm must be finite and in (0, 20]")


def _settings(job: dict) -> None:
    if not isinstance(job["pfm"], str) or not job["pfm"].strip() or not isinstance(job["settings"], dict):
        raise ValueError("pfm and native settings are required")
    seed = job["settings"].get("Random Seed")
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("settings must explicitly specify an integer Random Seed")
    for key, value in job["settings"].items():
        if not isinstance(key, str) or not key or not isinstance(value, (str, int, float, bool)):
            raise ValueError("settings must contain named scalar values")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("settings must be finite")


def _state(state: dict) -> None:
    defaults = read_json(DEFAULTS / "state.json")
    if not isinstance(state, dict) or set(state) != {"area", "export"}:
        raise ValueError("state requires explicit area and export settings")
    if not isinstance(state["area"], dict) or set(state["area"]) != set(defaults["area"]):
        raise ValueError("state.area requires units, crop, rescale and clipping")
    if state["area"]["units"] != "mm":
        raise ValueError("the physical acquisition boundary uses mm")
    if not isinstance(state["export"], dict) or set(state["export"]) != set(defaults["export"]):
        raise ValueError("all export settings must be explicit")
    for key, value in defaults["export"].items():
        actual = state["export"][key]
        if isinstance(value, bool):
            valid = isinstance(actual, bool)
        elif isinstance(value, (int, float)):
            valid = type(actual) in (int, float) and math.isfinite(actual) and actual >= 0
        else:
            valid = isinstance(actual, str) and bool(actual)
        if not valid:
            raise ValueError(f"invalid export setting {key}")
    if state["export"]["svgLayerNaming"] != "%NAME%":
        raise ValueError("svgLayerNaming must be %NAME% to preserve logical pen IDs")


def _number(value, maximum: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 < value <= maximum


def load_job(path: Path) -> tuple[dict, Path]:
    job = validate_job(read_json(path))
    source = (path.parent / job["source"]["file"]).expanduser().resolve()
    if not source.is_file() or digest(source) != job["source"]["sha256"]:
        raise ValueError("source bytes differ from the job's sha256")
    return job, source


def generate(path: Path, output: Path, app: Path) -> dict:
    """Publish result.json only after native export, readback and dimension checks succeed."""
    from drawingbot.bridge import Bridge

    job, source = load_job(path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "input").mkdir()
    shutil.copy2(source, output / "input/source.png")
    portable = copy.deepcopy(job)
    portable["source"]["file"] = "input/source.png"
    write_json(output / "job.json", portable)
    bridge = None
    try:
        bridge = Bridge(app, output / "worker")
        settings = bridge.export(job, output / "input", output / "raw")
        svg, normalization = normalize_svg((output / "raw/source.svg").read_bytes())
        if any(abs(a - b) > 1e-6 for a, b in zip(normalization["size_mm"], job["size_mm"], strict=True)):
            raise ValueError("exported page dimensions differ from the requested job")
        (output / "normalized.svg").write_text(svg)
        geometry, decoding = decode_export(svg, chord_error_m=.000005, pen_width_m=job["pen_width_mm"] / 1000,
                                          pens=settings["drawing_set"]["pens"])
        decoding["python"] = platform.python_version()
        decoding["sources"] = _decoder_sources()
        program, preview = freeze_program(geometry, source_sha256=job["source"]["sha256"],
                                          name=job["id"], adapter=decoding["adapter"])
        (output / "preview.svg").write_text(preview)
        write_json(output / "decoding.json", decoding)
        write_json(output / "db-settings.json", settings)
        recipe = read_json(output / "recipe/recipe.json")
        recipe["outputs"] = {"normalized.svg": digest(output / "normalized.svg")}
        write_json(output / "recipe/recipe.json", recipe)
        record = freeze_artwork(program, name=job["id"], source=job["source"]["provenance"],
                                conversion={"adapter": decoding["adapter"], "chord_error_m": decoding["chord_error_m"],
                                            "recipe_sha256": digest(output / "recipe/recipe.json")})
        write_json(output / "artwork.json", record)
        result = {"schema": "tatbot.dbv3-acquisition/1", "job_sha256": digest(output / "job.json"),
                  "recipe_sha256": digest(output / "recipe/recipe.json"), "normalization": normalization,
                  "outputs": {name: digest(output / name) for name in ("raw/source.svg", "normalized.svg", "artwork.json", "preview.svg", "decoding.json")}}
        write_json(output / "result.json", result)
        return result
    except BaseException as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if bridge is not None:
            bridge.close()


def _decoder_sources() -> dict:
    files = ["scripts/lib/drawingbot/job.py", "scripts/lib/drawingbot/artifacts.py",
             "scripts/lib/drawingbot/native_export.py", "scripts/lib/drawingbot/native_path.py",
             "scripts/lib/drawingbot/pens.py",
             "python/tatbot_contracts/src/tatbot_contracts/canonical.py",
             "python/tatbot_contracts/src/tatbot_contracts/paths.py",
             "python/tatbot_contracts/src/tatbot_contracts/artwork.py"]
    return {name: digest(REPO / name) for name in files}
