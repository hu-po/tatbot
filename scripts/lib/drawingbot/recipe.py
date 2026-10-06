"""Portable job bundles: native project plus explicit globals, inputs and identities."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from drawingbot.artifacts import digest, write_json
from drawingbot.pens import native_drawing_sets

DEFAULTS = Path(__file__).with_name("defaults")
SCHEMA = "tatbot.dbv3-recipe/3"


def read_json(path: Path):
    return json.loads(path.read_text())


def runtime_project(project: Path, source: Path, destination: Path, *, drawing_set: dict | None = None) -> Path:
    """Resolve the native project's image reference without editing the portable copy."""
    document = read_json(project)
    document["data"]["imagePath"] = str(source.resolve())
    if drawing_set is not None:
        document["data"]["settings"]["drawing_sets"] = native_drawing_sets(drawing_set)
    write_json(destination, document)
    return destination


def save_bundle(folder: Path, project: Path, source: Path, variant: dict, state: dict, settings: dict,
                identity: dict) -> dict:
    folder.mkdir()
    shutil.copy2(source, folder / "source.png")
    document = read_json(project)
    document["data"].update(imagePath="source.png", timeStamp="", thumbnailID="")
    settings_state = document["data"]["settings"]
    settings_state.pop("ui_state", None)
    # A recipe contains generation inputs. Cached geometry and job-local batch
    # directories are acquisition state, not inputs to the next generation.
    settings_state.pop("drawingState", None)
    batch = settings_state.get("batch_processing", {})
    for name in ("inputFolder", "outputFolder"):
        if name in batch:
            batch[name] = ""
    write_json(folder / "project.json", document)
    write_json(folder / "state.json", state)
    variant = copy.deepcopy(variant)
    if isinstance(variant.get("source"), dict):
        variant["source"]["file"] = "source.png"
    recipe = {"schema": SCHEMA, "variant": variant, "software": identity, "effective": settings,
              "widths": {"generation_mm": settings["drawing"]["pen_width_mm"],
                         "pens": [{"id": pen["id"], "generation_width_m": pen["generation_width_m"],
                                   "stroke_factor": pen["stroke_factor"]} for pen in settings["drawing_set"]["pens"]]},
              "files": {name: digest(folder / name) for name in ("source.png", "project.json", "state.json")}}
    write_json(folder / "recipe.json", recipe)
    return recipe


def verify_bundle(folder: Path, identity: dict | None = None, *, migrate_runtime: bool = False) -> dict:
    recipe = read_json(folder / "recipe.json")
    if recipe.get("schema") != SCHEMA:
        raise ValueError("unsupported DBV3 recipe schema")
    files = recipe["files"]
    if not {"source.png", "project.json", "state.json"} <= files.keys():
        raise ValueError("recipe is missing required inputs")
    for name, expected in files.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"recipe input differs or escapes bundle: {name}")
    if identity:
        names = ("jar_sha256", "bridge_sha256")
        if not migrate_runtime:
            names += ("java_version", "runtime_flags")
        for name in names:
            if recipe["software"].get(name) != identity.get(name):
                raise ValueError(f"recipe requires its recorded {name}; create a new acquisition to migrate")
    return recipe
