"""Replay one immutable native acquisition; independent of robot preparation."""
from __future__ import annotations

import shutil
from pathlib import Path

from drawingbot.artifacts import digest, normalize_svg, write_json
from drawingbot.bridge import Bridge
from drawingbot.recipe import read_json, verify_bundle


def replay(bundle: Path, output: Path, app: Path, *, migrate_runtime: bool = False) -> dict:
    """Replay one immutable bundle; export-only is also the container acquisition boundary."""
    recipe = verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    (output / "input").mkdir()
    shutil.copy2(bundle / "source.png", output / "input/source.png")
    bridge = Bridge(app, output / "worker")
    try:
        settings = bridge.export(recipe["variant"], output / "input", output / "raw", replay=bundle,
                                 migrate_runtime=migrate_runtime)
    finally:
        bridge.close()
    write_json(output / "db-settings.json", settings)
    result = {"recipe_sha256": digest(bundle / "recipe.json"), "raw_sha256": digest(output / "raw/source.svg")}
    captured = read_json(output / "recipe/recipe.json")
    captured["parent_recipe_sha256"] = result["recipe_sha256"]
    if migrate_runtime:
        captured["runtime_migration"] = {"from": recipe["software"], "to": captured["software"]}
        result["runtime_migration"] = captured["runtime_migration"]
    svg, normalization = normalize_svg((output / "raw/source.svg").read_bytes())
    (output / "normalized.svg").write_text(svg)
    captured["outputs"] = {"normalized.svg": digest(output / "normalized.svg")}
    result["normalization"] = normalization
    write_json(output / "recipe/recipe.json", captured)
    result["exact_matches"] = {name: digest(output / name) == expected
                               for name, expected in recipe.get("outputs", {}).items() if (output / name).is_file()}
    write_json(output / "replay.json", result)
    return result
