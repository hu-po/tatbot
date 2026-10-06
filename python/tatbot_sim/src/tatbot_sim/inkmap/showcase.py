"""Regenerate the five preview poses with frozen shared artwork and typed traces."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from tatbot_sim.inkmap.collection import collection_artifacts
from tatbot_sim.inkmap.contracts import POSE_ASSET_SHA256
from tatbot_sim.inkmap.rig import CATALOG_DIGEST_PATH
from tatbot_sim.inkmap.sampler import compile_artwork_placement
from tatbot_sim.repo import repo_root, source_state


def _current_scenario(path: Path, design) -> bool:
    if not path.is_file():
        return False
    scenario = json.loads(path.read_text())
    digest = json.loads(CATALOG_DIGEST_PATH.read_text())["sha256"]
    return (scenario["body"]["pose_asset_sha256"] == POSE_ASSET_SHA256
            and scenario["pose"]["catalog_sha256"] == digest
            and scenario.get("design", {}).get("id") == design.id
            and scenario.get("design", {}).get("sha256") == design.embedded()["source_sha256"]
            and scenario.get("program_binding", {}).get("bundle", {}).get("artworks", {}).get(design.id) == design.embedded())


def materialize_showcase(output: Path, install: bool = False, scenarios_only: bool = False):
    root = repo_root()
    if not install and output.resolve().is_relative_to(root.resolve()):
        raise ValueError("generate outside the repository, then review before installing assets")
    output.mkdir(parents=True, exist_ok=install)
    public = root / "web/inkmap/public/showcase"
    manifest = json.loads((public / "manifest.json").read_text())
    body = json.loads((root / "config/inkmap/examples/forearm-placement-v6.json").read_text())["body"]
    designs = {d.id: d for d in collection_artifacts()}
    ids = ("dbv3-orbit", "dbv3-sprout", "dbv3-ridges", "dbv3-orbit", "dbv3-sprout")
    state = ({"revision": manifest["source_commit"], "dirty": manifest.get("source_dirty", False)}
             if scenarios_only else source_state())
    for index, (slide, design_id) in enumerate(zip(manifest["slides"], ids, strict=True)):
        design = designs[design_id]
        if scenarios_only and _current_scenario(output / slide["scenario"], design):
            continue
        placement = deepcopy(slide["placement"])
        placement.pop("source_sha256", None)
        placement.update(id=f"showcase-{slide['id']}-{design.id}", design_id=design.id, size_mm=list(design.size_mm))
        if "language" in placement:
            language = placement["language"]
            old_motif = language["program"]["motif"]
            language["program"]["motif"] = design.name.lower()
            language["sentence"] = language["sentence"].replace(old_motif, design.name.lower())
        file = {"schema_version": 6, "units": {"length": "m", "tattoo_size": "mm", "up": "+z"},
                "body": body, "placements": [placement], "designs": {design.id: design.embedded()}}
        scenario = compile_artwork_placement(file, design, pose_id=slide["pose_id"], seed=904 + index,
            target=[.29, 0, .04], created_at="2026-10-02T00:00:00Z", git_sha=state["revision"])
        (output / slide["scenario"]).write_text(json.dumps(scenario, separators=(",", ":")) + "\n")
        site = placement["site"]
        slide["placement"] = placement
        slide["title"] = f"{design.name} on {site.get('laterality', '')} {site['id']}"
        slide["description"] = f"The shared {design.name.lower()} artwork is placed on the named {slide['pose_id']} pose. Preview paint and the typed material trace retain the same frozen source."
        slide["artwork_id"] = design.id
        slide["artwork_sha256"] = design.embedded()["source_sha256"]
        print(f"compiled {slide['id']}: {design.id}", flush=True)
    manifest.update(title="Acquired DBV3 artwork across five poses", source_commit=state["revision"],
                    source_dirty=state["dirty"], generated_at="2026-10-02T00:00:00Z")
    manifest["coverage"]["designs"] = len(set(ids))
    manifest["validation"].update(trace_compiler_version=3, pose_asset_sha256=POSE_ASSET_SHA256,
                                  reach_audited=False, visual_review="pending")
    if not scenarios_only:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--install", action="store_true",
                        help="permit writing into web/inkmap/public/showcase, whose scenarios are ignored build output")
    parser.add_argument("--scenarios-only", action="store_true",
                        help="rebuild ignored scenarios with the retained manifest provenance; do not rewrite the manifest")
    args = parser.parse_args()
    materialize_showcase(args.output_dir.expanduser().resolve(), install=args.install, scenarios_only=args.scenarios_only)


if __name__ == "__main__":
    main()
