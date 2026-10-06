"""Expanding a frozen artwork library into reproducible scenario recipes.

A recipe is not a render, and a render is not a completed drawing. Keeping those
three apart is most of what this module is for: it plans and admits *recipes*,
cheaply and offline, and records per-stage counts that a later compiler, a later
renderer and a later executor each advance on their own. Nothing here compiles a
scenario, opens a renderer, loads a model or touches a network.

Determinism is by construction rather than by discipline. Every axis of a recipe
is drawn from an independently seeded stream keyed by (plan seed, recipe key,
axis name), and the recipe key comes from the plan's own content — so sharding
the work, resuming it, or reordering it produces byte-identical recipes. A run
that dies halfway and one that never stopped write the same files.

Artwork family and identity splits are assigned before any augmentation, from
the artwork's *family*, never from a variant's seed, crop or SVG digest. A
rotated copy of a training artwork therefore cannot become held-out artwork.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tatbot_sim.inkmap.contracts import document_sha256
from tatbot_sim.inkmap.designs import DesignArtifact
from tatbot_sim.inkmap.perception_variation import Range, split_for_groups, stable_seed
from tatbot_sim.site_sampling import site_choices

PLAN_SCHEMA = "tatbot.inkmap-recipe-plan/1"
RECIPE_SCHEMA = "tatbot.inkmap-recipe/1"
LEDGER_SCHEMA = "tatbot.inkmap-recipe-ledger/1"

MAX_RECIPES = 100_000
MAX_REASON_CHARS = 240
# Every stage a scenario passes through, and the fact that reaching one says
# nothing about the next. `tatbot sim recipes` fills the first two.
STAGES = ("requested", "admitted", "rejected", "compiled", "rendered", "executed")


class RecipeError(ValueError):
    """A plan or a recipe could not be built, with the reason a caller can act on."""


@dataclass(frozen=True)
class RecipeAxes:
    """Independently seeded axes. A zero-width range disables one exactly."""

    scale: Range = field(default_factory=lambda: Range(0.85, 1.15))
    rotation_deg: Range = field(default_factory=lambda: Range(-25.0, 25.0))
    mirror_probability: float = 0.25
    target_x_m: Range = field(default_factory=lambda: Range(0.30, 0.32))
    target_y_m: Range = field(default_factory=lambda: Range(-0.035, 0.045))
    target_z_m: Range = field(default_factory=lambda: Range(0.04, 0.04))

    def as_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> RecipeAxes:
        ranges = {name: Range(**document[name]) for name in
                  ("scale", "rotation_deg", "target_x_m", "target_y_m", "target_z_m")}
        return cls(**ranges, mirror_probability=float(document["mirror_probability"]))


def _artwork_entry(design: DesignArtifact) -> dict[str, Any]:
    """What a recipe needs to know about one artwork, and nothing more."""
    source = design.source
    family = source.get("family") or source.get("collection") or "unknown"
    return {
        "id": design.id,
        "sha256": design.sha256,
        "name": design.name,
        "family": family,
        # The family, not the artwork: this is what keeps a variant of a
        # training artwork out of the held-out partitions.
        "family_sha256": document_sha256({"family": family}),
        "size_mm": [float(value) for value in design.size_mm],
        "kind": source.get("kind", "unknown"),
        "license": source.get("license"),
        "collection_split": source.get("split"),
    }


def build_plan(
    designs: tuple[DesignArtifact, ...],
    *,
    count: int,
    seed: int,
    poses: tuple[str, ...],
    sites: tuple[str, ...],
    identities: tuple[str, ...] = ("mhr-soma-v1",),
    axes: RecipeAxes | None = None,
    holdout_basis_points: int = 1000,
    label: str = "",
) -> dict[str, Any]:
    """Freeze what is being asked for. The plan's digest is the run's identity."""
    if not designs:
        raise RecipeError("a plan needs at least one artwork")
    if not 0 < count <= MAX_RECIPES:
        raise RecipeError(f"count must be 1-{MAX_RECIPES}, got {count}")
    if not poses or not sites or not identities:
        raise RecipeError("a plan needs at least one pose, site and identity")
    try:
        site_values = site_choices(tuple(sites))
    except ValueError as exc:
        raise RecipeError(str(exc)) from exc
    entries = [_artwork_entry(design) for design in designs]
    if len({entry["id"] for entry in entries}) != len(entries):
        raise RecipeError("artwork ids must be unique within a plan")
    # Sorted everywhere: two callers who listed the same work in a different
    # order must produce the same plan and therefore the same recipes.
    document = {
        "schema": PLAN_SCHEMA, "label": label, "seed": int(seed), "requested": int(count),
        "artworks": sorted(entries, key=lambda entry: entry["id"]),
        "poses": sorted(set(poses)),
        "sites": sorted(({"id": site.id, "laterality": site.laterality} for site in site_values),
                        key=lambda site: (site["id"], site["laterality"] or "")),
        "identities": sorted(set(identities)),
        "axes": (axes or RecipeAxes()).as_json(),
        "holdout_basis_points": int(holdout_basis_points),
    }
    document["plan_id"] = document_sha256(document)
    return document


def _stream(plan: dict[str, Any], key: str, axis: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(int(plan["seed"]), key, axis))


def _pick(values: list, plan: dict[str, Any], key: str, axis: str):
    """One value from a list, from this recipe's own stream for this axis."""
    return values[int(_stream(plan, key, axis).integers(0, len(values)))]


def recipe_key(plan: dict[str, Any], index: int) -> str:
    """Stable, content-derived, and independent of who computed it when."""
    return f"{plan['plan_id'][:8]}-{index:06d}"


def expand_recipe(plan: dict[str, Any], index: int) -> dict[str, Any]:
    """One recipe. Pure: same plan and index give byte-identical output."""
    if not 0 <= index < int(plan["requested"]):
        raise RecipeError(f"recipe {index} is outside the plan's {plan['requested']} slots")
    key = recipe_key(plan, index)
    axes = RecipeAxes.from_json(plan["axes"])
    artwork = _pick(list(plan["artworks"]), plan, key, "artwork")
    identity = _pick(list(plan["identities"]), plan, key, "identity")
    pose = _pick(list(plan["poses"]), plan, key, "pose")
    site = _pick(list(plan["sites"]), plan, key, "site")
    scale = axes.scale.sample(_stream(plan, key, "scale"))
    rotation_deg = axes.rotation_deg.sample(_stream(plan, key, "rotation_deg"))
    mirror = bool(_stream(plan, key, "mirror").random() < axes.mirror_probability)
    target = [axes.target_x_m.sample(_stream(plan, key, "target_x_m")),
              axes.target_y_m.sample(_stream(plan, key, "target_y_m")),
              axes.target_z_m.sample(_stream(plan, key, "target_z_m"))]
    # Assigned from the family and the identity, before any of the above was
    # drawn. Changing a seed, a scale or a crop cannot move a recipe's split.
    split = split_for_groups(
        artwork_family_sha256=artwork["family_sha256"],
        identity_sha256=document_sha256({"identity": identity}),
        seed=int(plan["seed"]), holdout_basis_points=int(plan["holdout_basis_points"]))
    recipe = {
        "schema": RECIPE_SCHEMA, "plan_id": plan["plan_id"], "key": key, "index": int(index),
        "artwork": {name: artwork[name] for name in ("id", "sha256", "family", "family_sha256")},
        "artwork_family_sha256": artwork["family_sha256"],
        "identity": identity, "identity_sha256": document_sha256({"identity": identity}),
        "split": split,
        "pose": pose, "site": site["id"], "laterality": site["laterality"],
        "size_mm": [round(value * scale, 6) for value in artwork["size_mm"]],
        "scale": round(scale, 6), "rotation_deg": round(rotation_deg, 6), "mirror": mirror,
        "target_world_m": [round(value, 6) for value in target],
        # Named downstream streams, so a renderer and a sensor model vary
        # independently of the placement and of each other.
        "streams": {axis: stable_seed(int(plan["seed"]), key, axis)
                    for axis in ("camera", "appearance", "sensor", "compile")},
    }
    recipe["recipe_sha256"] = document_sha256(recipe)
    return recipe


def validate_recipe(recipe: dict[str, Any], plan: dict[str, Any]) -> None:
    """Refuse a recipe that no longer matches the plan it claims to come from."""
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise RecipeError(f"not a {RECIPE_SCHEMA}")
    if recipe.get("plan_id") != plan["plan_id"]:
        raise RecipeError("recipe belongs to a different plan")
    stated = recipe.get("recipe_sha256")
    if stated != document_sha256({k: v for k, v in recipe.items() if k != "recipe_sha256"}):
        raise RecipeError(f"recipe {recipe.get('key')} digest differs from its content")


def admit(recipe: dict[str, Any], plan: dict[str, Any]) -> str | None:
    """The reason this recipe cannot be used, or None when it can.

    Geometry checks that need a body rig, a compiler or a reach audit are not
    here: they belong to compilation, which is a later stage with its own
    counter. What this rejects is a recipe that is unusable on its face.
    """
    size = recipe["size_mm"]
    if not all(math.isfinite(value) and value > 0 for value in size):
        return "nonfinite_or_nonpositive_size"
    entry = next((item for item in plan["artworks"] if item["id"] == recipe["artwork"]["id"]), None)
    if entry is None:
        return "artwork_not_in_plan"
    if entry["sha256"] != recipe["artwork"]["sha256"]:
        return "artwork_bytes_changed"
    if max(size) > 200:
        return "size_outside_domain"
    if not all(math.isfinite(value) for value in recipe["target_world_m"]):
        return "nonfinite_target"
    return None


# ---- persistence -----------------------------------------------------------
def _atomic_write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        for row in rows))
    temporary.replace(path)


def load_plan(root: Path) -> dict[str, Any]:
    path = Path(root) / "plan.json"
    if not path.is_file():
        raise RecipeError(f"{root} is not a recipe directory")
    document = json.loads(path.read_text())
    if document.get("schema") != PLAN_SCHEMA:
        raise RecipeError(f"{path} is not a {PLAN_SCHEMA}")
    return document


def recipe_path(root: Path, recipe: dict[str, Any]) -> Path:
    return Path(root) / "recipes" / f"{recipe['index']:06d}-{recipe['key']}.json"


def materialize_recipes(root: Path, plan: dict[str, Any], *,
                        indices: list[int] | None = None) -> dict[str, Any]:
    """Write (or resume writing) a plan's recipes. Offline; nothing is compiled.

    Resume is trivial because expansion is pure: a recipe already on disk is
    verified against the plan and kept, and anything that no longer matches is
    rewritten from the plan rather than trusted.
    """
    root = Path(root)
    frozen = root / "plan.json"
    if frozen.is_file():
        existing = json.loads(frozen.read_text())
        if existing.get("plan_id") != plan["plan_id"]:
            raise RecipeError(
                f"{root} holds plan {str(existing.get('plan_id'))[:12]} and this is "
                f"{plan['plan_id'][:12]}; resume the original or choose a new directory")
        plan = existing
    else:
        _atomic_write_json(frozen, plan)
    wanted = list(range(int(plan["requested"]))) if indices is None else sorted(set(indices))
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reused = 0
    for index in wanted:
        recipe = expand_recipe(plan, index)
        reason = admit(recipe, plan)
        if reason is not None:
            rejected.append({"key": recipe["key"], "index": index, "reason": reason,
                             "artwork": recipe["artwork"]["id"]})
            continue
        path = recipe_path(root, recipe)
        if path.is_file():
            try:
                validate_recipe(json.loads(path.read_text()), plan)
                reused += 1
            except (RecipeError, json.JSONDecodeError):
                _atomic_write_json(path, recipe)
            else:
                admitted.append(recipe)
                continue
        else:
            _atomic_write_json(path, recipe)
        admitted.append(recipe)
    _jsonl(root / "recipes.jsonl", [
        {"key": recipe["key"], "index": recipe["index"], "recipe_sha256": recipe["recipe_sha256"],
         "artwork": recipe["artwork"]["id"], "split": recipe["split"], "pose": recipe["pose"],
         "site": recipe["site"], "laterality": recipe["laterality"],
         "path": str(recipe_path(root, recipe).relative_to(root))}
        for recipe in admitted])
    _jsonl(root / "rejected.jsonl", rejected)
    ledger = {
        "schema": LEDGER_SCHEMA, "plan_id": plan["plan_id"], "root": str(root),
        "counts": {"requested": int(plan["requested"]), "planned": len(wanted),
                   "admitted": len(admitted), "rejected": len(rejected),
                   "reused": reused,
                   # Later stages advance these; a recipe is not a render and a
                   # render is not a drawing, so they start at zero and stay
                   # there until something actually does the work.
                   "compiled": 0, "rendered": 0, "executed": 0},
        "splits": {name: sum(1 for recipe in admitted if recipe["split"] == name)
                   for name in sorted({recipe["split"] for recipe in admitted})},
        # Artwork with no reviewed family — a freshly generated library, for
        # instance — is one group, so the whole of it moves across splits
        # together. That is the conservative reading of unknown provenance, not
        # a meaningful family split, and the ledger says which artwork it is
        # rather than leaving a reader to infer it from a single-split tally.
        "unknown_family_artworks": sorted({entry["id"] for entry in plan["artworks"]
                                           if entry["family"] == "unknown"}),
        "coverage": {
            "artworks": sorted({recipe["artwork"]["id"] for recipe in admitted}),
            "poses": sorted({recipe["pose"] for recipe in admitted}),
            "sites": sorted({recipe["site"] for recipe in admitted}),
        },
        "complete": len(admitted) == int(plan["requested"]) and indices is None,
    }
    _atomic_write_json(root / "ledger.json", ledger)
    return ledger


def read_ledger(root: Path) -> dict[str, Any]:
    path = Path(root) / "ledger.json"
    if not path.is_file():
        raise RecipeError(f"{root} has no recipe ledger; run `tatbot sim recipes` there first")
    return json.loads(path.read_text())


def read_recipes(root: Path, plan: dict[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    """The admitted recipes on disk, verified against the plan."""
    root = Path(root)
    index = root / "recipes.jsonl"
    if not index.is_file():
        raise RecipeError(f"{root} has no recipes.jsonl")
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    output = []
    for row in rows[:limit]:
        recipe = json.loads((root / row["path"]).read_text())
        validate_recipe(recipe, plan)
        if recipe["recipe_sha256"] != row["recipe_sha256"]:
            raise RecipeError(f"recipe {recipe['key']} differs from the index")
        output.append(recipe)
    return output


def split_audit(root: Path) -> list[str]:
    """Family and identity leakage across the held-out partitions, if any."""
    from tatbot_sim.inkmap.perception_variation import audit_split_records

    index = Path(root) / "recipes.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    plan = load_plan(root)
    by_id = {entry["id"]: entry for entry in plan["artworks"]}
    return audit_split_records([
        {"artwork_family_sha256": by_id[row["artwork"]]["family_sha256"],
         "identity_sha256": document_sha256({"identity": plan["identities"][0]}),
         "split": row["split"]}
        for row in rows])
