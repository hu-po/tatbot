"""Recipe expansion: reproducible, resumable, and honest about which stage ran.

Nothing here compiles a scenario or renders a frame. These are recipes, and the
counts they carry say so.
"""
from __future__ import annotations

import json

import pytest
from tatbot_sim.inkmap.collection import collection_artifacts
from tatbot_sim.inkmap.recipes import (
    Range,
    RecipeAxes,
    RecipeError,
    build_plan,
    expand_recipe,
    load_plan,
    materialize_recipes,
    read_recipes,
    split_audit,
)

POSES = ("supine", "prone", "reclined-seated")
SITES = ("forearm", "thigh")


def plan(count=40, seed=11, **kwargs):
    return build_plan(collection_artifacts("all"), count=count, seed=seed,
                      poses=POSES, sites=SITES, **kwargs)


# ---- identity and determinism ---------------------------------------------
def test_the_same_plan_expands_to_byte_identical_recipes():
    document = plan()
    first = [expand_recipe(document, index) for index in range(document["requested"])]
    second = [expand_recipe(document, index) for index in reversed(range(document["requested"]))]
    assert [item["recipe_sha256"] for item in first] == [item["recipe_sha256"] for item in reversed(second)]


def test_a_recipe_does_not_depend_on_what_ran_before_it():
    """Sharding is the point: worker 2 must not need worker 1 to have run."""
    document = plan()
    alone = expand_recipe(document, 17)
    assert alone == expand_recipe(document, 17)
    # Expanding an unrelated slice first changes nothing.
    for index in (0, 3, 39):
        expand_recipe(document, index)
    assert expand_recipe(document, 17) == alone


def test_plan_identity_ignores_the_order_things_were_listed_in():
    designs = collection_artifacts("all")
    a = build_plan(designs, count=5, seed=1, poses=POSES, sites=SITES)
    b = build_plan(tuple(reversed(designs)), count=5, seed=1,
                   poses=tuple(reversed(POSES)), sites=tuple(reversed(SITES)))
    assert a["plan_id"] == b["plan_id"]


def test_a_different_seed_or_axis_range_is_a_different_plan():
    base = plan()
    assert plan(seed=12)["plan_id"] != base["plan_id"]
    narrowed = plan(axes=RecipeAxes(scale=Range(1.0, 1.0)))
    assert narrowed["plan_id"] != base["plan_id"]
    assert {expand_recipe(narrowed, index)["scale"] for index in range(5)} == {1.0}


def test_axes_vary_independently_of_each_other():
    document = plan(count=200)
    recipes = [expand_recipe(document, index) for index in range(200)]
    for axis in ("pose", "site", "scale", "rotation_deg"):
        assert len({str(recipe[axis]) for recipe in recipes}) > 1, axis
    # Downstream streams are named and distinct, so a camera and a sensor model
    # can vary without disturbing the placement.
    assert len({tuple(recipe["streams"].values()) for recipe in recipes}) == 200
    assert all(len(set(recipe["streams"].values())) == 4 for recipe in recipes)


# ---- splits ---------------------------------------------------------------
def test_a_variant_of_a_training_artwork_stays_in_its_family_split():
    """Scale, rotation and mirror are augmentation; they must not move a split."""
    document = plan(count=300)
    by_artwork: dict[str, set[str]] = {}
    for index in range(300):
        recipe = expand_recipe(document, index)
        by_artwork.setdefault(recipe["artwork"]["family_sha256"], set()).add(recipe["split"])
    assert all(len(splits) == 1 for splits in by_artwork.values())


def test_no_family_leaks_across_the_held_out_partitions(tmp_path):
    # A wide holdout, so the partitions are actually populated: an audit that
    # passes because everything is `train` proves nothing.
    ledger = materialize_recipes(tmp_path / "run", plan(count=200, holdout_basis_points=4000))
    assert len(ledger["splits"]) > 1, ledger["splits"]
    assert set(ledger["splits"]) <= {"train", "design-held-out", "identity-held-out", "joint-held-out"}
    assert split_audit(tmp_path / "run") == []


# ---- persistence, resume and counts ---------------------------------------
def test_recipes_are_written_once_and_resume_reuses_them(tmp_path):
    root = tmp_path / "run"
    document = plan(count=30)
    first = materialize_recipes(root, document, indices=list(range(10)))
    assert first["counts"]["admitted"] == 10 and first["counts"]["reused"] == 0
    assert first["complete"] is False
    before = {path.name: path.read_bytes() for path in (root / "recipes").iterdir()}

    full = materialize_recipes(root, document)
    assert full["counts"]["admitted"] == 30 and full["counts"]["reused"] == 10
    assert full["complete"] is True
    after = {path.name: path.read_bytes() for path in (root / "recipes").iterdir()}
    assert all(after[name] == data for name, data in before.items())


def test_an_interrupted_run_and_an_uninterrupted_one_agree(tmp_path):
    document = plan(count=24)
    whole = materialize_recipes(tmp_path / "whole", document)
    for shard in ([0, 4, 8, 12, 16, 20], list(range(24))):
        materialize_recipes(tmp_path / "pieces", document, indices=shard)
    pieces = materialize_recipes(tmp_path / "pieces", document)
    assert whole["counts"]["admitted"] == pieces["counts"]["admitted"]
    left = read_recipes(tmp_path / "whole", document)
    right = read_recipes(tmp_path / "pieces", document)
    assert [item["recipe_sha256"] for item in left] == [item["recipe_sha256"] for item in right]


def test_a_tampered_recipe_is_rewritten_from_the_plan(tmp_path):
    root = tmp_path / "run"
    document = plan(count=6)
    materialize_recipes(root, document)
    victim = sorted((root / "recipes").iterdir())[0]
    tampered = json.loads(victim.read_text())
    tampered["rotation_deg"] = 999.0
    victim.write_text(json.dumps(tampered))
    ledger = materialize_recipes(root, document)
    assert ledger["counts"]["reused"] == 5
    assert json.loads(victim.read_text())["rotation_deg"] != 999.0


def test_a_different_plan_cannot_resume_in_an_occupied_directory(tmp_path):
    root = tmp_path / "run"
    materialize_recipes(root, plan(count=4))
    with pytest.raises(RecipeError, match="resume the original"):
        materialize_recipes(root, plan(count=4, seed=99))


def test_counts_distinguish_a_recipe_from_a_render_and_a_drawing(tmp_path):
    ledger = materialize_recipes(tmp_path / "run", plan(count=12))
    counts = ledger["counts"]
    assert counts["requested"] == 12 and counts["admitted"] == 12
    # Nothing has been compiled, rendered or executed, and the ledger says so
    # rather than reporting one number that could be read as any of them.
    assert (counts["compiled"], counts["rendered"], counts["executed"]) == (0, 0, 0)


def test_a_rejected_recipe_keeps_its_reason(tmp_path):
    root = tmp_path / "run"
    document = plan(count=8, axes=RecipeAxes(scale=Range(50.0, 50.0)))
    ledger = materialize_recipes(root, document)
    assert ledger["counts"]["admitted"] == 0 and ledger["counts"]["rejected"] == 8
    rows = [json.loads(line) for line in (root / "rejected.jsonl").read_text().splitlines()]
    assert {row["reason"] for row in rows} == {"size_outside_domain"}
    assert all(row["artwork"] and row["key"] for row in rows)


def test_a_thousand_recipes_expand_offline(tmp_path):
    """The plan's D5 scale, with no generator, no compiler and no network."""
    ledger = materialize_recipes(tmp_path / "run", plan(count=1000, seed=5))
    assert ledger["counts"]["admitted"] == 1000 and ledger["complete"] is True
    assert len(ledger["coverage"]["poses"]) == len(POSES)
    assert split_audit(tmp_path / "run") == []


def test_a_stored_plan_reads_back(tmp_path):
    root = tmp_path / "run"
    document = plan(count=3)
    materialize_recipes(root, document)
    assert load_plan(root)["plan_id"] == document["plan_id"]
    assert len(read_recipes(root, document)) == 3


def test_artwork_with_no_reviewed_family_is_named_rather_than_invented(tmp_path):
    """A generated library has no families; that is recorded, not fabricated."""
    from tatbot_sim.inkmap.designs import DesignArtifact

    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
           '<rect x="1" y="1" width="8" height="8" fill="#111"/></svg>')
    designs = tuple(DesignArtifact(id=f"gen-{index}", name=f"generated {index}", svg=svg,
                                   size_mm=(20.0, 20.0),
                                   source={"kind": "generated", "model": "m", "seed": index})
                    for index in range(4))
    document = build_plan(designs, count=20, seed=3, poses=POSES, sites=SITES)
    ledger = materialize_recipes(tmp_path / "run", document)
    assert ledger["unknown_family_artworks"] == ["gen-0", "gen-1", "gen-2", "gen-3"]
    # One group, so the whole library moves together — safe, and not a split.
    assert len(ledger["splits"]) == 1
    assert split_audit(tmp_path / "run") == []
