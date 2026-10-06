from __future__ import annotations

import hashlib
import json
from xml.etree import ElementTree as ET

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from tatbot_sim.human_rep.contracts import ContractError, canonical_digest
from tatbot_sim.inkmap.artwork import make_artwork_record, validate_artwork_record
from tatbot_sim.repo import repo_root


def fixture(name="blackwork"):
    svg = (repo_root() / "web/inkmap/tests/fixtures/artwork" / f"{name}.svg").read_text()
    box = [float(v) for v in ET.fromstring(svg).attrib["viewBox"].split()]
    return make_artwork_record(
        name=name, original_svg=svg,
        source={"kind": "fixture", "identifier": name, "license": "CC0-1.0", "attribution": "Tatbot", "generation": None},
        conversion={"adapter": "tatbot-svg-paint/1", "canvas_m": [box[2] / 1000, box[3] / 1000],
                    "semantic_intent": name, "width_m": 0.0008, "deposition": 0.7, "chord_error_m": 0.0001},
    )


@pytest.mark.parametrize("name", ["linework", "blackwork", "negative-space", "stipple", "color-layers"])
def test_browser_python_record_and_json_schema_agree(name):
    record = fixture(name)
    assert validate_artwork_record(record) == record
    assert canonical_digest(record) == record["content_sha256"]
    assert hashlib.sha256((repo_root() / "web/inkmap/tests/fixtures/artwork" / f"{name}.svg").read_bytes()).hexdigest() == record["source_sha256"]
    assert "original_svg" not in record and "preview_svg" not in record
    resources = []
    for path in (repo_root() / "config/human-representation").glob("*.schema.json"):
        schema = json.loads(path.read_text())
        resources.append((schema["$id"], Resource.from_contents(schema)))
    schema = json.loads((repo_root() / "config/inkmap/artwork.schema.json").read_text())
    Draft202012Validator(schema, registry=Registry().with_resources(resources)).validate(record)


def test_rehashed_envelope_cannot_hide_stale_program_identity():
    broken = fixture()
    broken["program"]["layers"][0]["elements"][0]["width_m"] *= 2
    broken["content_sha256"] = canonical_digest(broken)
    with pytest.raises(ContractError, match="wrong_hash"):
        validate_artwork_record(broken)


def test_unknown_license_is_explicit_and_does_not_become_a_grant():
    record = fixture()
    record["source"]["license"] = None
    record["source"]["attribution"] = None
    record["content_sha256"] = canonical_digest(record)
    assert validate_artwork_record(record)["source"]["license"] is None
