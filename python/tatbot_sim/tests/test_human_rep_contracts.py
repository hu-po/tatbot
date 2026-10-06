from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from tatbot_sim.human_rep.contracts import (
    KNOWN_SCHEMAS,
    REVIEWED_MODEL_SPEC_SHA256,
    ContractError,
    canonical_bytes,
    canonical_digest,
    load_contract,
    parse_json,
    validate_contract,
)

REPO = Path(__file__).resolve().parents[3]
SCHEMAS = REPO / "config" / "human-representation"
EXAMPLES = SCHEMAS / "examples"
TOPOLOGY = "e0ca7ee25dc0b4c8d841bb2626e364bb88b7af7fae037e30854728842e320a18"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_fixture_set_round_trips_and_binds_current_outputs():
    fixture = json.loads((EXAMPLES / "fixture-set.json").read_text())
    assert "no motion" in fixture["description"].lower()
    for entry in fixture["files"]:
        path = EXAMPLES / entry["path"]
        value = load_contract(
            path,
            expected_schema=entry["schema"],
            expected_topology_sha256=TOPOLOGY,
        )
        assert canonical_digest(value) == entry["content_sha256"]
        canonical = canonical_bytes(value)
        assert validate_contract(
            parse_json(canonical),
            expected_schema=entry["schema"],
            expected_topology_sha256=TOPOLOGY,
        ) == value
    for entry in fixture["python_files"]:
        value = load_contract(EXAMPLES / entry["path"], expected_schema=entry["schema"])
        assert canonical_digest(value) == entry["content_sha256"]
        assert validate_contract(parse_json(canonical_bytes(value))) == value
    for entry in fixture["derived"]:
        assert sha256(EXAMPLES / entry["path"]) == entry["sha256"]

    tattoo = load_contract(EXAMPLES / "tattoo-program.json")
    identity = load_contract(EXAMPLES / "body-identity.json")
    body_state = load_contract(EXAMPLES / "body-state.json")
    placement = load_contract(EXAMPLES / "surface-placement.json")
    ink_program = load_contract(EXAMPLES / "ink-program.json")
    registration = load_contract(EXAMPLES / "surface-registration.json")
    execution = load_contract(EXAMPLES / "execution-program.json")
    samples_sidecar = load_contract(EXAMPLES / "current-samples.manifest.json")
    artwork = load_contract(EXAMPLES / "artwork-matrix.json")

    elements = [element for layer in artwork["layers"] for element in layer["elements"]]
    assert {element["kind"] for element in elements} >= {
        "path",
        "cubic_bezier",
        "region",
        "stipple",
    }
    cubic = next(element for element in elements if element["kind"] == "cubic_bezier")
    assert len(cubic["control_points_m"]) == 4
    assert artwork["negative_space_masks"]
    assert len(artwork["inks"]) > 1

    assert placement["tattoo_program_sha256"] == tattoo["content_sha256"]
    assert identity["model_spec_sha256"] == fixture["model_spec_sha256"]
    assert body_state["model_spec_sha256"] == fixture["model_spec_sha256"]
    assert placement["body_identity_sha256"] == identity["content_sha256"]
    assert placement["rest_surface_sha256"] == identity["rest_surface_sha256"]
    assert body_state["body_identity_sha256"] == identity["content_sha256"]
    assert ink_program["tattoo_program_sha256"] == tattoo["content_sha256"]
    assert ink_program["surface_placement_sha256"] == placement["content_sha256"]
    assert registration["body_state_sha256"] == body_state["content_sha256"]
    assert execution["ink_program"] == ink_program
    assert execution["body_state"] == body_state
    assert execution["surface_registration"] == registration
    assert execution["samples_manifest"]["schema"] == "tatbot.draw-samples/1"
    assert execution["samples_manifest"]["sha256"] == sha256(EXAMPLES / "current-samples.csv")
    assert execution["samples_manifest"]["sidecar_sha256"] == sha256(EXAMPLES / "current-samples.manifest.json")
    assert execution["measured_surface"]["sha256"] == sha256(EXAMPLES / "current-surface.npz")
    assert samples_sidecar["samples"]["sha256"] == execution["samples_manifest"]["sha256"]
    assert samples_sidecar["inputs"] == {
        "ink_program_sha256": ink_program["content_sha256"],
        "body_state_sha256": body_state["content_sha256"],
        "surface_registration_sha256": registration["content_sha256"],
        "measured_surface_sha256": execution["measured_surface"]["sha256"],
        "tool_datasheet_sha256": execution["tool"]["datasheet_sha256"],
        "robot_urdf_sha256": execution["robot"]["urdf_sha256"],
        "support_configuration_sha256": execution["support"]["configuration_sha256"],
        "palette_snapshot_sha256": execution["palette"]["snapshot_sha256"],
        "palette_load_state_sha256": execution["palette"]["load_state_sha256"],
        "calibration_sha256": execution["calibration"]["sha256"],
        "policy_sha256": execution["preflight"]["policy_sha256"],
    }
    assert samples_sidecar["sample_ranges"] == [
        {
            "ink_event_index": event["ink_event_index"],
            "kind": event["kind"],
            "start": event["sample_range"][0],
            "stop": event["sample_range"][1],
        }
        for event in execution["exact_events"]
    ]
    assert samples_sidecar["preflight"] == {
        "mode": "fixture-only",
        "motion_authorized": False,
        "observed_cells_only": execution["preflight"]["observed_cells_only"],
        "registration_sigma_m": execution["uncertainty"]["registration_sigma_m"],
        "surface_sigma_m": execution["uncertainty"]["surface_sigma_m"],
        "max_surface_age_s": execution["preflight"]["max_surface_age_s"],
    }


def test_current_surface_fixture_uses_pickle_free_schema():
    with np.load(EXAMPLES / "current-surface.npz", allow_pickle=False) as surface:
        assert str(surface["schema"]) == "tatbot.surface/1"
        assert surface["height"].dtype == np.dtype("<f8")
        assert surface["count"].dtype == np.dtype("<i4")
        assert surface["height"].shape == surface["count"].shape == (2, 2)


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        (1.0, b"1"),
        (1e-7, b"1e-7"),
        (1e-6, b"0.000001"),
        (1e20, b"100000000000000000000"),
        (1e21, b"1e+21"),
    ],
)
def test_canonical_numbers_follow_ecmascript_spelling(value: float, canonical: bytes):
    assert canonical_bytes(value) == canonical


def test_canonical_json_rejects_integers_javascript_cannot_preserve():
    with pytest.raises(ContractError, match="unsafe_integer"):
        canonical_bytes(2**53)
    with pytest.raises(ContractError, match="unsafe_integer"):
        parse_json(str(2**53))


def test_canonical_keys_use_utf16_order_for_browser_parity():
    assert canonical_bytes({"\ue000": 1, "\U00010000": 2}) == '{"𐀀":2,"":1}'.encode()


def test_tracked_body_model_spec_is_strict_and_self_hashing():
    value = load_contract(REPO / "config" / "body-models" / "mhr-soma-v1.json")
    assert value["content_sha256"] == REVIEWED_MODEL_SPEC_SHA256
    assert value["model"]["identity_model_type"] == "mhr"
    assert value["model"]["shape_components"] == 45
    assert value["security"]["automatic_download"] is False


def test_body_model_reader_does_not_treat_integer_one_as_boolean_true():
    value = json.loads((REPO / "config" / "body-models" / "mhr-soma-v1.json").read_text())
    value["model"]["apply_correctives"] = 1
    value["content_sha256"] = canonical_digest(value)
    with pytest.raises(ContractError) as caught:
        validate_contract(value)
    assert caught.value.code == "wrong_value"


def test_body_model_reader_refuses_self_hashed_but_unreviewed_specs():
    value = json.loads((REPO / "config" / "body-models" / "mhr-soma-v1.json").read_text())
    value["name"] = "self-hashed-but-unreviewed"
    value["content_sha256"] = canonical_digest(value)

    with pytest.raises(ContractError) as caught:
        validate_contract(value)
    assert caught.value.code == "body_model_unpinned"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda value: value["sources"]["soma_x"].update(repository="file:///tmp/source"),
            "mutable_revision",
        ),
        (
            lambda value: value["asset_source"]["manifest"].update(path="../manifest.json"),
            "path_escape",
        ),
        (
            lambda value: value["software_lock"].update(path="locks//requirements.txt"),
            "path_escape",
        ),
    ],
)
def test_body_model_reader_refuses_unsafe_sources_and_paths(mutate, code):
    value = json.loads((REPO / "config" / "body-models" / "mhr-soma-v1.json").read_text())
    mutate(value)
    value["content_sha256"] = canonical_digest(value)

    with pytest.raises(ContractError) as caught:
        validate_contract(value)
    assert caught.value.code == code


def test_body_state_supports_exactly_one_named_or_tracked_pose_source():
    value = json.loads((EXAMPLES / "body-state.json").read_text())
    value["named_pose"] = None
    value["tracked_source"] = {
        "tracker": "synthetic-fixture",
        "sample_time_utc": "2026-09-04T01:09:37Z",
        "capture_sha256": "7" * 64,
        "source_frame": "body_tracker",
    }
    value["content_sha256"] = canonical_digest(value)
    assert validate_contract(value) == value

    value["named_pose"] = "ambiguous"
    value["content_sha256"] = canonical_digest(value)
    with pytest.raises(ContractError) as caught:
        validate_contract(value)
    assert caught.value.code == "wrong_pose_source"


@pytest.mark.parametrize(
    ("name", "mutate", "code"),
    [
        ("surface-coordinate.json", lambda value: value.update(face_index=36_108), "out_of_range"),
        (
            "surface-curve.json",
            lambda value: value["coordinates"][1].update(topology_sha256="1" * 64),
            "wrong_topology",
        ),
        (
            "surface-placement.json",
            lambda value: value["supported_domain"].update(
                face_indices=[value["anchor"]["face_index"] + 1]
            ),
            "anchor_outside_domain",
        ),
        ("body-state.json", lambda value: value.update(correctives_enabled=False), "wrong_value"),
        (
            "tattoo-program.json",
            lambda value: value["layers"][0].update(elements=[]),
            "wrong_length",
        ),
    ],
)
def test_contract_readers_enforce_mid_topology_and_required_modes(name, mutate, code):
    value = json.loads((EXAMPLES / name).read_text())
    mutate(value)
    value["content_sha256"] = canonical_digest(value)

    with pytest.raises(ContractError) as caught:
        validate_contract(value)
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda value: value["events"][1].update(contact_envelope_m=[0.001, -0.001]),
            "out_of_range",
        ),
        (
            lambda value: value["events"][3].update(target_load=[0.9, 0.7]),
            "out_of_range",
        ),
        (
            lambda value: value.update(total_material_path_length_m=0.02),
            "wrong_value",
        ),
    ],
)
def test_ink_program_reader_enforces_ordered_envelopes_and_derived_length(mutate, code):
    value = json.loads((EXAMPLES / "ink-program.json").read_text())
    mutate(value)
    value["content_sha256"] = canonical_digest(value)

    with pytest.raises(ContractError) as caught:
        validate_contract(value)
    assert caught.value.code == code


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["calibration"].update(sha256="1" * 64),
        lambda value: value["tool"].update({"class": "marker"}),
        lambda value: value["palette"]["resolved_caps"][0].update(ink_id="blue"),
        lambda value: value["exact_events"][0].update(ink_event_index=0),
    ],
)
def test_execution_reader_enforces_session_bindings(mutate):
    value = json.loads((EXAMPLES / "execution-program.json").read_text())
    mutate(value)
    value["content_sha256"] = canonical_digest(value)

    with pytest.raises(ContractError) as caught:
        validate_contract(value)
    assert caught.value.code == "execution_binding_mismatch"


def test_samples_sidecar_refuses_empty_sample_ranges():
    value = json.loads((EXAMPLES / "current-samples.manifest.json").read_text())
    value["sample_ranges"][0]["stop"] = value["sample_ranges"][0]["start"]
    value["content_sha256"] = canonical_digest(value)

    with pytest.raises(ContractError) as caught:
        validate_contract(value)
    assert caught.value.code == "out_of_range"


def test_research_contract_nested_objects_are_closed():
    value = json.loads((EXAMPLES / "tissue-patch.json").read_text())
    value["parameters"]["normal_stiffness"]["unexpected"] = True
    value["content_sha256"] = canonical_digest(value)
    with pytest.raises(ContractError) as caught:
        validate_contract(value)
    assert caught.value.code == "unknown_field"


def test_execution_reader_closes_nested_objects_before_digest_check():
    value = json.loads((EXAMPLES / "execution-program.json").read_text())
    value["tool"]["unexpected"] = True
    with pytest.raises(ContractError) as caught:
        validate_contract(value)
    assert caught.value.code == "unknown_field"


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("malformed.json", "invalid_json"),
        ("unknown-field.json", "unknown_field"),
        ("nonfinite.json", "non_finite"),
        ("wrong-unit.json", "unknown_field"),
        ("wrong-frame.json", "wrong_frame"),
        ("wrong-hash.json", "wrong_hash"),
        ("duplicate-key.json", "duplicate_key"),
        ("negative-zero.json", "negative_zero"),
        ("lone-surrogate.json", "invalid_json"),
        ("unsupported-artwork.json", "unsupported_element"),
    ],
)
def test_named_invalid_fixtures_are_refused(name: str, code: str):
    with pytest.raises(ContractError) as caught:
        load_contract(EXAMPLES / "invalid" / name)
    assert caught.value.code == code


def test_wrong_topology_fixture_is_refused_against_the_model_spec():
    with pytest.raises(ContractError) as caught:
        load_contract(
            EXAMPLES / "invalid" / "wrong-topology.json",
            expected_topology_sha256=TOPOLOGY,
        )
    assert caught.value.code == "wrong_topology"


def test_every_contract_has_a_draft_2020_12_closed_root_schema():
    schema_files = sorted(SCHEMAS.glob("*.schema.json"))
    ids = set()
    for path in schema_files:
        schema = json.loads(path.read_text())
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] not in ids
        ids.add(schema["$id"])
        if path.name != "common.schema.json":
            assert schema["additionalProperties"] is False
    declared = {
        json.loads(path.read_text())["properties"]["schema"]["const"]
        for path in schema_files
        if path.name != "common.schema.json"
    }
    assert declared == KNOWN_SCHEMAS


def test_the_paint_coverage_floor_still_refuses_a_drawing_that_covers_nothing():
    """0.80 admits a drawing with ink missing; it does not admit a non-drawing.

    The floor moved from 0.98 to 0.80 deliberately. What must not move is the
    other end: the nine candidates in the measured suite that covered ~0 are
    still refused, and a caller can restore the old strictness per call.
    """
    import numpy as np
    from shapely import Polygon
    from tatbot_sim.human_rep.contracts import ContractError
    from tatbot_sim.human_rep.fill_geometry import PAINT_COVERAGE_FLOOR, fill_paths

    assert 0.5 < PAINT_COVERAGE_FLOOR < 0.98

    # A square far wider than the stroke: the planner hatches it completely.
    square = Polygon([(0, 0), (0.02, 0), (0.02, 0.02), (0, 0.02)])
    paths = fill_paths(square, 0.0003)
    assert paths and all(isinstance(p, np.ndarray) for p in paths)

    # The same region under the old strictness is still admitted, so relaxing
    # the default did not paper over a shape that already worked.
    assert fill_paths(square, 0.0003, coverage_floor=0.98)

    # A stroke wider than the region cannot be centred inside it at all.
    with pytest.raises(ContractError) as excinfo:
        fill_paths(square, 0.05)
    assert "narrower than the planning width" in str(excinfo.value)

    # And a region the planner covers only partially is refused by the floor,
    # naming the fraction it reached rather than the threshold alone.
    sliver = Polygon([(0, 0), (0.02, 0), (0.02, 0.0007), (0, 0.0007)])
    with pytest.raises(ContractError) as excinfo:
        fill_paths(sliver, 0.0006, coverage_floor=0.999)
    assert "loses paint component coverage" in str(excinfo.value)


def test_paint_thinner_than_the_tool_is_traced_down_the_middle():
    """A tool that cannot draw a line thinner than its tip still draws the line.

    It traces the middle and the line comes out at tool width, which is what a
    person does with a fine liner. The whole drawing used to be refused for it.
    What is bounded is how much wider than the artwork the result may be.
    """
    from shapely import Polygon
    from tatbot_sim.human_rep.contracts import ContractError
    from tatbot_sim.human_rep.fill_geometry import (
        NARROW_OVERDRAW_LIMIT,
        deposited_coverage,
        fill_paths,
    )

    tool = 0.0003
    line = Polygon([(0, 0), (0.020, 0), (0.020, 0.0002), (0, 0.0002)])
    paths = fill_paths(line, tool)
    covered = deposited_coverage(paths, tool)
    assert covered.intersection(line).area / line.area > 0.99, "the line is drawn end to end"
    assert covered.area / line.area < NARROW_OVERDRAW_LIMIT, "drawn heavy, not smeared"

    # A hair far finer than the tool is still refused: tracing it would put
    # most of the ink outside the artwork.
    hair = Polygon([(0, 0), (0.020, 0), (0.020, 0.00005), (0, 0.00005)])
    with pytest.raises(ContractError, match="narrower than the planning width"):
        fill_paths(hair, tool)

    # And a tool larger than the whole region is refused, as it always was.
    square = Polygon([(0, 0), (0.02, 0), (0.02, 0.02), (0, 0.02)])
    with pytest.raises(ContractError, match="narrower than the planning width"):
        fill_paths(square, 0.05)

    # A region the tool fits inside is untouched by any of this: no halo, and
    # the strict spill rule still applies.
    assert deposited_coverage(fill_paths(square, tool), tool).difference(square).area == 0
