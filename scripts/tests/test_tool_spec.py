"""Pin the tool registry: the datasheet must reproduce what it replaced.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_tool_spec.py

The registry was introduced by MOVING constants out of
python/tatbot_sim/src/tatbot_sim/urdf.py into config/tools/lutin-ballpoint-dot.yaml.
The whole claim of that refactor is that the sim's geometry did not change, so
the first test here is a literal transcript of the visuals the old code built.
If a datasheet edit is meant to change the sim, that expectation changes with
it — deliberately, in the same commit.

The rest guards the properties the design depends on: a dataset's tool stamp
is self-contained, a calibration cannot be filed under the wrong tool, and a
safety floor is never derived from a surface nobody touched.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

import il_touchoff  # noqa: E402
import tool_spec  # noqa: E402

FITTED = "lutin-ballpoint-dot"


def test_datasheet_snapshot_is_parsed_without_a_second_file_read(tmp_path):
    raw = (REPO/'config/tools'/f'{FITTED}.yaml').read_bytes()
    frozen = tool_spec.load_tool(FITTED, tmp_path, snapshot=raw)
    assert frozen.tool_id == FITTED and frozen.touchoff_nominal_m == tool_spec.load_tool(FITTED, REPO).touchoff_nominal_m


def test_workspace_yaml_duplicate_key_is_refused():
    with pytest.raises(ValueError, match='duplicate key'):
        tool_spec.parse_simple_yaml('right:\n  tool_id: first\n  tool_id: second\n')

def calibrated_workspace(spec):
    """Synthetic good pivot fixture independent of the installed calibration."""
    return {"right": {
        "tool_id": spec.tool_id, "tip_frame": "right/tool_mount",
        "pen_tip_offset_x": 0.001, "pen_tip_offset_y": 0.002,
        "pen_tip_offset_z": spec.protrusion_m,
        "touchoff": {"n_pad": 9, "cond": 6.2, "spread_deg": 141.1,
                     "residual_mm": 1.997, "holdout_mm": 1.970,
                     "tip_loo_max_mm": 1.020},
    }}


@pytest.fixture(scope="module")
def spec():
    return tool_spec.load_tool(FITTED, REPO)


def test_fitted_tool_matches_the_operator_measurements(spec):
    assert spec.protrusion_m == pytest.approx(0.073)
    assert spec.back_m == pytest.approx(-0.042)
    # 115 mm assembled = 80 mm machine body + 35 mm cartridge
    assert (spec.profile[-1][0] - spec.profile[0][0]) == pytest.approx(0.115)
    assert (spec.profile[-2][0] - spec.profile[0][0]) == pytest.approx(0.080)
    assert (spec.profile[-1][0] - spec.profile[-2][0]) == pytest.approx(0.035)
    assert spec.body_radius_m == pytest.approx(0.0165)
    assert spec.mount == "tool_mount"
    assert spec.nominal_tip_offset_m == (0.0, 0.0, pytest.approx(0.073))
    assert spec.prompt_phrase == "using pen tip"


def test_v17_taper_does_not_repeat_the_cartridge_mesh(spec):
    parts = spec.geometry_parts()
    meshes = [part for part in parts if part["kind"] == "mesh"]
    assert len(meshes) == 1
    assert meshes[0]["z"] == pytest.approx(0.038)
    assert spec.rings == ()
    body = [part for part in parts if part["kind"] == "cylinder"]
    assert all(part["color"] == spec.body_color for part in body)
    assert all(0.016 <= part["radius"] <= 0.0165 for part in body)


def test_a_tool_without_a_mesh_renders_its_taper_as_a_stack(tmp_path):
    """A new pen is describable with calipers alone — no scan, no mesh."""
    (tmp_path / "config" / "tools").mkdir(parents=True)
    (tmp_path / "config" / "tools" / "bare.yaml").write_text(
        "schema_version: 2\ntool_id: bare\nkind: rotary_pen\n"
        'display_name: "Bare"\nprompt_phrase: "using a needle"\n'
        "profile: [[-0.04, 0.010], [0.01, 0.010], [0.05, 0.001]]\n")
    bare = tool_spec.load_tool("bare", tmp_path)
    kinds = [p["kind"] for p in bare.geometry_parts()]
    assert kinds == ["cylinder"] * (1 + tool_spec.TAPER_STEPS)
    radii = [p["radius"] for p in bare.geometry_parts()[1:]]
    assert radii == sorted(radii, reverse=True)  # it tapers toward the tip


@pytest.mark.parametrize("profile,message", [
    ("[[-0.05, 0.01], [-0.02, 0.001]]", "must protrude"),
    ("[[-0.05, 0.01], [-0.06, 0.01], [0.06, 0.001]]", "strictly increase"),
    ("[[-0.05, 0.0]]", "at least two"),
])
def test_a_nonsense_profile_is_refused_at_load(tmp_path, profile, message):
    (tmp_path / "config" / "tools").mkdir(parents=True)
    (tmp_path / "config" / "tools" / "bad.yaml").write_text(
        "schema_version: 2\ntool_id: bad\nkind: rotary_pen\n"
        f'display_name: "Bad"\nprompt_phrase: "x"\nprofile: {profile}\n')
    with pytest.raises(ValueError, match=message):
        tool_spec.load_tool("bad", tmp_path)


def test_the_yaml_subset_reads_nesting_arrays_and_quoted_hashes():
    parsed = tool_spec.parse_simple_yaml(
        'name: "a # not a comment"   # this one is\n'
        "profile: [\n  [-0.05, 0.0145],  # back\n  [0.06, 0.001]\n]\n"
        "measured:\n  utc: 2026-08-22T20:04:37Z\n  n: 3\n  missing: null\n")
    assert parsed["name"] == "a # not a comment"
    assert parsed["profile"] == [[-0.05, 0.0145], [0.06, 0.001]]
    assert parsed["measured"] == {"utc": "2026-08-22T20:04:37Z", "n": 3, "missing": None}


def _write_tool(tmp_path, name, extra=""):
    (tmp_path / "config" / "tools").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "tools" / f"{name}.yaml").write_text(
        f"schema_version: 2\ntool_id: {name}\nkind: rotary_pen\n"
        f'display_name: "T"\nprompt_phrase: "x"\n'
        "profile: [[-0.04, 0.010], [0.01, 0.010], [0.05, 0.001]]\n" + extra)
    return tool_spec.load_tool(name, tmp_path)


def test_a_non_contact_tool_works_at_a_distance_from_its_own_body(tmp_path):
    """A laser's focus is in free space: past the body, with nothing at it."""
    laser = _write_tool(tmp_path, "beam", "contact: false\ntcp_z_m: 0.090\n")
    assert laser.body_tip_z_m == pytest.approx(0.05)   # the aperture
    assert laser.protrusion_m == pytest.approx(0.090)  # the working point
    assert laser.standoff_m == pytest.approx(0.040)
    # a touch-off plants the aperture, not the focus — so they are checked
    # against different nominals, or every good calibration refuses
    assert laser.touchoff_nominal_m[2] == pytest.approx(0.05)
    assert laser.nominal_tip_offset_m[2] == pytest.approx(0.090)


def test_the_working_point_extends_along_the_measured_direction(tmp_path):
    laser = _write_tool(tmp_path, "beam", "contact: false\ntcp_z_m: 0.090\n")
    measured = (0.05, 0.0, 0.0)
    tcp = tool_spec.tcp_from_touchoff_m(laser, measured)
    assert tcp[0] == pytest.approx(0.090)
    # off-axis: the standoff follows where the tool actually points
    tilted = (0.03, 0.04, 0.0)  # 50 mm long
    tcp = tool_spec.tcp_from_touchoff_m(laser, tilted)
    assert sum(v * v for v in tcp) ** 0.5 == pytest.approx(0.090)
    assert tcp[1] / tcp[0] == pytest.approx(tilted[1] / tilted[0])


def test_a_contact_tool_working_point_is_its_own_tip(tmp_path, spec):
    plain = _write_tool(tmp_path, "plain")
    assert plain.standoff_m == 0
    assert plain.touchoff_nominal_m == plain.nominal_tip_offset_m
    measured = (0.063, -0.007, -0.002)
    assert tool_spec.tcp_from_touchoff_m(spec, measured) == measured


def test_resolved_contact_geometry_puts_the_visible_body_at_the_tcp(spec):
    """A touch-off moves the physical body endpoint, not an invisible child."""
    workspace = calibrated_workspace(spec)
    geometry = tool_spec.resolved_tool_geometry(spec, workspace)
    measured = tool_spec.tip_offset_m(workspace)
    assert measured is not None
    assert geometry.source == "touch-axis-inferred"
    assert geometry.status == "contact-qualified"
    assert geometry.contact_status == "pivot-calibrated"
    assert geometry.body_pose_status == "axis-inferred"
    assert geometry.contact_qualification_error is None
    assert geometry.contact_uncertainty_m == pytest.approx(0.001997)
    assert geometry.touch_offset_m == pytest.approx(measured)
    assert geometry.body_tip_offset_m == pytest.approx(measured)
    assert geometry.tcp_offset_m == pytest.approx(measured)
    assert geometry.alignment_error_m <= tool_spec.CONTACT_ALIGNMENT_TOLERANCE_M
    assert spec.contact_radius_m == pytest.approx(0.00025)


def test_pivot_contact_qualification_is_separate_from_body_pose(spec):
    workspace = {"right": {
        "tool_id": spec.tool_id,
        "tip_frame": "right/tool_mount",
        "pen_tip_offset_x": 0.0,
        "pen_tip_offset_y": 0.0,
        "pen_tip_offset_z": 0.060,
        "touchoff": {
            "n_plate": 0, "n_pad": 9, "cond": 7.0,
            "residual_mm": 1.0, "holdout_mm": 4.0,
            "tip_loo_max_mm": 1.5, "spread_deg": 60.0,
        },
    }}
    geometry = tool_spec.resolved_tool_geometry(spec, workspace)
    assert geometry.status == "contact-qualified"
    assert geometry.contact_status == "pivot-calibrated"
    assert geometry.body_pose_status == "axis-inferred"
    assert geometry.contact_uncertainty_m == pytest.approx(0.004)


def test_tip_calibration_delta_moves_body_and_tcp_together(spec):
    workspace = calibrated_workspace(spec)
    central = tool_spec.resolved_tool_geometry(spec, workspace)
    delta = (0.001, -0.002, 0.003)
    varied = tool_spec.resolved_tool_geometry(
        spec, workspace, tip_delta_m=delta)
    assert varied.contact_status == "pivot-calibrated"
    assert varied.body_pose_status == "axis-inferred"
    assert varied.calibration_delta_m == delta
    assert varied.touch_offset_m == pytest.approx(
        tuple(a + b for a, b in zip(central.touch_offset_m, delta, strict=True)))
    assert varied.body_tip_offset_m == pytest.approx(varied.touch_offset_m)
    assert varied.tcp_offset_m == pytest.approx(varied.touch_offset_m)
    assert varied.alignment_error_m <= tool_spec.CONTACT_ALIGNMENT_TOLERANCE_M


def test_tip_calibration_delta_requires_a_measured_touch_off(tmp_path):
    plain = _write_tool(tmp_path, "plain-delta")
    with pytest.raises(ValueError, match="measured mount-frame touch-off"):
        tool_spec.resolved_tool_geometry(
            plain, workspace=None, repo=tmp_path, tip_delta_m=(0.001, 0.0, 0.0))


@pytest.mark.parametrize("field,value,message", [
    ("cond", 51.0, "condition number"),
    ("spread_deg", 29.0, "rotation spread"),
    ("residual_mm", 3.6, "residual"),
])
def test_bad_pivot_observability_remains_provisional(spec, field, value, message):
    receipt = {"n_plate": 0, "n_pad": 9, "cond": 7.0,
               "residual_mm": 1.0, "holdout_mm": 0.5,
               "spread_deg": 60.0}
    receipt[field] = value
    workspace = {"right": {
        "tool_id": spec.tool_id,
        "tip_frame": "right/tool_mount",
        "pen_tip_offset_x": 0.0,
        "pen_tip_offset_y": 0.0,
        "pen_tip_offset_z": 0.060,
        "touchoff": receipt,
    }}
    geometry = tool_spec.resolved_tool_geometry(spec, workspace)
    assert geometry.status == "provisional"
    assert geometry.contact_status == "unqualified"
    assert message in geometry.contact_qualification_error


def test_explicit_body_pose_without_report_cannot_claim_qualification(tmp_path):
    spec = _write_tool(tmp_path, "posed")
    workspace = {"right": {
        "tool_id": spec.tool_id,
        "tip_frame": "right/tool_mount",
        "pen_tip_offset_x": 0.001,
        "pen_tip_offset_y": 0.002,
        "pen_tip_offset_z": 0.050,
        "tool_body_status": "qualified",
        "tool_body_utc": "2026-09-01T12:00:00Z",
        "tool_body_method": "reseat-axis-study-v1",
        "tool_body_frame": "right/tool_mount",
        "tool_body_origin_x": 0.001,
        "tool_body_origin_y": 0.002,
        "tool_body_origin_z": 0.0,
        "tool_body_rpy_x": 0.0,
        "tool_body_rpy_y": 0.0,
        "tool_body_rpy_z": 0.0,
    }}
    geometry = tool_spec.resolved_tool_geometry(spec, workspace, repo=tmp_path)
    assert geometry.source == "touch-axis-inferred"
    assert geometry.status == "provisional"
    assert "no report path" in geometry.qualification_error
    assert geometry.body_tip_offset_m == pytest.approx((0.001, 0.002, 0.050))
    assert geometry.alignment_error_m == pytest.approx(0.0)


def test_body_coordinates_without_qualification_remain_provisional(tmp_path):
    spec = _write_tool(tmp_path, "unqualified-pose")
    workspace = {"right": {
        "tool_id": spec.tool_id,
        "tip_frame": "right/tool_mount",
        "pen_tip_offset_x": 0.001,
        "pen_tip_offset_y": 0.002,
        "pen_tip_offset_z": 0.050,
        "tool_body_frame": "right/tool_mount",
        "tool_body_origin_x": 0.001,
        "tool_body_origin_y": 0.002,
        "tool_body_origin_z": 0.0,
        "tool_body_rpy_x": 0.0,
        "tool_body_rpy_y": 0.0,
        "tool_body_rpy_z": 0.0,
    }}
    geometry = tool_spec.resolved_tool_geometry(spec, workspace, repo=tmp_path)
    assert geometry.source == "touch-axis-inferred"
    assert geometry.status == "provisional"


@pytest.mark.parametrize("extra,message", [
    ("contact: true\ntcp_z_m: 0.090\n", "cannot float"),
    ("contact: false\ntcp_z_m: 0.030\n", "buried in the tool"),
    ("grip_force_n: 33.0\n", "schema-1 field"),
])
def test_an_impossible_tool_is_refused_at_load(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        _write_tool(tmp_path, "bad", extra)


def test_the_drawn_line_is_a_complete_measurement_record_or_absent(tmp_path, spec):
    """`line:` is what the tool lays on its substrate, recorded like `measured:`.
    The planning width and the schedule's dedup footprint read it; nothing
    about motion does. A partial or out-of-range block is refused at load."""
    bare = _write_tool(tmp_path, "bare")
    assert bare.line == {} and bare.line_width_m is None
    block = ('line:\n  width_mm: 0.5\n  status: assumed\n  utc: 2026-09-15\n'
             '  method: "datasheet ball diameter"\n  substrate: paper_pad\n')
    recorded = _write_tool(tmp_path, "recorded", block)
    assert recorded.line_width_m == pytest.approx(0.0005)
    assert recorded.line["status"] == "assumed"
    for extra, message in [
        (block.replace("  utc: 2026-09-15\n", ""), "missing utc"),
        (block.replace("0.5", "0.02"), "width_mm must be in"),
        (block.replace("0.5", "2.5"), "width_mm must be in"),
        (block.replace("assumed", "guessed"), "measured or assumed"),
    ]:
        with pytest.raises(ValueError, match=message):
            _write_tool(tmp_path, "bad", extra)
    # The fitted ballpoint records its line, and says it is assumed until a
    # swatch is measured.
    assert spec.line_width_m is not None and spec.line["status"] in ("measured", "assumed")


def test_the_stroke_is_where_the_running_tip_travels_or_absent(tmp_path, spec):
    """stroke_mm and tip_out_at_top_mm place a rotary machine's working tip: a touch jams it at the top of its
    stroke, tip_out_at_top_mm out of the tube. They are optional, and a nonsense value is refused at load."""
    assert (spec.stroke_m, spec.tip_out_at_top_m) == (pytest.approx(0.0035), pytest.approx(0.002))
    bare = _write_tool(tmp_path, "bare")
    assert bare.stroke_m is None and bare.tip_out_at_top_m is None
    assert _write_tool(tmp_path, "recessed", "stroke_mm: 4.0\ntip_out_at_top_mm: -1.5\n").tip_out_at_top_m == -0.0015
    for extra, message in [("stroke_mm: 0\n", "stroke_mm must be in"), ("stroke_mm: 12\n", "stroke_mm must be in"),
                           ("tip_out_at_top_mm: 9\n", "tip_out_at_top_mm must be within"),
                           ("stroke_mm: long\n", "number of millimetres")]:
        with pytest.raises(ValueError, match=message):
            _write_tool(tmp_path, "bad", extra)


def test_an_unmeasured_datasheet_says_so(tmp_path, spec):
    # The fitted tool went measured with the 2026-08-31 touch-off
    # (fixed EE mount) and must no longer
    # flag itself; a datasheet that is still a guess has to say so, and does.
    assert ("UNVERIFIED" not in spec.summary()) == spec.verified
    guessed = _write_tool(tmp_path, "guessed", "measured:\n  status: nominal\n")
    assert not guessed.verified
    assert "UNVERIFIED" in guessed.summary()


def test_the_yaml_subset_reads_block_scalars_and_refuses_what_it_cannot():
    """Provenance notes are prose and run long. Before block scalars existed
    here, a `>-` note was silently shredded into garbage keys — which is worse
    than any parse error, so the unsupported case now raises."""
    parsed = tool_spec.parse_simple_yaml(
        "measured:\n  note: >-\n    first line # not a comment\n    second line\n"
        "  after: 3\n")
    assert parsed["measured"]["note"] == "first line # not a comment second line"
    assert parsed["measured"]["after"] == 3

    literal = tool_spec.parse_simple_yaml("note: |\n  one\n  two\n")
    assert literal["note"] == "one\ntwo"

    with pytest.raises(ValueError, match="not a `key: value` pair"):
        tool_spec.parse_simple_yaml("tool_id: x\nthis line has no colon\n")


def test_the_fitted_tool_is_named_by_the_workspace():
    """Whatever is fitted, the workspace names it and the name resolves.

    Pinned the ballpoint by name until 2026-08-26, which made a legitimate
    tool swap look like a test failure. What has to hold is the link — that
    the pointer exists and loads — not which tool happens to be in the
    gripper today.
    """
    workspace = tool_spec.read_workspace(REPO)
    fitted = tool_spec.active_tool_id(REPO, workspace=workspace)
    assert fitted in tool_spec.list_tools(REPO), "workspace names an unknown tool"
    assert tool_spec.load_active_tool(REPO, workspace=workspace).tool_id == fitted


def test_tool_id_survives_a_workspace_rewrite():
    """il_touchoff regenerates the whole file; the tool must not fall out."""
    right = {"tool_id": FITTED, "tip_frame": "right/tool_mount", "carriage_m": 0.0,
             "pen_tip_offset_x": 0.01, "pen_tip_offset_y": 0.02,
             "pen_tip_offset_z": 0.03, "paper_plane_z": 0.04, "paper_band_mm": None,
             "ee_contact_z": None, "touchoff": {}}
    rendered = il_touchoff.render_workspace(right)
    parsed = tool_spec.parse_simple_yaml(rendered)
    assert parsed["right"]["tool_id"] == FITTED
    assert parsed["right"]["tip_frame"] == "right/tool_mount"
    assert tool_spec.tip_offset_m(parsed) == pytest.approx((0.01, 0.02, 0.03))
    assert parsed["right"]["tool_body_status"] is None
    assert parsed["right"]["tool_body_report_sha256"] is None


def test_a_gripper_era_tip_offset_reads_as_no_touchoff():
    """Every workspace.yaml before 2026-08-30 solved the tip in
    right/ee_gripper_link. The tool has no fixed relation to that frame any
    more, so those numbers must not be stale-but-close: they are absent."""
    legacy = {"right": {"tool_id": FITTED, "pen_tip_offset_x": 0.0597,
                        "pen_tip_offset_y": -0.0032, "pen_tip_offset_z": -0.0005}}
    assert tool_spec.tip_offset_m(legacy) is None
    wrong_frame = {"right": {**legacy["right"], "tip_frame": "right/ee_gripper_link"}}
    assert tool_spec.tip_offset_m(wrong_frame) is None
    assert tool_spec.derive_z_floor_m(tool_spec.load_tool(FITTED, REPO), legacy)["trustworthy"] is False


def test_a_tool_without_a_mount_cannot_be_flown(tmp_path):
    """A datasheet that says `mount: none` (the laser pen's state until its
    leader mount existed, D8): everything that would fit it refuses, and
    says why."""
    wand = _write_tool(tmp_path, "wand", "mount: none\n")
    assert not wand.mounted
    with pytest.raises(tool_spec.ToolMountError, match="no mount"):
        wand.mount_frame("right")
    with pytest.raises(tool_spec.ToolMountError):
        tool_spec.require_stated_tool("wand", tmp_path, workspace={})
    ballpoint = tool_spec.load_tool(FITTED, REPO)
    assert ballpoint.mount_frame("right") == "right/tool_mount"
    # The laser pen's mount is the leader arm's; the same datasheet field
    # resolves per arm, and nothing but the workspace section says which.
    laser = tool_spec.load_tool("picosecond-laser-pen", REPO)
    assert laser.mounted and laser.mount_frame("left") == "left/tool_mount"


def test_the_tool_axis_is_the_mount_z():
    """The mount origin sits on the bore axis, so the solved tip's direction
    from it is the axis; a crooked tool leans, and 5 deg is the line."""
    assert tool_spec.axis_lean_deg((0.0, 0.0, 0.060)) == pytest.approx(0.0)
    assert tool_spec.axis_lean_deg((0.003, 0.0, 0.060)) == pytest.approx(2.86, abs=0.01)
    assert tool_spec.axis_lean_deg((0.0, 0.010, 0.060)) > tool_spec.AXIS_TOLERANCE_DEG
    # and the URDF rpy that points local +z down that direction has no roll
    roll, pitch, yaw = tool_spec.axis_rpy((0.0, 0.0, 0.060))
    assert (roll, pitch, yaw) == (0.0, 0.0, 0.0)
    roll, pitch, _ = tool_spec.axis_rpy((0.060, 0.0, 0.0))
    assert roll == 0.0 and pitch == pytest.approx(1.5707963)


def test_a_datasheet_owns_its_seat_budgets(tmp_path):
    """A clearance-bore mount grants its seat freedom in the datasheet
    (sweep-20260831_082526: the clamp locates the tool, not the ~33 mm bore);
    a sheet that says nothing keeps the snug-seat defaults."""
    snug = _write_tool(tmp_path, "snug")
    assert snug.seat_tolerance_deg == tool_spec.AXIS_TOLERANCE_DEG
    assert snug.seat_residual_m == 0.0
    loose = _write_tool(tmp_path, "loose",
                        "seat_tolerance_deg: 15.0\nseat_residual_m: 0.0035\n")
    assert loose.seat_tolerance_deg == 15.0
    assert loose.seat_residual_m == pytest.approx(0.0035)
    with pytest.raises(ValueError, match="seat_tolerance_deg"):
        _write_tool(tmp_path, "wild", "seat_tolerance_deg: 60\n")
    with pytest.raises(ValueError, match="seat_residual_m"):
        _write_tool(tmp_path, "broken", "seat_residual_m: 0.05\n")
    # the fitted ballpoint carries the measured seat, and its residual budget
    # widens the pivot gate without touching the point-contact floor
    ballpoint = tool_spec.load_tool(FITTED, REPO)
    assert ballpoint.seat_tolerance_deg == 15.0
    assert ballpoint.seat_residual_m == pytest.approx(0.0035)
    assert il_touchoff.residual_gate_mm(ballpoint) == pytest.approx(3.5)
    assert il_touchoff.residual_gate_mm(snug) == il_touchoff.PIVOT_RESIDUAL_MAX_MM


def test_a_tip_that_does_not_match_the_datasheet_is_refused(spec):
    """The gate that catches a swapped pen nobody wrote down."""
    nominal = spec.nominal_tip_offset_m
    assert il_touchoff.tool_refusal(spec, list(nominal)) is None
    assert tool_spec.tip_offset_error_m(spec, nominal) == pytest.approx(0.0)

    # The live measurement belongs to whichever tool the touch-off used, so
    # check it against THAT datasheet — pairing it with this fixture's tool
    # was only ever right while the two happened to be the same pen.
    workspace = tool_spec.read_workspace(REPO)
    fitted = tool_spec.load_active_tool(REPO, workspace=workspace)
    measured = tool_spec.tip_offset_m(workspace)
    if measured is not None:  # None until the first touch-off in the mount frame
        assert il_touchoff.tool_refusal(fitted, list(measured)) is None  # real fit passes

    far = (nominal[0], nominal[1], nominal[2] + 0.04)
    refusal = il_touchoff.tool_refusal(spec, list(far))
    assert refusal is not None and FITTED in refusal


def test_a_dataset_stamp_carries_the_geometry_not_a_pointer(tmp_path, spec):
    workspace = calibrated_workspace(spec)
    tool_spec.write_dataset_tool_metadata(tmp_path, spec, workspace)
    payload = json.loads((tmp_path / "meta" / "tool.json").read_text())
    assert payload["tool_id"] == FITTED
    assert payload["spec_sha256"] == spec.sha256
    # the whole datasheet, inlined: readable after the file itself is retired
    assert payload["spec"]["profile"] == [list(p) for p in spec.profile]
    measured = tool_spec.tip_offset_m(workspace)
    assert payload["tip_offset_m"] == (pytest.approx(list(measured)) if measured else None)
    assert payload["tool_geometry_version"] == tool_spec.TOOL_GEOMETRY_VERSION
    assert payload["geometry_source"] == "touch-axis-inferred"
    assert payload["geometry_status"] == "contact-qualified"
    assert payload["contact_geometry_status"] == "pivot-calibrated"
    assert payload["body_pose_status"] == "axis-inferred"
    assert payload["body_tip_offset_m"] == pytest.approx(payload["tcp_offset_m"])
    assert payload["alignment_error_m"] <= tool_spec.CONTACT_ALIGNMENT_TOLERANCE_M
    assert payload["tip_link"] == "right/tattoo_needle"
    assert payload["tip_frame"] == "right/tool_mount"
    assert payload["embodiment"] == "fixed-mount-v2"
    assert tool_spec.read_dataset_tool_metadata(tmp_path) == payload


def test_nominal_dataset_geometry_is_concrete_and_labelled(spec):
    payload = tool_spec.dataset_tool_metadata(spec, workspace=None)
    assert payload["geometry_source"] == "datasheet-nominal"
    assert payload["geometry_status"] == "nominal"
    assert payload["geometry_measured"] is False
    assert payload["contact_geometry_status"] == "unqualified"
    assert payload["body_pose_status"] == "nominal"
    assert payload["tip_offset_m"] == pytest.approx(spec.touchoff_nominal_m)
    assert payload["tcp_offset_m"] == pytest.approx(spec.nominal_tip_offset_m)


def test_the_z_floor_is_not_derived_from_a_surface_nobody_touched(spec):
    """paper_plane_z after a palette-only session is the palette, not paper."""
    palette_only = {"right": {"paper_plane_z": 0.0655, "tip_frame": "right/tool_mount",
                              "pen_tip_offset_x": -0.007, "pen_tip_offset_y": -0.002,
                              "pen_tip_offset_z": 0.063,
                              "touchoff": {"n_pad": 0}}}
    result = tool_spec.derive_z_floor_m(spec, palette_only)
    assert result["trustworthy"] is False
    assert result["z_floor_m"] is None
    assert any("no pad touches" in reason for reason in result["reasons"])

    touched = json.loads(json.dumps(palette_only))
    touched["right"]["touchoff"]["n_pad"] = 4
    result = tool_spec.derive_z_floor_m(spec, touched, margin_m=0.010)
    assert result["trustworthy"] is True
    reach = (0.007 ** 2 + 0.002 ** 2 + 0.063 ** 2) ** 0.5
    assert result["z_floor_m"] == pytest.approx(0.0655 - reach - 0.010, abs=1e-6)


def test_a_non_contact_tool_is_modelled_at_its_working_point(tmp_path):
    """End to end: what the touch-off plants is the aperture, but the link the
    URDF exposes as the TCP has to sit at the focus, standoff further along."""
    import gen_tool_urdf

    laser = _write_tool(tmp_path, "beam", "contact: false\ntcp_z_m: 0.090\n")
    measured = (-0.004, 0.001, 0.0498)  # aperture, slightly off the bore axis
    # the gate judges it against the aperture; against the TCP it would refuse
    assert tool_spec.tip_offset_error_m(laser, measured) < laser.tip_tolerance_m
    off_by_standoff = sum(
        (a - b) ** 2 for a, b in zip(measured, laser.nominal_tip_offset_m, strict=True)) ** 0.5
    assert off_by_standoff > laser.tip_tolerance_m

    block = gen_tool_urdf.render_block("right", laser, measured, measured=True)
    reach = [float(v) for v in
             block.split('name="right/tattoo_needle_joint"')[1]
                  .split('xyz="')[1].split('"')[0].split()]
    # The body origin moves so its physical aperture lands on the measured
    # point. In body coordinates the virtual focus remains exactly the
    # datasheet's 90 mm working distance.
    assert reach == pytest.approx([0.0, 0.0, laser.protrusion_m])


def test_generated_tool_blocks_cover_every_fitted_arm():
    """One block per fitted arm, rendered together: asking for a subset used to
    strip the other arm's block from the file without a word."""
    import gen_tool_urdf

    workspace = tool_spec.read_workspace(REPO)
    fitted = gen_tool_urdf.fitted_arms(workspace)
    assert fitted[0] == "right" and "left" in fitted
    rendered = gen_tool_urdf.build()
    for arm in fitted:
        assert f'<link name="{arm}/tattoo_pen">' in rendered, arm
        assert f'<parent link="{arm}/tool_mount"/>' in rendered, arm
    assert rendered == gen_tool_urdf.build(list(fitted))
    with pytest.raises(SystemExit, match="models every fitted tool"):
        gen_tool_urdf.build(["left"])
    with pytest.raises(SystemExit, match="models every fitted tool"):
        gen_tool_urdf.build(["right"])


def test_every_shipped_datasheet_loads():
    """A registry is only useful if every file in it is real. This is the guard
    that stops a broken datasheet reaching a commit."""
    names = tool_spec.list_tools(REPO)
    assert FITTED in names and len(names) >= 2
    for name in names:
        tool = tool_spec.load_tool(name, REPO)
        assert tool.tool_id == name
        assert tool.prompt_phrase and tool.display_name
        # A tool whose numbers nobody checked must SAY so. Which tools have
        # earned their way past that changes as they get measured — the laser
        # did on 2026-08-26 — so what is pinned here is that `verified` is an
        # honest boolean backed by a provenance note, not a fixed roster.
        assert isinstance(tool.verified, bool)
        if tool.verified:
            assert tool.measured.get("status") == "measured", name
            assert tool.measured.get("method"), f"{name}: verified without a method"
            assert tool.measured.get("utc"), f"{name}: verified without a date"










# check_tool_sync reports two different kinds of thing under one heading and
# one exit code: DRIFT (a copy of a number disagreeing with the datasheet) and
# PROVENANCE (a datasheet whose numbers were traced or taken from vendor copy
# rather than measured). Only drift means an artifact is stale, which is what
# this test is about. The provenance line is a real, deliberate refusal — it
# blocks flying an uncharacterised tool — but it is answered with calipers,
# not with a code change, and it must not be silenced by editing the status.
PROVENANCE_MISMATCH = "measured.status is"


def test_the_shipped_urdf_and_constants_match_the_datasheet():
    """The two generated-from-the-datasheet artifacts are not stale."""
    check = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_tool_urdf.py"), "--check"],
        capture_output=True, text=True, cwd=REPO)
    assert check.returncode == 0, f"gen_tool_urdf.py: {check.stdout}{check.stderr}"

    sync = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_tool_sync.py")],
        capture_output=True, text=True, cwd=REPO)
    # Only the MISMATCH section is drift; the "z_floor_m not derivable yet"
    # reasons above it are the expected state before a mount-frame touch-off.
    tail = sync.stdout.split("MISMATCH:", 1)[1] if "MISMATCH:" in sync.stdout else ""
    drift = [line.strip() for line in tail.splitlines()
             if line.strip().startswith("- ")
             and PROVENANCE_MISMATCH not in line]
    assert not drift, "a copy of a tool constant has drifted from its datasheet:\n" + "\n".join(drift)


def test_every_fitted_tool_names_substrates_that_exist():
    """A tool and its substrates are a pair: the ballpoint draws on the two
    gridded paper fixtures, the laser and the 3RL only ever work on the
    silicone skin. A tool naming a substrate nobody described would have the
    sim guessing its working area from context."""
    for tool_id in tool_spec.list_tools(REPO):
        spec = tool_spec.load_tool(tool_id, REPO)
        assert spec.substrates[0] == spec.substrate
        for name in spec.substrates:
            sub = tool_spec.substrate_for(spec, REPO, name=name)
            assert sub.width_m > 0 and sub.height_m > 0 and sub.thickness_m > 0
        want = ("paper_pad", "paper_cylinder") if spec.kind == "ballpoint_pen" else ("silicon_skin",)
        assert spec.substrates == want, (tool_id, spec.kind, spec.substrates)


def test_a_tool_refuses_a_substrate_it_does_not_admit():
    laser = tool_spec.load_tool("picosecond-laser-pen", REPO)
    with pytest.raises(ValueError, match="does not work on substrate"):
        tool_spec.substrate_for(laser, REPO, name="paper_pad")


def test_the_paper_fixtures_are_the_measured_ones():
    """7.5 x 11 in pad, 1 cm thick; 85 mm x 7.5 in cylinder; both on a 1/4 in
    grid, white with faint blue rules (measured 2026-09-11)."""
    pad = tool_spec.load_substrate("paper_pad", REPO)
    assert (pad.width_m, pad.height_m, pad.thickness_m) == (0.1905, 0.2794, 0.010)
    assert pad.shape == "pad" and pad.radius_m is None
    tube = tool_spec.load_substrate("paper_cylinder", REPO)
    assert tube.shape == "cylinder" and tube.radius_m == 0.0425 and tube.height_m == 0.1905
    assert tube.thickness_m == 0.085  # the diameter: the actor origin is the axis
    assert abs(tube.width_m - 1.5 * math.pi * tube.radius_m) < 1e-4  # all but the bottom quarter
    assert tube.width_m < 2 * math.pi * tube.radius_m
    for sub in (pad, tube):
        assert sub.ruled and sub.grid_pitch_m == 0.00635
        assert sub.rgb("base_color", (0, 0, 0)) == (1.0, 1.0, 1.0)
        r, g, b = sub.rgb("rule_color", (0, 0, 0))
        assert b > g > r > 0.5  # faint blue
    assert set(tool_spec.list_substrates(REPO)) == {"paper_pad", "paper_cylinder", "silicon_skin"}


def test_a_cylinder_substrate_is_checked_for_its_own_geometry(tmp_path):
    (tmp_path / "config").mkdir()
    def write(body):
        (tmp_path / "config/substrates.yaml").write_text("schema_version: 1\n" + body)
    common = ("  display_name: t\n  height_m: 0.19\n  texel_cols: 10\n  texel_rows: 10\n"
              "  surface_phrase: on it\n  ruled: true\n  grid_pitch_m: 0.00635\n")
    write("t:\n  shape: cylinder\n  width_m: 0.06\n  thickness_m: 0.085\n" + common)
    with pytest.raises(ValueError, match="diameter_m"):
        tool_spec.load_substrate("t", tmp_path)
    write("t:\n  shape: cylinder\n  diameter_m: 0.085\n  width_m: 0.06\n  thickness_m: 0.010\n" + common)
    with pytest.raises(ValueError, match="thickness_m is its diameter"):
        tool_spec.load_substrate("t", tmp_path)
    write("t:\n  shape: cylinder\n  diameter_m: 0.085\n  width_m: 0.3\n  thickness_m: 0.085\n" + common)
    with pytest.raises(ValueError, match="full"):
        tool_spec.load_substrate("t", tmp_path)
    write("t:\n  shape: pad\n  diameter_m: 0.085\n  width_m: 0.06\n  thickness_m: 0.01\n" + common)
    with pytest.raises(ValueError, match="belongs to a cylinder"):
        tool_spec.load_substrate("t", tmp_path)
    write("t:\n  width_m: 0.06\n  thickness_m: 0.01\n" + common.replace("  grid_pitch_m: 0.00635\n", ""))
    with pytest.raises(ValueError, match="grid_pitch_m"):
        tool_spec.load_substrate("t", tmp_path)


def test_a_substrate_texture_is_square_pixelled():
    """Kernels are sized from one texels-per-metre number, so a substrate whose
    texture is not square-pixelled would skew every stamp on it by a constant
    nobody measured."""
    for name in ("paper_pad", "paper_cylinder", "silicon_skin"):
        sub = tool_spec.load_substrate(name, REPO)
        px_x = sub.texel_cols / sub.width_m
        px_y = sub.texel_rows / sub.height_m
        assert abs(px_x - px_y) / px_x < 0.01, (name, px_x, px_y)
        assert 2000 < sub.texel_per_m < 2800, (name, sub.texel_per_m)


def test_an_unknown_substrate_says_which_ones_exist():
    with pytest.raises(ValueError, match="unknown substrate"):
        tool_spec.load_substrate("forearm", REPO)


def test_stated_tool_is_required_and_cross_checked(tmp_path):
    """The tool is an argument, not an inference.

    Two ways to be wrong, both refusals: saying nothing (which used to inherit
    whatever workspace.yaml named — the PREVIOUS tool after a swap), and saying
    something that contradicts the calibration (which mixes one tool's
    datasheet with another tool's measured geometry).
    """
    workspace = {"right": {"tool_id": "lutin-ballpoint-dot"}}

    with pytest.raises(tool_spec.ToolMismatchError) as unstated:
        tool_spec.require_stated_tool(None, REPO, "right", workspace)
    # The remedy has to name the real options, not just complain.
    assert "lutin-ballpoint-dot" in str(unstated.value)
    assert "picosecond-laser-pen" in str(unstated.value)

    with pytest.raises(tool_spec.ToolMismatchError) as wrong:
        tool_spec.require_stated_tool("lutin-3rl-bugpin", REPO, "right", workspace)
    assert "measured with" in str(wrong.value)
    assert "ros calib run" in str(wrong.value), "refusal must carry the fix"

    # Agreeing is the only way through.
    spec = tool_spec.require_stated_tool(
        "lutin-ballpoint-dot", REPO, "right", workspace)
    assert spec.tool_id == "lutin-ballpoint-dot"

    # An uncalibrated workspace cannot contradict anything, so a stated tool
    # stands on its own rather than being blocked by a missing pointer.
    assert tool_spec.require_stated_tool(
        "lutin-3rl-bugpin", REPO, "right", {}).tool_id == "lutin-3rl-bugpin"


def test_carriage_contact_deflect_m_in_carriage_sites():
    """check_tool_sync must verify carriage_contact_deflect_m across all copies."""
    import check_tool_sync

    assert "carriage_contact_deflect_m" in check_tool_sync.CARRIAGE_SITES
    sites = check_tool_sync.CARRIAGE_SITES["carriage_contact_deflect_m"]
    site_paths = [relpath for relpath, _ in sites]
    assert "python/lerobot_robot_tatbot/src/lerobot_robot_tatbot/config_tatbot_follower.py" in site_paths
    assert "cpp/teleop/wxai_teleop.cpp" in site_paths


def test_check_carriage_constants_detects_deflect_mismatch(monkeypatch):
    """If carriage_contact_deflect_m drifts in any site, check_carriage_constants flags it."""
    import check_tool_sync

    real_read_text = Path.read_text
    target_path = check_tool_sync.REPO / "python/lerobot_robot_tatbot/src/lerobot_robot_tatbot/config_tatbot_follower.py"

    def mock_read_text(self, *args, **kwargs):
        content = real_read_text(self, *args, **kwargs)
        if self == target_path:
            content = content.replace("carriage_contact_deflect_m: float = 0.002",
                                      "carriage_contact_deflect_m: float = 0.005")
        return content

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    problems = check_tool_sync.check_carriage_constants()
    assert any("carriage_contact_deflect_m" in p and "0.005 != 0.002" in p for p in problems)
