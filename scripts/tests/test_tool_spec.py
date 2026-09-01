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
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import il_touchoff  # noqa: E402
import tool_spec  # noqa: E402

FITTED = "lutin-ballpoint-dot"

# The fitted tool's visuals: body cylinder, tip mesh, tip detail sphere, then four rings.
# (kind, z, dimensions). Updated 2026-08-25 when the operator measured the
# assembly at 115 mm with a 35 mm cartridge — the body grew 70 -> 80 mm and the
# cone shrank 40 -> 35 mm. Updated 2026-08-27 with tip_detail ball.
MEASURED_VISUALS = [
    ("cylinder", -0.015, 0.08, 0.0145),
    ("mesh", 0.025, "pen_tip.stl"),
    ("sphere", 0.0589, 0.0011),
    ("cylinder", -0.036, 0.008, 0.0153),
    ("cylinder", -0.020, 0.006, 0.0153),
    ("cylinder", -0.004, 0.010, 0.0155),
    ("cylinder", 0.012, 0.006, 0.0153),
]


@pytest.fixture(scope="module")
def spec():
    return tool_spec.load_tool(FITTED, REPO)


def test_fitted_tool_matches_the_operator_measurements(spec):
    assert spec.protrusion_m == pytest.approx(0.060)
    assert spec.back_m == pytest.approx(-0.055)
    # 115 mm assembled = 80 mm machine body + 35 mm cartridge
    assert (spec.profile[-1][0] - spec.profile[0][0]) == pytest.approx(0.115)
    assert (spec.profile[1][0] - spec.profile[0][0]) == pytest.approx(0.080)
    assert (spec.profile[-1][0] - spec.profile[1][0]) == pytest.approx(0.035)
    assert spec.body_radius_m == pytest.approx(0.0145)
    assert spec.mount == "tool_mount"
    assert spec.nominal_tip_offset_m == (0.0, 0.0, pytest.approx(0.060))
    assert spec.prompt_phrase == "using pen tip"


def test_geometry_matches_the_measured_assembly(spec):
    parts = spec.geometry_parts()
    assert len(parts) == len(MEASURED_VISUALS)
    for part, expected in zip(parts, MEASURED_VISUALS, strict=True):
        assert part["kind"] == expected[0]
        assert part["z"] == pytest.approx(expected[1])
        if part["kind"] == "mesh":
            assert part["mesh"] == expected[2]
        elif part["kind"] == "sphere":
            assert part["radius"] == pytest.approx(expected[2])
        else:
            assert part["length"] == pytest.approx(expected[2])
            assert part["radius"] == pytest.approx(expected[3])


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


@pytest.mark.parametrize("extra,message", [
    ("contact: true\ntcp_z_m: 0.090\n", "cannot float"),
    ("contact: false\ntcp_z_m: 0.030\n", "buried in the tool"),
    ("grip_force_n: 33.0\n", "schema-1 field"),
])
def test_an_impossible_tool_is_refused_at_load(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        _write_tool(tmp_path, "bad", extra)


def test_an_unmeasured_datasheet_says_so(tmp_path, spec):
    # The fitted tool went measured with the 2026-08-31 touch-off
    # (fixed EE mount) and must no longer
    # flag itself; a datasheet that is still a guess has to say so, and does.
    assert spec.verified and "UNVERIFIED" not in spec.summary()
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


def test_a_tool_without_a_mount_cannot_be_flown():
    """The laser pen has no adapter for the bore (D8): everything that
    would fit it refuses, and says why."""
    laser = tool_spec.load_tool("picosecond-laser-pen", REPO)
    assert not laser.mounted
    with pytest.raises(tool_spec.ToolMountError, match="no mount"):
        laser.mount_frame("right")
    with pytest.raises(tool_spec.ToolMountError):
        tool_spec.require_stated_tool("picosecond-laser-pen", REPO, workspace={})
    ballpoint = tool_spec.load_tool(FITTED, REPO)
    assert ballpoint.mount_frame("right") == "right/tool_mount"


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
    workspace = tool_spec.read_workspace(REPO)
    tool_spec.write_dataset_tool_metadata(tmp_path, spec, workspace)
    payload = json.loads((tmp_path / "meta" / "tool.json").read_text())
    assert payload["tool_id"] == FITTED
    assert payload["spec_sha256"] == spec.sha256
    # the whole datasheet, inlined: readable after the file itself is retired
    assert payload["spec"]["profile"] == [list(p) for p in spec.profile]
    measured = tool_spec.tip_offset_m(workspace)
    assert payload["tip_offset_m"] == (pytest.approx(list(measured)) if measured else None)
    assert payload["tip_link"] == "right/tattoo_needle"
    assert payload["tip_frame"] == "right/tool_mount"
    assert payload["embodiment"] == "fixed-mount-v2"
    assert tool_spec.read_dataset_tool_metadata(tmp_path) == payload


def test_the_z_floor_is_not_derived_from_a_surface_nobody_touched(spec):
    """paper_plane_z after a palette-only session is the palette, not paper."""
    palette_only = {"right": {"paper_plane_z": 0.0655, "tip_frame": "right/tool_mount",
                              "pen_tip_offset_x": -0.007, "pen_tip_offset_y": -0.002,
                              "pen_tip_offset_z": 0.063,
                              "touchoff": {"n_pad": 0}}}
    result = tool_spec.derive_z_floor_m(spec, palette_only)
    assert result["trustworthy"] is False
    assert result["z_floor_m"] is None
    assert any("n_pad" in reason for reason in result["reasons"])

    touched = json.loads(json.dumps(palette_only))
    touched["right"]["touchoff"]["n_pad"] = 4
    result = tool_spec.derive_z_floor_m(spec, touched, margin_m=0.010)
    assert result["trustworthy"] is True
    reach = (0.007 ** 2 + 0.002 ** 2 + 0.063 ** 2) ** 0.5
    assert result["z_floor_m"] == pytest.approx(0.0655 - reach - 0.010, abs=1e-6)


def test_a_non_contact_tool_is_modelled_at_its_working_point(tmp_path):
    """End to end: what the touch-off plants is the aperture, but the link the
    URDF exposes as the TCP has to sit at the focus, standoff further along."""
    sys.path.insert(0, str(REPO / "scripts"))
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
    measured_len = sum(v * v for v in measured) ** 0.5
    assert reach[2] == pytest.approx(measured_len + laser.standoff_m)


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


def test_every_fitted_tool_names_a_substrate_that_exists():
    """A tool and a substrate are a pair: the ballpoint only ever draws on the
    paper pad, the laser and the 3RL only ever work on the silicone skin. A
    tool naming a substrate nobody described would have the sim guessing its
    working area from context."""
    for tool_id in tool_spec.list_tools(REPO):
        spec = tool_spec.load_tool(tool_id, REPO)
        sub = tool_spec.substrate_for(spec, REPO)
        assert sub.width_m > 0 and sub.height_m > 0 and sub.thickness_m > 0
        want = "paper_pad" if spec.kind == "ballpoint_pen" else "silicon_skin"
        assert sub.name == want, (tool_id, spec.kind, sub.name)


def test_a_substrate_texture_is_square_pixelled():
    """Kernels are sized from one texels-per-metre number, so a substrate whose
    texture is not square-pixelled would skew every stamp on it by a constant
    nobody measured."""
    for name in ("paper_pad", "silicon_skin"):
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
    assert "calib_sweep.sh" in str(wrong.value), "refusal must carry the fix"

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
    assert "config/trossen-batteryA/tatbot.yaml" in site_paths
    assert "python/lerobot_robot_tatbot/src/lerobot_robot_tatbot/config_tatbot_follower.py" in site_paths
    assert "cpp/teleop/wxai_teleop.cpp" in site_paths


def test_check_carriage_constants_detects_deflect_mismatch(monkeypatch):
    """If carriage_contact_deflect_m drifts in any site, check_carriage_constants flags it."""
    import check_tool_sync

    real_read_text = Path.read_text
    target_path = check_tool_sync.REPO / "config/trossen-batteryA/tatbot.yaml"

    def mock_read_text(self, *args, **kwargs):
        content = real_read_text(self, *args, **kwargs)
        if self == target_path:
            content = content.replace("carriage_contact_deflect_m: 0.002", "carriage_contact_deflect_m: 0.005")
        return content

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    problems = check_tool_sync.check_carriage_constants()
    assert any("carriage_contact_deflect_m" in p and "0.005 != 0.002" in p for p in problems)
