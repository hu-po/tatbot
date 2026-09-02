#!/usr/bin/env python3
"""Verify that the fitted tool, the carriage constants and the code agree.

    check_tool_sync.py [--ee-tool <id>]

Two things drift if nobody checks them:

- The CARRIAGE CONSTANTS (rest, retract, contact cap — since 2026-08-30 the
  carriage is the tool's contact axis)
  live in config/trossen/tatbot.yaml and are copied into the batteryA golden,
  the Python follower config defaults and the C++ teleop's compiled defaults.
  Four copies of three numbers; this checks them against tatbot.yaml.
- The MEASURED TIP in config/workspace.yaml against the fitted tool's
  datasheet nominal, and its lean off the mount's bore axis.

It also REPORTS, without changing anything, the workspace floor that the tool
geometry implies — see ``tool_spec.derive_z_floor_m`` for why that number is
usually not yet trustworthy, and why a safety constant is not something to
derive automatically.

Nothing here moves the arm and nothing writes a file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import tool_spec  # noqa: E402

SOURCE = "config/trossen/tatbot.yaml"
# key -> (label, path, regex capturing the value) for every copy of it
CARRIAGE_SITES = {
    "carriage_rest_m": (
        ("config/trossen-batteryA/tatbot.yaml", r"^\s*carriage_rest_m:\s*([0-9.]+)"),
        ("python/lerobot_robot_tatbot/src/lerobot_robot_tatbot/config_tatbot_follower.py",
         r"^\s*carriage_rest_m:\s*float\s*=\s*([0-9.]+)"),
        ("cpp/teleop/wxai_teleop.cpp", r"CARRIAGE_REST_M\s*=\s*([0-9.]+)"),
    ),
    "carriage_retract_m": (
        ("config/trossen-batteryA/tatbot.yaml", r"^\s*carriage_retract_m:\s*([0-9.]+)"),
        ("python/lerobot_robot_tatbot/src/lerobot_robot_tatbot/config_tatbot_follower.py",
         r"^\s*carriage_retract_m:\s*float\s*=\s*([0-9.]+)"),
        ("cpp/teleop/wxai_teleop.cpp", r"CARRIAGE_RETRACT_M\s*=\s*([0-9.]+)"),
    ),
    "carriage_contact_deflect_m": (
        ("config/trossen-batteryA/tatbot.yaml", r"^\s*carriage_contact_deflect_m:\s*([0-9.]+)"),
        ("python/lerobot_robot_tatbot/src/lerobot_robot_tatbot/config_tatbot_follower.py",
         r"^\s*carriage_contact_deflect_m:\s*float\s*=\s*([0-9.]+)"),
        ("cpp/teleop/wxai_teleop.cpp", r"CARRIAGE_CONTACT_DEFLECT_M\s*=\s*([0-9.]+)"),
    ),
    "carriage_contact_cap_n": (
        ("config/trossen-batteryA/tatbot.yaml", r"^\s*carriage_contact_cap_n:\s*([0-9.]+)"),
        ("python/lerobot_robot_tatbot/src/lerobot_robot_tatbot/config_tatbot_follower.py",
         r"^\s*carriage_contact_cap_n:\s*float\s*=\s*([0-9.]+)"),
        ("cpp/teleop/wxai_teleop.cpp", r"CARRIAGE_CONTACT_CAP_N\s*=\s*([0-9.]+)"),
    ),
}


def check_carriage_constants() -> list[str]:
    """Every copy of the carriage constants against config/trossen/tatbot.yaml."""
    source_text = (REPO / SOURCE).read_text()
    problems = []
    for key, sites in CARRIAGE_SITES.items():
        match = re.search(rf"^\s*{key}:\s*([0-9.]+)", source_text, re.MULTILINE)
        if match is None:
            problems.append(f"{SOURCE}: no {key} (pattern moved?)")
            continue
        value = float(match.group(1))
        print(f"  ok  {SOURCE}: {key} {match.group(1)}")
        for relpath, pattern in sites:
            path = REPO / relpath
            if not path.is_file():
                problems.append(f"{relpath}: missing")
                continue
            found = re.search(pattern, path.read_text(), re.MULTILINE)
            if found is None:
                problems.append(f"{relpath}: no {key} found (pattern moved?)")
            elif float(found.group(1)) != value:
                problems.append(f"{relpath}: {key} {found.group(1)} != {value} from {SOURCE}")
            else:
                print(f"  ok  {relpath}: {key} {found.group(1)}")
    return problems


def check_tip(spec, workspace) -> list[str]:
    measured = tool_spec.tip_offset_m(workspace)
    if measured is None:
        print("  --  no measured tip offset yet (run a touch-off)")
        return []
    error_m = tool_spec.tip_offset_error_m(spec, measured)
    lean = tool_spec.axis_lean_deg(measured)
    line = (f"tip offset sits {error_m * 1000:.1f} mm from {spec.tool_id}'s nominal "
            f"{spec.protrusion_m * 1000:.0f} mm protrusion "
            f"(tolerance {spec.tip_tolerance_m * 1000:.0f} mm), leaning {lean:.1f} deg "
            f"off the bore axis (tolerance {spec.seat_tolerance_deg:.0f})")
    if error_m > spec.tip_tolerance_m or lean > spec.seat_tolerance_deg:
        return [f"config/workspace.yaml: {line}"]
    print(f"  ok  {line}")
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ee-tool", "--tool-id", dest="tool_id", default=None, help="override the fitted tool")
    ap.add_argument("--arm", default="right")
    args = ap.parse_args()

    workspace = tool_spec.read_workspace(REPO)
    spec = (tool_spec.load_tool(args.tool_id, REPO) if args.tool_id
            else tool_spec.load_active_tool(REPO, args.arm, workspace))

    print(f"tool {spec.summary()}")
    print(f"  source {spec.source.relative_to(REPO)} ({spec.sha256[:12]})")
    print(f"  mount  {spec.mount or 'NONE — cannot be fitted'}")

    problems = check_carriage_constants() + check_tip(spec, workspace)
    geometry = tool_spec.resolved_tool_geometry(spec, workspace, args.arm, REPO)
    uncertainty = (f", uncertainty {geometry.contact_uncertainty_m * 1000:.3f} mm"
                   if geometry.contact_uncertainty_m is not None else "")
    print(f"  body   {geometry.source} / {geometry.body_pose_status}; endpoint/TCP error "
          f"{geometry.alignment_error_m * 1000:.3f} mm")
    print(f"  TCP    {geometry.contact_status}{uncertainty}")
    if geometry.contact_qualification_error:
        print(f"  --  contact qualification: {geometry.contact_qualification_error}")
    requested_body_status = (workspace.get(args.arm) or {}).get("tool_body_status")
    if requested_body_status == "qualified" and geometry.status != "qualified":
        problems.append(
            "config/workspace.yaml: tool body claims qualified but its evidence "
            f"does not revalidate: {geometry.qualification_error or 'unknown reason'}")
    if not spec.mounted:
        problems.append(f"{spec.source.relative_to(REPO)}: mount is none — this tool has no "
                        "mount on the arm and cannot be fitted, calibrated or flown")
    if not spec.verified:
        problems.append(
            f"{spec.source.relative_to(REPO)}: measured.status is "
            f"{spec.measured.get('status', 'absent')!r}, not 'measured' — these numbers "
            "came from vendor copy or a guess. Put calipers on the tool and set the "
            "status before flying it.")
    if spec.mass_kg is None:
        print("  --  mass_kg not recorded")

    floor = tool_spec.derive_z_floor_m(spec, workspace, args.arm)
    if floor["trustworthy"]:
        print(f"\nz_floor_m derivable as {floor['z_floor_m']:.4f} m — {floor['note']}")
        print("  compare against config_tatbot_follower.z_floor_m and change it "
              "deliberately; a safety floor is not derived behind your back.")
    else:
        print("\nz_floor_m not derivable yet:")
        for reason in floor["reasons"]:
            print(f"  - {reason}")

    if problems:
        print("\nMISMATCH:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nall tool constants agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
