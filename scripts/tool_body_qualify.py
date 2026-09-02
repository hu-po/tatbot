#!/usr/bin/env python3
"""Validate and bind an independent tool-body remove/reseat study.

This script only reads measurement files and, with ``--write``, updates tracked
calibration evidence.  It never connects to or commands an arm.

    scripts/tatbot --ee-tool lutin-ballpoint-dot tool qualify-body -- \
      --report /path/to/body-reseat-report.json
    scripts/tatbot --ee-tool lutin-ballpoint-dot tool qualify-body -- \
      --report /path/to/body-reseat-report.json --write

The selected (last) report cycle must be the touch-off currently recorded in
``config/workspace.yaml``.  The body origin/+z axis must be independently
measured in ``right/tool_mount`` for at least five remove/reseat cycles; each
cycle's physical endpoint is cross-checked against its planted tip.  A passing
write copies a canonical report under ``internal/calibration/tool-body/`` and
binds its SHA-256 plus computed metrics into the workspace.  Runtime geometry
revalidates the report and digest before treating the pose as qualified.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import tool_spec  # noqa: E402


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def canonical_report_bytes(report: dict) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()


def report_relpath(qualification: tool_spec.BodyPoseQualification,
                   tool_id: str) -> Path:
    stamp = datetime.fromisoformat(
        qualification.selected_utc[:-1] + "+00:00").strftime("%Y%m%dT%H%M%SZ")
    safe_tool = re.sub(r"[^a-z0-9._-]+", "-", tool_id.lower()).strip("-")
    return Path(tool_spec.BODY_POSE_REPORT_DIR) / f"{stamp}-{safe_tool}.json"


def qualification_workspace(right: dict, qualification: tool_spec.BodyPoseQualification,
                            report_path: Path, digest: str, arm: str) -> dict:
    """Return the complete right-side workspace record for a qualified seat."""
    updated = dict(right)
    updated.update({
        "tool_body_status": "qualified",
        "tool_body_utc": qualification.selected_utc,
        "tool_body_method": qualification.method,
        "tool_body_measurement_source": qualification.measurement_source,
        "tool_body_report": report_path.as_posix(),
        "tool_body_report_sha256": digest,
        "tool_body_samples": qualification.sample_count,
        "tool_body_selected_cycle": qualification.selected_cycle,
        "tool_body_alignment_max_mm": qualification.endpoint_alignment_max_m * 1000,
        "tool_body_tip_repeatability_mm": qualification.tip_repeatability_max_m * 1000,
        "tool_body_origin_repeatability_mm": qualification.origin_repeatability_max_m * 1000,
        "tool_body_axis_repeatability_deg": qualification.axis_repeatability_max_deg,
        "tool_body_frame": tool_spec.tip_frame(arm),
        "tool_body_origin_x": qualification.body_origin_m[0],
        "tool_body_origin_y": qualification.body_origin_m[1],
        "tool_body_origin_z": qualification.body_origin_m[2],
        "tool_body_rpy_x": qualification.body_rpy_rad[0],
        "tool_body_rpy_y": qualification.body_rpy_rad[1],
        "tool_body_rpy_z": qualification.body_rpy_rad[2],
    })
    return updated


def qualify(report_path: Path, tool_id: str, *, arm: str = "right",
            repo: Path = REPO, workspace_path: Path | None = None,
            write: bool = False) -> tuple[tool_spec.BodyPoseQualification, Path]:
    repo = repo.resolve()
    workspace_path = workspace_path or repo / tool_spec.WORKSPACE_RELPATH
    try:
        report = json.loads(report_path.expanduser().read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"body report does not exist: {report_path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"body report is not valid UTF-8 JSON: {report_path}") from exc
    workspace = (tool_spec.parse_simple_yaml(workspace_path.read_text())
                 if workspace_path.is_file() else {})
    spec = tool_spec.require_stated_tool(
        tool_id, repo, arm, workspace, context="tool body qualification")
    current_tip = tool_spec.tip_offset_m(workspace, arm)
    if current_tip is None:
        raise ValueError("workspace has no current planted-tip calibration")
    current_session = ((workspace.get(arm) or {}).get("touchoff") or {}).get("session")
    if not current_session:
        raise ValueError("workspace touch-off has no session id")
    qualification = tool_spec.validate_body_pose_report(
        spec, report, arm, expected_tip=current_tip,
        expected_session=current_session)
    canonical = canonical_report_bytes(report)
    digest = hashlib.sha256(canonical).hexdigest()
    relative_report = report_relpath(qualification, tool_id)
    target_report = repo / relative_report

    print(f"tool body: {tool_id} in {tool_spec.tip_frame(arm)}")
    print(f"  method/source       {qualification.method} / "
          f"{qualification.measurement_source}")
    print(f"  reseats/current     {qualification.sample_count} / cycle "
          f"{qualification.selected_cycle}")
    print(f"  endpoint alignment {qualification.endpoint_alignment_max_m * 1000:.3f} mm max "
          f"(limit {tool_spec.CONTACT_ALIGNMENT_TOLERANCE_M * 1000:.3f})")
    print(f"  tip/origin repeat  {qualification.tip_repeatability_max_m * 1000:.3f} / "
          f"{qualification.origin_repeatability_max_m * 1000:.3f} mm max")
    print(f"  axis repeatability {qualification.axis_repeatability_max_deg:.3f} deg max")
    print(f"  selected origin    {' '.join(f'{value:.6f}' for value in qualification.body_origin_m)} m")
    print(f"  selected axis      {' '.join(f'{value:.6f}' for value in qualification.body_axis_unit)}")
    print(f"  report             {relative_report} sha256 {digest}")

    if not write:
        print("dry run — pass --write to copy the report and qualify config/workspace.yaml")
        return qualification, target_report

    if target_report.exists() and target_report.read_bytes() != canonical:
        raise ValueError(
            f"refusing to replace different evidence at {relative_report}")

    right = qualification_workspace(
        workspace.get(arm) or {}, qualification, relative_report, digest, arm)
    # Import the canonical workspace renderer only for a write; dry-run and
    # runtime validation stay stdlib-only.
    sys.path.insert(0, str(repo / "scripts"))
    import il_touchoff  # noqa: PLC0415

    rendered_workspace = il_touchoff.render_workspace(right).encode()
    _atomic_write(target_report, canonical)
    _atomic_write(workspace_path, rendered_workspace)
    written_workspace = tool_spec.parse_simple_yaml(rendered_workspace.decode())
    geometry = tool_spec.resolved_tool_geometry(spec, written_workspace, arm, repo)
    if geometry.status != "qualified":
        raise RuntimeError(
            "internal error: written body evidence did not revalidate: "
            f"{geometry.qualification_error}")
    print(f"wrote {relative_report}")
    print(f"qualified {workspace_path}")
    print("next: regenerate the URDF, then run tool sync and the production sim audit")
    return qualification, target_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="independent body-axis/reseat JSON report")
    parser.add_argument("--ee-tool", "--tool-id", dest="tool_id", required=True,
                        help="physical tool currently fitted in the mount")
    parser.add_argument("--arm", default="right", choices=("right", "left"))
    parser.add_argument("--write", action="store_true",
                        help="copy evidence into the repo and update workspace.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        qualify(args.report, args.tool_id, arm=args.arm, write=args.write)
    except (ValueError, tool_spec.ToolMismatchError, tool_spec.ToolMountError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
