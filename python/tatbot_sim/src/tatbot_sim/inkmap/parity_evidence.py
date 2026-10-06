"""Generate reproducible browser/simulator mapping and mask parity evidence."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tatbot_contracts.digest import sha256_file

from tatbot_sim.human_rep.contracts import canonical_digest
from tatbot_sim.inkmap.program_target import (
    SUPERSAMPLE,
    render_program_target,
    write_program_target,
)
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.inkmap.surface_trace import (
    SurfaceAnchor,
    SurfaceTraceError,
    unfold_body_patch,
    validate_patch_footprint,
)
from tatbot_sim.repo import git_output, repo_root

SEED = 9_042_026
POSE_POSITION_P95_M = 0.0001
POSE_POSITION_MAX_M = 0.0005
MASK_IOU_MIN = 0.98
BOUNDARY_P95_PX = 1.0
REFERENCE_PIXELS_PER_M = 12_000
FIXTURES = ("linework", "blackwork", "negative-space", "stipple", "color-layers")
SURFACE_SPECS = (
    ("linework-small-left-forearm", 8729, 0.015, 0.009375, 0.0, False),
    ("blackwork-right-forearm", 26998, 0.040, 0.040, 0.4, True),
    ("negative-space-large-left-shin", 11212, 0.060, 0.060, -0.5, False),
    ("stipple-right-shin", 29266, 0.030, 0.020, 0.75, True),
    ("color-layers-large-left-thigh", 8225, 0.080, 0.050, 1.0, False),
    ("linework-mirrored-right-thigh", 26278, 0.020, 0.0125, -1.2, True),
    ("blackwork-curved-left-shoulder", 6361, 0.030, 0.030, 0.25, False),
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, env=env, check=True, capture_output=True, text=True, timeout=300,
    )


def build_surface_cases(seed: int = SEED) -> list[dict[str, Any]]:
    """Return the fixed dense and coverage cases used by tests and evidence."""

    rng = np.random.default_rng(seed)
    points: list[list[float]] = []
    for row_index, row in enumerate(np.linspace(-0.009, 0.009, 100)):
        columns = np.linspace(-0.009, 0.009, 100)
        if row_index % 2:
            columns = columns[::-1]
        points.extend([[float(column + rng.uniform(-1e-7, 1e-7)), float(row)] for column in columns])
    cases = [{
        "id": "dense-left-forearm-10000",
        "anchor": {"face": 8729, "barycentric": [1 / 3, 1 / 3, 1 / 3]},
        "rotation_rad": 0.0,
        "radius_m": 0.018,
        "points_m": points,
        "footprint_m": [0.018, 0.018],
        "expected_status": "accepted",
    }]
    for name, face, width, height, rotation, mirrored in SURFACE_SPECS:
        sample: list[list[float]] = []
        for row_index, row in enumerate(np.linspace(-height * 0.35, height * 0.35, 11)):
            columns = np.linspace(-width * 0.35, width * 0.35, 11)
            if row_index % 2:
                columns = columns[::-1]
            if mirrored:
                columns = -columns
            sample.extend([[float(column), float(row)] for column in columns])
        cases.append({
            "id": name,
            "anchor": {"face": face, "barycentric": [1 / 3, 1 / 3, 1 / 3]},
            "rotation_rad": rotation,
            "radius_m": float(np.hypot(width / 2, height / 2) + 0.003),
            "points_m": sample,
            "footprint_m": [width, height],
            "expected_status": "accepted",
        })
    width = height = 0.5
    cases.append({
        "id": "complete-wrap-left-forearm",
        "anchor": {"face": 8729, "barycentric": [1 / 3, 1 / 3, 1 / 3]},
        "rotation_rad": 0.0,
        "radius_m": float(np.hypot(width, height) / 2 + 0.002),
        "points_m": [[0.0, 0.0]],
        "footprint_m": [width, height],
        "expected_status": "rejected",
    })
    return cases


def surface_parity_report(output: Path, seed: int = SEED) -> dict[str, Any]:
    """Run one hermetic TypeScript request and compare it with Python."""

    root = repo_root()
    rig = load_body_rig()
    cases = build_surface_cases(seed)
    request_cases = [{key: value for key, value in item.items() if key != "expected_status"} for item in cases]
    with tempfile.TemporaryDirectory(prefix="tatbot-inkmap-surface-parity-") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        _write_json(request, {"cases": request_cases, "pose_ids": list(rig.pose_ids)})
        completed = _run(
            ["node", "--experimental-strip-types", "web/inkmap/tools/surface_parity.ts", str(request), str(response)],
            cwd=root,
        )
        browser = json.loads(response.read_text())
    (output / "surface-browser.log").write_text(completed.stdout + completed.stderr)
    _write_json(output / "surface-browser-response.json", browser)

    atlas = json.loads((root / "web/inkmap/public/bodies/mhr-soma-v1.regions.json").read_text())
    eligible = np.asarray(atlas["eligible_faces"], dtype=bool)
    region_codes = np.asarray(atlas["faces"], dtype=np.int32)
    position_errors: list[float] = []
    barycentric_max = 0.0
    face_mismatches = 0
    disconnected_face_samples = 0
    excluded_faces = 0
    region_leaks = 0
    point_count = 0
    cases_report: list[dict[str, Any]] = []
    for case, result in zip(cases, browser["results"], strict=True):
        expected_status = case["expected_status"]
        row: dict[str, Any] = {
            "id": case["id"], "expected_status": expected_status, "browser_status": result["status"],
        }
        source = case["anchor"]
        anchor = SurfaceAnchor(source["face"], tuple(source["barycentric"]))
        patch = unfold_body_patch(rig, anchor, case["rotation_rad"], case["radius_m"])
        try:
            validate_patch_footprint(patch, *case["footprint_m"])
            python_status = "accepted"
            python_reason = None
        except SurfaceTraceError as error:
            python_status = "rejected"
            python_reason = str(error)
        row.update({"python_status": python_status, "python_reason": python_reason})
        if expected_status == "rejected":
            row["browser_reason"] = result.get("reason")
            row["status"] = "pass" if result["status"] == python_status == expected_status else "fail"
            cases_report.append(row)
            continue
        if result["status"] != "accepted" or python_status != "accepted":
            row["browser_reason"] = result.get("reason")
            row["status"] = "fail"
            cases_report.append(row)
            continue
        python_anchors = patch.anchors(np.asarray(case["points_m"], dtype=np.float64))
        browser_anchors = result["anchors"]
        point_count += len(python_anchors)
        case_face_mismatches = sum(
            actual["face"] != expected.face
            for actual, expected in zip(browser_anchors, python_anchors, strict=True)
        )
        face_mismatches += case_face_mismatches
        case_barycentric_max = float(np.max(np.abs(
            np.asarray([item["barycentric"] for item in browser_anchors])
            - np.asarray([item.barycentric for item in python_anchors])
        )))
        barycentric_max = max(barycentric_max, case_barycentric_max)
        faces = [item.face for item in python_anchors]
        connected = {patch.seed_face}
        frontier = [patch.seed_face]
        while frontier:
            for neighbor in patch.adjacent.get(frontier.pop(), set()):
                if neighbor not in connected:
                    connected.add(neighbor)
                    frontier.append(neighbor)
        case_disconnected = sum(face not in connected for face in faces)
        disconnected_face_samples += case_disconnected
        case_excluded = int(np.count_nonzero(~eligible[faces]))
        case_region_leaks = int(np.count_nonzero(region_codes[faces] != region_codes[source["face"]]))
        excluded_faces += case_excluded
        region_leaks += case_region_leaks
        case_position_errors = []
        for pose_id in rig.pose_ids:
            posed = rig.posed(pose_id).vertices
            expected_positions = np.stack([
                np.asarray(item.barycentric) @ posed[item.face] for item in python_anchors
            ])
            actual_positions = np.asarray(result["posed_positions_m"][pose_id])
            case_position_errors.extend(np.linalg.norm(actual_positions - expected_positions, axis=1))
        position_errors.extend(case_position_errors)
        row.update({
            "point_count": len(python_anchors),
            "pose_position_count": len(case_position_errors),
            "face_id_mismatches": case_face_mismatches,
            "barycentric_max_absolute_error": case_barycentric_max,
            "position_p95_m": float(np.quantile(case_position_errors, 0.95)),
            "position_max_m": float(np.max(case_position_errors)),
            "disconnected_face_samples": case_disconnected,
            "excluded_face_samples": case_excluded,
            "semantic_region_leaks": case_region_leaks,
            "chart": result["chart"],
            "status": "pass",
        })
        cases_report.append(row)
    errors = np.asarray(position_errors)
    expected_accepted = sum(item["expected_status"] == "accepted" for item in cases)
    expected_rejected = len(cases) - expected_accepted
    report = {
        "schema": "tatbot.inkmap-surface-parity-evidence/1",
        "seed": seed,
        "poses": list(rig.pose_ids),
        "denominators": {
            "requested_cases": len(cases), "expected_accepted_cases": expected_accepted,
            "expected_rejected_cases": expected_rejected, "mapped_points": point_count,
            "posed_position_comparisons": len(position_errors),
        },
        "observed": {
            "browser_accepted_cases": browser["accepted"], "browser_rejected_cases": browser["rejected"],
            "case_failures": sum(item["status"] != "pass" for item in cases_report),
            "face_id_mismatches": face_mismatches,
            "barycentric_max_absolute_error": barycentric_max,
            "position_p95_m": float(np.quantile(errors, 0.95)),
            "position_max_m": float(errors.max()),
            "disconnected_face_samples": disconnected_face_samples,
            "excluded_face_samples": excluded_faces,
            "semantic_region_leaks": region_leaks,
        },
        "thresholds": {
            "position_p95_m": POSE_POSITION_P95_M,
            "position_max_m": POSE_POSITION_MAX_M,
            "barycentric_max_absolute_error": 1e-9,
            "allowed_count_errors": 0,
        },
        "cases": cases_report,
    }
    observed = report["observed"]
    report["status"] = "pass" if (
        browser["accepted"] == expected_accepted
        and browser["rejected"] == expected_rejected
        and observed["case_failures"] == 0
        and observed["face_id_mismatches"] == 0
        and observed["barycentric_max_absolute_error"] <= 1e-9
        and observed["position_p95_m"] <= POSE_POSITION_P95_M
        and observed["position_max_m"] <= POSE_POSITION_MAX_M
        and observed["disconnected_face_samples"] == 0
        and observed["excluded_face_samples"] == 0
        and observed["semantic_region_leaks"] == 0
    ) else "fail"
    _write_json(output / "surface-parity.json", report)
    return report


def mask_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float | int | str]:
    """Compare thresholded masks and symmetric one-pixel boundaries."""

    left = np.asarray(expected, dtype=bool)
    right = np.asarray(actual, dtype=bool)
    if left.shape != right.shape:
        raise ValueError(f"mask shape differs: {left.shape} != {right.shape}")
    union = int(np.count_nonzero(left | right))
    intersection = int(np.count_nonzero(left & right))
    if union == 0:
        return {
            "status": "empty_match", "union_pixels": 0, "intersection_pixels": 0,
            "mask_iou": 1.0, "boundary_sample_count": 0, "boundary_distance_p95_px": 0.0,
            "boundary_distance_max_px": 0.0,
        }
    kernel = np.ones((3, 3), dtype=np.uint8)
    left_boundary = left & ~cv2.erode(left.astype(np.uint8), kernel, iterations=1).astype(bool)
    right_boundary = right & ~cv2.erode(right.astype(np.uint8), kernel, iterations=1).astype(bool)
    if not left_boundary.any() or not right_boundary.any():
        boundary = np.asarray([float("inf")])
    else:
        to_left = cv2.distanceTransform((~left_boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        to_right = cv2.distanceTransform((~right_boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        boundary = np.concatenate([to_right[left_boundary], to_left[right_boundary]])
    return {
        "status": "compared", "union_pixels": union, "intersection_pixels": intersection,
        "mask_iou": intersection / union, "boundary_sample_count": len(boundary),
        "boundary_distance_p95_px": float(np.quantile(boundary, 0.95)),
        "boundary_distance_max_px": float(boundary.max()),
    }


def _white_background(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image
    alpha = image[..., 3:4].astype(np.float32) / 255
    return np.rint(image[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)


def _write_contact_sheet(output: Path) -> None:
    rows = []
    headings = ("original", "browser target", "sim target", "binary difference")
    for name in FIXTURES:
        paths = (
            output / "browser-artwork" / f"{name}.original.png",
            output / "browser-artwork" / f"{name}.browser.png",
            output / "mask" / name / "plain" / "simulator-aligned-reference.png",
            output / "mask" / name / "plain" / "difference.png",
        )
        images = [_white_background(path) for path in paths]
        images = [
            cv2.resize(
                image,
                (640, max(1, round(image.shape[0] * 640 / image.shape[1]))),
                interpolation=cv2.INTER_AREA,
            )
            if image.shape[1] > 640
            else image
            for image in images
        ]
        height = max(image.shape[0] for image in images)
        cells = []
        cell_width = 660
        for heading, image in zip(headings, images, strict=True):
            cell = np.full((height + 34, cell_width, 3), 245, dtype=np.uint8)
            cell[34 : 34 + image.shape[0], : image.shape[1]] = image
            cv2.putText(cell, heading, (7, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            cells.append(cell)
        row = np.concatenate(cells, axis=1)
        cv2.putText(row, name, (210, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(row)
    cv2.imwrite(str(output / "mask-contact-sheet.png"), np.concatenate(rows, axis=0))


def mask_parity_report(output: Path) -> dict[str, Any]:
    """Rasterize canonical previews in Chromium and compare Python targets."""

    root = repo_root()
    browser_root = output / "browser-artwork"
    browser_root.mkdir(parents=True)
    environment = os.environ.copy()
    environment["INKMAP_E2E_EVIDENCE"] = str(browser_root)
    environment["INKMAP_ARTWORK_PIXELS_PER_MM"] = str(REFERENCE_PIXELS_PER_M // 1000)
    completed = _run(["node", "tests/e2e/artwork.mjs"], cwd=root / "web/inkmap", env=environment)
    (output / "mask-browser.log").write_text(completed.stdout + completed.stderr)
    base_placement = json.loads(
        (root / "config/human-representation/examples/surface-placement.json").read_text()
    )
    cases: list[dict[str, Any]] = []
    for name in FIXTURES:
        program = json.loads((browser_root / f"{name}.program.json").read_text())
        browser_rgba = cv2.imread(str(browser_root / f"{name}.browser.png"), cv2.IMREAD_UNCHANGED)
        if browser_rgba is None or browser_rgba.shape[2] != 4:
            raise ValueError(f"browser mask for {name} is not RGBA")
        base_alpha = browser_rgba[..., 3].astype(np.float32) / 255
        for mirrored in (False, True):
            variant = "mirrored" if mirrored else "plain"
            placement = copy.deepcopy(base_placement)
            placement["tattoo_program_sha256"] = program["content_sha256"]
            placement["physical_scale_m"] = [program["canvas_m"]["width"], program["canvas_m"]["height"]]
            placement["rotation_rad"] = 0.73
            placement["mirrored"] = mirrored
            placement["content_sha256"] = canonical_digest(placement)
            target = render_program_target(program, placement, pixels_per_m=REFERENCE_PIXELS_PER_M)
            case_root = output / "mask" / name / variant
            write_program_target(case_root, target, "#c07f57")
            browser_alpha = base_alpha[:, ::-1] if mirrored else base_alpha
            # TattooProgram uses +y up. Browser SVG row zero is therefore +y,
            # while simulator field arrays intentionally store chart -y in row
            # zero. Flip only for this aligned image-space comparison.
            aligned_target = target.coverage[::-1]
            values = mask_metrics(browser_alpha >= 0.5, aligned_target >= 0.5)
            values["soft_coverage_mae"] = float(np.mean(np.abs(browser_alpha - aligned_target)))
            values.update({
                "id": f"{name}-{variant}", "fixture": name, "mirrored": mirrored,
                "width_px": target.cols, "height_px": target.rows,
                "pixels_per_mm": target.pixels_per_m / 1000,
                "supersample": target.supersample,
                "program_sha256": program["content_sha256"], "target_sha256": target.digest(),
            })
            mask_iou = values["mask_iou"]
            boundary_p95 = values["boundary_distance_p95_px"]
            values["gate_status"] = "pass" if (
                isinstance(mask_iou, (int, float))
                and mask_iou >= MASK_IOU_MIN
                and isinstance(boundary_p95, (int, float))
                and boundary_p95 <= BOUNDARY_P95_PX
            ) else "fail"
            browser_mask = np.rint(browser_alpha * 255).astype(np.uint8)
            cv2.imwrite(str(case_root / "browser-coverage.png"), browser_mask)
            cv2.imwrite(
                str(case_root / "simulator-aligned-coverage.png"),
                np.rint(aligned_target * 255).astype(np.uint8),
            )
            reference = cv2.imread(str(case_root / "target-reference.png"), cv2.IMREAD_COLOR)
            cv2.imwrite(str(case_root / "simulator-aligned-reference.png"), reference[::-1])
            left = browser_alpha >= 0.5
            right = aligned_target >= 0.5
            difference = np.zeros((*left.shape, 3), dtype=np.uint8)
            difference[left & right] = (255, 255, 255)
            difference[left & ~right] = (0, 0, 255)
            difference[~left & right] = (255, 255, 0)
            cv2.imwrite(str(case_root / "difference.png"), difference)
            cases.append(values)
    report = {
        "schema": "tatbot.inkmap-mask-parity-evidence/1",
        "reference_raster": {
            "pixels_per_m": REFERENCE_PIXELS_PER_M, "pixels_per_mm": REFERENCE_PIXELS_PER_M / 1000,
            "supersample": SUPERSAMPLE, "binary_threshold": 0.5,
            "coordinate_alignment": {
                "browser_row_0": "TattooProgram chart positive-y",
                "simulator_row_0": "TattooProgram chart negative-y",
                "comparison": "vertical flip of simulator array; columns unchanged except declared mirror variant",
            },
            "visible_surface_policy": "chart-space target has no occlusion; all admitted target pixels are compared",
            "excluded_coverage_pixels": 0,
        },
        "denominators": {"fixtures": len(FIXTURES), "mirror_variants": 2, "mask_comparisons": len(cases)},
        "thresholds": {"mask_iou_min": MASK_IOU_MIN, "boundary_distance_p95_px_max": BOUNDARY_P95_PX},
        "observed": {
            "passing_comparisons": sum(item["gate_status"] == "pass" for item in cases),
            "failing_comparisons": sum(item["gate_status"] != "pass" for item in cases),
            "minimum_mask_iou": min(item["mask_iou"] for item in cases),
            "maximum_boundary_distance_p95_px": max(item["boundary_distance_p95_px"] for item in cases),
            "maximum_soft_coverage_mae": max(item["soft_coverage_mae"] for item in cases),
        },
        "cases": cases,
    }
    report["status"] = "pass" if report["observed"]["failing_comparisons"] == 0 else "fail"
    _write_json(output / "mask-parity.json", report)
    _write_contact_sheet(output)
    return report


def run_evidence(output: Path, seed: int = SEED) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evidence directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    surface = surface_parity_report(output, seed)
    masks = mask_parity_report(output)
    status = "automated_pass_visual_review_pending" if surface["status"] == masks["status"] == "pass" else "fail"
    review = (
        "# Inkmap renderer parity review\n\n"
        f"- Automated status: `{status}`\n"
        "- Human visual review: `pending`\n"
        "- GPU rendered-scene review: `not_run_no_compute_assignment`\n"
        "- Scope: software agreement only; no tissue, robot, deployment, or public-release acceptance.\n"
    )
    (output / "review.md").write_text(review)
    artifacts = [output / name for name in (
        "surface-parity.json", "surface-browser-response.json", "mask-parity.json",
        "mask-contact-sheet.png", "review.md",
    )]
    manifest = {
        "schema": "tatbot.inkmap-renderer-parity-packet/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vantage": f"{platform.node()}:{output}",
        "source": {
            "git_sha": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
            "branch": git_output("branch", "--show-current"),
        },
        "commands": {
            "surface": "node --experimental-strip-types web/inkmap/tools/surface_parity.ts REQUEST RESPONSE",
            "browser_masks": "node web/inkmap/tests/e2e/artwork.mjs",
            "runner": "python -m tatbot_sim.inkmap.parity_evidence --output OUTPUT",
        },
        "runtime": {
            "python": platform.python_version(), "node": _run(["node", "--version"], cwd=repo_root()).stdout.strip(),
            "opencv": cv2.__version__, "numpy": np.__version__,
        },
        "seed": seed,
        "surface_status": surface["status"], "mask_status": masks["status"],
        "human_visual_review": "pending", "gpu_rendered_scene_review": "not_run_no_compute_assignment",
        "status": status,
        "artifacts": [{"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path)} for path in artifacts],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    manifest = run_evidence(args.output, args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
