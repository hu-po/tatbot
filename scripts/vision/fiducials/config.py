"""Typed, fail-closed loader for the shared fiducial inventory."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY_PATH = Path(
    os.environ.get("TATBOT_FIDUCIAL_CONFIG", REPO / "config" / "fiducials.json")
)
SUPPORTED_FAMILIES = frozenset({"apriltag_16h5"})


@dataclasses.dataclass(frozen=True)
class DetectorProfile:
    scale: float = 1.0
    adaptive_window_max: int = 45
    min_side_px: float = 12.0
    corner_refinement: bool = True


@dataclasses.dataclass(frozen=True)
class TargetSpec:
    name: str
    role: str
    ids: tuple[int, ...]
    edge_m: float
    layout: str | None = None
    parent_frame: str | None = None
    minimum_acquisition_ids: int | None = None
    ambiguity_group: str | None = None
    root_id: int | None = None
    calibration_root_id: int | None = None
    grid: tuple[tuple[int, ...], ...] | None = None
    minimum_calibration_observations: int | None = None
    minimum_calibration_poses_per_id: int | None = None
    max_calibration_corner_px: float | None = None
    max_calibration_residual_mm: float | None = None
    max_calibration_parent_distance_mm: float | None = None
    max_calibration_reprojection_px: float | None = None
    max_calibration_consensus_mm: float | None = None
    max_calibration_regression_mm: float | None = None


@dataclasses.dataclass(frozen=True)
class FiducialInventory:
    schema_version: int
    family: str
    targets: dict[str, TargetSpec]
    detector_profiles: dict[str, DetectorProfile]
    spare_ids: tuple[int, ...]
    inventory_hash: str
    source: Path

    def target(self, name: str) -> TargetSpec:
        try:
            return self.targets[name]
        except KeyError as error:
            raise ValueError(f"{self.source}: no fiducial target named {name!r}") from error

    @property
    def known_ids(self) -> frozenset[int]:
        return frozenset(tag_id for target in self.targets.values() for tag_id in target.ids)

    @property
    def printed_ids(self) -> frozenset[int]:
        return self.known_ids | frozenset(self.spare_ids)

    def size_hypotheses(self, tag_id: int) -> tuple[float, ...]:
        return tuple(
            sorted({target.edge_m for target in self.targets.values() if tag_id in target.ids})
        )

    def owners(self, tag_id: int) -> tuple[str, ...]:
        """Physical target instances that carry one decoded numeric id."""
        return tuple(name for name, target in self.targets.items() if tag_id in target.ids)

    def exclusive_ids(self, target_name: str) -> tuple[int, ...]:
        """Ids with exactly one mounted physical instance, on ``target_name``."""
        target = self.target(target_name)
        return tuple(tag_id for tag_id in target.ids if self.owners(tag_id) == (target_name,))


def _profile(name: str, raw: dict) -> DetectorProfile:
    profile = DetectorProfile(
        scale=float(raw.get("scale", 1.0)),
        adaptive_window_max=int(raw.get("adaptive_window_max", 45)),
        min_side_px=float(raw.get("min_side_px", 12.0)),
        corner_refinement=bool(raw.get("corner_refinement", True)),
    )
    if not math.isfinite(profile.scale) or not 0 < profile.scale <= 1:
        raise ValueError(f"detector.{name}.scale must be in (0, 1]")
    if (
        profile.adaptive_window_max < 3
        or not math.isfinite(profile.min_side_px)
        or profile.min_side_px <= 0
    ):
        raise ValueError(f"detector.{name} window and minimum side must be positive")
    return profile


def load_inventory(path: str | Path = DEFAULT_INVENTORY_PATH) -> FiducialInventory:
    path = Path(path).expanduser().resolve()
    raw_bytes = path.read_bytes()
    data = json.loads(raw_bytes)
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported fiducial schema {data.get('schema_version')!r}")
    family = str(data.get("family", ""))
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"{path}: unsupported fiducial family {family!r}")

    targets = {}
    for name, entry in data.get("targets", {}).items():
        ids = tuple(int(tag_id) for tag_id in entry.get("ids", ()))
        if not ids or len(set(ids)) != len(ids) or any(tag_id < 0 for tag_id in ids):
            raise ValueError(f"{path}: targets.{name}.ids must be unique non-negative ids")
        edge_m = float(entry.get("edge_m", 0))
        if not math.isfinite(edge_m) or edge_m <= 0:
            raise ValueError(f"{path}: targets.{name}.edge_m must be positive")
        parent_frame = entry.get("parent_frame")
        if parent_frame is not None and (
            not isinstance(parent_frame, str) or not parent_frame.strip()
        ):
            raise ValueError(f"{path}: targets.{name}.parent_frame must be a link name")
        minimum = entry.get("minimum_acquisition_ids")
        if minimum is not None and not 1 <= int(minimum) <= len(ids):
            raise ValueError(f"{path}: targets.{name}.minimum_acquisition_ids is invalid")
        root_id = entry.get("root_id")
        if root_id is not None and int(root_id) not in ids:
            raise ValueError(f"{path}: targets.{name}.root_id must be one of its ids")
        calibration_root_id = entry.get("calibration_root_id")
        if calibration_root_id is not None and int(calibration_root_id) not in ids:
            raise ValueError(
                f"{path}: targets.{name}.calibration_root_id must be one of its ids"
            )
        grid = entry.get("grid")
        parsed_grid = None
        if grid is not None:
            if (
                not isinstance(grid, list)
                or not grid
                or any(not isinstance(row, list) or not row for row in grid)
                or len({len(row) for row in grid}) != 1
            ):
                raise ValueError(f"{path}: targets.{name}.grid must be a non-empty rectangle")
            parsed_grid = tuple(tuple(int(tag_id) for tag_id in row) for row in grid)
            flattened = tuple(tag_id for row in parsed_grid for tag_id in row)
            if len(set(flattened)) != len(flattened) or set(flattened) != set(ids):
                raise ValueError(
                    f"{path}: targets.{name}.grid must contain each target id exactly once"
                )
        minimum_observations = entry.get("minimum_calibration_observations")
        if minimum_observations is not None and int(minimum_observations) < 4:
            raise ValueError(f"{path}: targets.{name}.minimum_calibration_observations must be >=4")
        minimum_poses_per_id = entry.get("minimum_calibration_poses_per_id")
        if minimum_poses_per_id is not None and int(minimum_poses_per_id) < 2:
            raise ValueError(
                f"{path}: targets.{name}.minimum_calibration_poses_per_id must be >=2"
            )
        max_corner_px = entry.get("max_calibration_corner_px")
        max_residual_mm = entry.get("max_calibration_residual_mm")
        max_parent_distance_mm = entry.get("max_calibration_parent_distance_mm")
        max_reprojection_px = entry.get("max_calibration_reprojection_px")
        max_consensus_mm = entry.get("max_calibration_consensus_mm")
        max_regression_mm = entry.get("max_calibration_regression_mm")
        if max_corner_px is not None and (
            not math.isfinite(float(max_corner_px)) or float(max_corner_px) <= 0
        ):
            raise ValueError(f"{path}: targets.{name}.max_calibration_corner_px must be positive")
        if max_residual_mm is not None and (
            not math.isfinite(float(max_residual_mm)) or float(max_residual_mm) <= 0
        ):
            raise ValueError(f"{path}: targets.{name}.max_calibration_residual_mm must be positive")
        for field_name, value in (
            ("max_calibration_reprojection_px", max_reprojection_px),
            ("max_calibration_consensus_mm", max_consensus_mm),
            ("max_calibration_regression_mm", max_regression_mm),
            ("max_calibration_parent_distance_mm", max_parent_distance_mm),
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{path}: targets.{name}.{field_name} must be positive")
        targets[name] = TargetSpec(
            name=name,
            role=str(entry.get("role", "")),
            ids=ids,
            edge_m=edge_m,
            layout=entry.get("layout"),
            parent_frame=parent_frame,
            minimum_acquisition_ids=int(minimum) if minimum is not None else None,
            ambiguity_group=entry.get("ambiguity_group"),
            root_id=int(root_id) if root_id is not None else None,
            calibration_root_id=(
                int(calibration_root_id) if calibration_root_id is not None else None
            ),
            grid=parsed_grid,
            minimum_calibration_observations=(
                int(minimum_observations) if minimum_observations is not None else None
            ),
            minimum_calibration_poses_per_id=(
                int(minimum_poses_per_id) if minimum_poses_per_id is not None else None
            ),
            max_calibration_corner_px=float(max_corner_px) if max_corner_px is not None else None,
            max_calibration_residual_mm=(
                float(max_residual_mm) if max_residual_mm is not None else None
            ),
            max_calibration_parent_distance_mm=(
                float(max_parent_distance_mm) if max_parent_distance_mm is not None else None
            ),
            max_calibration_reprojection_px=(
                float(max_reprojection_px) if max_reprojection_px is not None else None
            ),
            max_calibration_consensus_mm=(
                float(max_consensus_mm) if max_consensus_mm is not None else None
            ),
            max_calibration_regression_mm=(
                float(max_regression_mm) if max_regression_mm is not None else None
            ),
        )
    if not targets:
        raise ValueError(f"{path}: at least one fiducial target is required")
    for name, target in targets.items():
        if target.role == "rigid_ee" and not target.parent_frame:
            raise ValueError(f"{path}: targets.{name}.parent_frame is required for rigid_ee")

    owners: dict[int, list[TargetSpec]] = {}
    for target in targets.values():
        for tag_id in target.ids:
            owners.setdefault(tag_id, []).append(target)
    for tag_id, matching in owners.items():
        if len(matching) < 2:
            continue
        groups = {target.ambiguity_group for target in matching}
        if None in groups or len(groups) != 1:
            names = ", ".join(target.name for target in matching)
            raise ValueError(
                f"{path}: id {tag_id} is shared by {names} without one explicit ambiguity_group"
            )
    for name, target in targets.items():
        if target.calibration_root_id is not None and len(owners[target.calibration_root_id]) != 1:
            raise ValueError(
                f"{path}: targets.{name}.calibration_root_id must identify one physical instance"
            )

    detector_profiles = {
        name: _profile(name, entry) for name, entry in data.get("detector", {}).items()
    }
    missing_profiles = {"calibration", "live"} - detector_profiles.keys()
    if missing_profiles:
        raise ValueError(f"{path}: missing detector profiles {sorted(missing_profiles)}")
    spare_ids = tuple(int(tag_id) for tag_id in data.get("printing", {}).get("spare_ids", ()))
    if (
        any(tag_id < 0 for tag_id in spare_ids)
        or len(set(spare_ids)) != len(spare_ids)
        or set(spare_ids) & set(owners)
    ):
        raise ValueError(
            f"{path}: spare ids must be non-negative, unique, and absent from mounted targets"
        )
    return FiducialInventory(
        schema_version=1,
        family=family,
        targets=targets,
        detector_profiles=detector_profiles,
        spare_ids=spare_ids,
        inventory_hash=hashlib.sha256(raw_bytes).hexdigest(),
        source=path,
    )
