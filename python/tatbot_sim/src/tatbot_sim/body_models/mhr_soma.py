"""Deterministic MHR identity transfer and posing on the canonical SOMA mesh."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tatbot_sim.body_models.io import (
    BodyModelError,
    load_body_model_spec,
    verify_body_cache,
    verify_software_lock,
)
from tatbot_sim.human_rep.contracts import load_contract

_SURFACE_QUANTIZATION_M = 0.00001
_SOMA_TO_TATBOT = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    dimensions = ",".join(str(item) for item in array.shape)
    header = f"dtype={array.dtype.str};shape={dimensions};order=C\n".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def canonical_topology_digest(faces: np.ndarray) -> str:
    """Hash the invariant indexed mid topology with its dtype and shape."""

    return _array_digest(np.asarray(faces, dtype="<i4"))


def canonical_surface_digest(vertices_m: np.ndarray) -> str:
    """Hash one Tatbot-frame surface after the reviewed 10 micrometre quantization."""

    vertices = np.asarray(vertices_m)
    if vertices.shape != (18_056, 3) or not np.isfinite(vertices).all():
        raise BodyModelError(
            "body_units_or_axes_invalid",
            "vertices",
            f"expected finite (18056,3), got {vertices.shape}",
        )
    quantized = np.rint(vertices / _SURFACE_QUANTIZATION_M).astype("<i8")
    header = (
        b"dtype=<i8;shape=18056,3;order=C;quantization_m=0.00001;axes=x,-z,y\n"
    )
    return hashlib.sha256(header + quantized.tobytes(order="C")).hexdigest()


def _quaternion_matrices_xyzw(values: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(values, dtype=np.float64)
    if quaternions.shape != (77, 4) or not np.isfinite(quaternions).all():
        raise BodyModelError(
            "pose_unsupported",
            "joint_rotations_xyzw",
            f"expected finite (77,4), got {quaternions.shape}",
        )
    norms = np.linalg.norm(quaternions, axis=1)
    if np.max(np.abs(norms - 1.0)) > 1e-5:
        raise BodyModelError("pose_unsupported", "joint_rotations_xyzw", "non-unit quaternion")
    quaternions = quaternions / norms[:, None]
    x, y, z, w = quaternions.T
    matrices = np.empty((77, 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def _tatbot_points(value: torch.Tensor) -> np.ndarray:
    points = value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(points @ _SOMA_TO_TATBOT.T, dtype=np.float32)


def _tatbot_transforms(value: torch.Tensor) -> np.ndarray:
    transforms = value.detach().cpu().numpy().astype(np.float32, copy=False)
    change = np.eye(4, dtype=np.float32)
    change[:3, :3] = _SOMA_TO_TATBOT
    return np.ascontiguousarray(change[None, :, :] @ transforms, dtype=np.float32)


@dataclass(frozen=True)
class SOMASurface:
    """One indexed canonical or posed surface in Tatbot coordinates."""

    vertices_m: np.ndarray
    faces: np.ndarray
    joints_m: np.ndarray
    transforms: np.ndarray
    joint_names: tuple[str, ...]
    surface_sha256: str

    @property
    def face_vertices_m(self) -> np.ndarray:
        """Return the representation-neutral expanded `(F,3,3)` surface."""

        return self.vertices_m[self.faces]


class SOMAPosedBody:
    """The fixed MHR-through-SOMA provider used by every nominal-body consumer."""

    def __init__(
        self,
        *,
        spec_path: str | Path,
        cache_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.spec_path = Path(spec_path)
        self.spec = load_body_model_spec(self.spec_path)
        self.cache_dir = verify_body_cache(self.spec, cache_dir)
        verify_software_lock(self.spec)
        if str(device) != "cpu" and not str(device).startswith("cuda"):
            raise BodyModelError("body_model_unsupported", "device", str(device))
        self.device = torch.device(device)

        # Import only after the complete byte and environment audit. A missing
        # cache therefore cannot reach SOMA's automatic Hub-download branch.
        try:
            from soma import SOMALayer
            from soma.units import Unit
        except ImportError as exc:
            raise BodyModelError("body_model_unpinned", "py-soma-x", str(exc)) from exc

        model = self.spec["model"]
        self._layer = SOMALayer(
            data_root=self.cache_dir,
            device=self.device,
            identity_model_type=model["identity_model_type"],
            mode=model["mode"],
            output_unit=Unit.METERS,
            lod=model["lod"],
            enable_procedural_transforms=model["enable_procedural_transforms"],
            correctives_model_path=self.cache_dir / "correctives_model.pt",
        )
        self.faces = np.ascontiguousarray(self._layer.faces.detach().cpu().numpy(), dtype="<i4")
        topology = canonical_topology_digest(self.faces)
        if topology != self.spec["geometry"]["topology_sha256"]:
            raise BodyModelError(
                "body_topology_mismatch",
                "SOMALayer.faces",
                f"expected {self.spec['geometry']['topology_sha256']}, got {topology}",
            )
        self.joint_names = tuple(str(name) for name in self._layer.public_joint_names)
        if len(self.joint_names) != model["public_joints"]:
            raise BodyModelError(
                "body_topology_mismatch",
                "SOMALayer.public_joint_names",
                f"expected {model['public_joints']}, got {len(self.joint_names)}",
            )

    def _identity_tensors(self, identity: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if identity["model_spec_sha256"] != self.spec["content_sha256"]:
            raise BodyModelError(
                "body_model_unpinned",
                "BodyIdentity.model_spec_sha256",
                identity["model_spec_sha256"],
            )
        coefficients = torch.tensor(identity["coefficients"], dtype=torch.float32, device=self.device)
        scales = torch.tensor(identity["scales"], dtype=torch.float32, device=self.device)
        return coefficients.reshape(1, 45), scales.reshape(1, 68)

    def generate(
        self,
        identity: dict[str, Any],
        joint_rotations_xyzw: np.ndarray | list[list[float]],
        *,
        apply_correctives: bool = True,
    ) -> SOMASurface:
        """Generate one deterministic posed surface from validated contract values."""

        coefficients_raw = np.asarray(identity.get("coefficients"), dtype=np.float64)
        scales_raw = np.asarray(identity.get("scales"), dtype=np.float64)
        prior = identity.get("bounds_prior")
        if coefficients_raw.shape != (45,) or scales_raw.shape != (68,):
            raise BodyModelError(
                "identity_out_of_prior",
                "BodyIdentity",
                f"coefficients={coefficients_raw.shape}; scales={scales_raw.shape}",
            )
        if not np.isfinite(coefficients_raw).all() or not np.isfinite(scales_raw).all():
            raise BodyModelError("identity_out_of_prior", "BodyIdentity", "non-finite identity parameter")
        if (
            not isinstance(prior, dict)
            or not isinstance(prior.get("max_abs_coefficient"), (int, float))
            or not isinstance(prior.get("max_abs_scale"), (int, float))
        ):
            raise BodyModelError("identity_out_of_prior", "BodyIdentity.bounds_prior", "missing bound")
        if float(np.abs(coefficients_raw).max(initial=0.0)) > float(prior["max_abs_coefficient"]):
            raise BodyModelError("identity_out_of_prior", "BodyIdentity.coefficients", "coefficient exceeds prior")
        if float(np.abs(scales_raw).max(initial=0.0)) > float(prior["max_abs_scale"]):
            raise BodyModelError("identity_out_of_prior", "BodyIdentity.scales", "scale exceeds prior")
        coefficients, scales = self._identity_tensors(identity)
        rotations = torch.tensor(
            _quaternion_matrices_xyzw(np.asarray(joint_rotations_xyzw)),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 77, 3, 3)
        torch.use_deterministic_algorithms(True)
        with torch.inference_mode():
            output = self._layer(
                rotations,
                coefficients,
                scale_params=scales,
                transl=torch.zeros((1, 3), dtype=torch.float32, device=self.device),
                pose2rot=False,
                apply_correctives=apply_correctives,
            )
        vertices = _tatbot_points(output.vertices[0])
        joints = _tatbot_points(output.joints[0])
        transforms = _tatbot_transforms(output.transforms[0])
        self._validate_geometry(vertices, joints)
        return SOMASurface(
            vertices_m=vertices,
            faces=self.faces,
            joints_m=joints,
            transforms=transforms,
            joint_names=self.joint_names,
            surface_sha256=canonical_surface_digest(vertices),
        )

    def rest(self, identity: dict[str, Any]) -> SOMASurface:
        """Generate and verify the canonical T-pose surface for an identity."""

        rotations = np.zeros((77, 4), dtype=np.float64)
        rotations[:, 3] = 1.0
        result = self.generate(identity, rotations)
        self._validate_reference_landmarks(result.vertices_m, result.joints_m)
        expected = identity["rest_surface_sha256"]
        if result.surface_sha256 != expected:
            raise BodyModelError(
                "body_rest_surface_mismatch",
                "BodyIdentity.rest_surface_sha256",
                f"expected {expected}, got {result.surface_sha256}",
            )
        return result

    def from_contracts(
        self,
        identity_path: str | Path,
        state_path: str | Path,
    ) -> SOMASurface:
        """Load a BodyIdentity/1 and BodyState/1 and materialize the exact surface."""

        identity = load_contract(identity_path, expected_schema="tatbot.body-identity/1")
        state = load_contract(state_path, expected_schema="tatbot.body-state/1")
        if state["model_spec_sha256"] != self.spec["content_sha256"]:
            raise BodyModelError("body_model_unpinned", "BodyState.model_spec_sha256", "mismatch")
        if state["body_identity_sha256"] != identity["content_sha256"]:
            raise BodyModelError(
                "body_rest_surface_mismatch",
                "BodyState.body_identity_sha256",
                "does not bind the supplied identity",
            )
        if state["correctives_enabled"] is not True:
            raise BodyModelError("pose_unsupported", "BodyState.correctives_enabled", "must be true")
        result = self.generate(identity, state["joint_rotations_xyzw"])
        if result.surface_sha256 != state["posed_surface_sha256"]:
            raise BodyModelError(
                "body_rest_surface_mismatch",
                "BodyState.posed_surface_sha256",
                f"expected {state['posed_surface_sha256']}, got {result.surface_sha256}",
            )
        return result

    def _validate_geometry(self, vertices: np.ndarray, joints: np.ndarray) -> None:
        if vertices.shape != (18_056, 3) or joints.shape != (77, 3):
            raise BodyModelError(
                "body_topology_mismatch",
                "SOMALayer.output",
                f"vertices={vertices.shape}; joints={joints.shape}",
            )
        if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
            raise BodyModelError("body_units_or_axes_invalid", "SOMALayer.output", "non-finite value")
        face_vertices = vertices[self.faces]
        doubled_area = np.linalg.norm(
            np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0]),
            axis=1,
        )
        if float(doubled_area.min()) <= 1e-12:
            raise BodyModelError("body_topology_mismatch", "SOMALayer.faces", "degenerate triangle")
        span = np.ptp(vertices, axis=0)
        if not (1.4 < float(span.max()) < 2.2):
            raise BodyModelError(
                "body_units_or_axes_invalid",
                "SOMALayer.vertices",
                f"largest span={float(span.max()):.6f} m",
            )

    def _validate_reference_landmarks(self, vertices: np.ndarray, joints: np.ndarray) -> None:
        """Gate the one canonical rest surface with asymmetric frame sentinels.

        Head-above-feet and stature are properties of the canonical standing
        frame, not arbitrary supported poses.  Applying those tests to a
        supine or seated body would turn valid articulation into an axes
        refusal.
        """

        joint_index = {name: index - 1 for index, name in enumerate(self.joint_names) if index > 0}
        left = joints[joint_index["LeftHand"]]
        right = joints[joint_index["RightHand"]]
        head = joints[joint_index["HeadEnd"]]
        feet = joints[[joint_index["LeftFoot"], joint_index["RightFoot"]]]
        chest = joints[joint_index["Chest"]]
        if not (
            left[0] > right[0]
            and head[2] > float(feet[:, 2].max())
            and chest[1] < 0.05
        ):
            raise BodyModelError(
                "body_units_or_axes_invalid",
                "SOMALayer.joints",
                "left/right, front/back, or head/feet sentinel failed",
            )
        height = float(vertices[:, 2].max() - vertices[:, 2].min())
        if not (1.4 < height < 2.2) or not math.isclose(float(np.linalg.det(_SOMA_TO_TATBOT)), 1.0):
            raise BodyModelError(
                "body_units_or_axes_invalid",
                "SOMALayer.vertices",
                f"height={height:.6f} m or handedness invalid",
            )
