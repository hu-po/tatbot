"""Write one offline synthetic registration shadow through Tatbot's Rerun contract.

The input is a derived ``.npz`` bundle created by the synthetic suite.  This
module has no network or robot entry point: it either writes one ``.rrd`` file
or refuses.  All geometry is logged in the observed-patch frame and the exact
source/target transform is recorded under named frame entities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rerun_contract():
    import tatbot_rerun  # noqa: PLC0415

    return tatbot_rerun


def write_registration_replay(
    bundle_path: str | Path,
    output_path: str | Path,
    *,
    recording_id: str = "human-representation-p5-offline",
) -> dict[str, object]:
    """Log the exact bundle to one file-backed Tatbot recording."""

    bundle_path = Path(bundle_path)
    output_path = Path(output_path)
    with np.load(bundle_path, allow_pickle=False) as payload:
        body_vertices = np.asarray(payload["body_vertices_m"], dtype=np.float32)
        body_faces = np.asarray(payload["body_faces"], dtype=np.uint32)
        observed = np.asarray(payload["observed_points_m"], dtype=np.float32)
        landmarks = np.asarray(payload["landmarks_m"], dtype=np.float32)
        measured_vertices = np.asarray(payload["measured_vertices_m"], dtype=np.float32)
        measured_faces = np.asarray(payload["measured_faces"], dtype=np.uint32)
        measured_normals = np.asarray(payload["measured_normals"], dtype=np.float32)
        art = np.asarray(payload["placed_art_m"], dtype=np.float32)
        exact_path = np.asarray(payload["exact_path_m"], dtype=np.float32)
        transform = np.asarray(payload["observed_patch_from_body"], dtype=np.float64)
        sigma = float(payload["translation_sigma_m"])
        capture_time_ns = int(payload["capture_time_ns"])
        hashes = json.loads(str(payload["hashes_json"]))
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("registration replay transform is not a homogeneous matrix4")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tr = _rerun_contract()
    rr = tr.start(
        "human_rep_registration",
        output=output_path,
        recording_id=recording_id,
        view_coordinates=True,
    )
    rr.log(
        "world/registration/nominal_body",
        rr.Mesh3D(
            vertex_positions=body_vertices,
            triangle_indices=body_faces,
            vertex_colors=[[184, 188, 196]],
        ),
        static=True,
    )
    rr.log(
        "world/registration/observed_points",
        rr.Points3D(observed, colors=[[40, 145, 230]], radii=0.0007),
        static=True,
    )
    rr.log(
        "world/registration/landmarks",
        rr.Points3D(landmarks, colors=[[240, 80, 45]], radii=max(0.001, 3.0 * sigma)),
        static=True,
    )
    rr.log(
        "world/registration/measured_surface",
        rr.Mesh3D(
            vertex_positions=measured_vertices,
            triangle_indices=measured_faces,
            vertex_normals=measured_normals,
            vertex_colors=[[225, 220, 205]],
        ),
        static=True,
    )
    if len(art):
        rr.log(
            "world/registration/placed_art",
            rr.LineStrips3D([art], colors=[[35, 35, 35]], radii=0.00035),
            static=True,
        )
    if len(exact_path):
        rr.log(
            "world/registration/exact_path",
            rr.LineStrips3D([exact_path], colors=[[90, 200, 90]], radii=0.00025),
            static=True,
        )
    frame_note = {
        "source_frame": "body",
        "target_frame": "observed_patch",
        "observed_patch_from_body": transform.tolist(),
        "translation_sigma_m": sigma,
        "hashes": hashes,
        "motion_authorized": False,
    }
    rr.log(
        "world/frames/body_to_observed_patch",
        rr.TextDocument(json.dumps(frame_note, indent=2, sort_keys=True)),
        static=True,
    )
    tr.set_capture_time(capture_time_ns)
    rr.log(
        "session/status",
        rr.TextLog("offline synthetic registration replay; geometry only; motion_authorized=false"),
    )
    # File sinks are asynchronous. Finalize the recording before callers hash
    # it into the phase manifest.
    rr.disconnect()
    return {
        "bundle": str(bundle_path),
        "output": str(output_path),
        "recording_id": recording_id,
        "points": int(len(observed)),
        "landmarks": int(len(landmarks)),
        "hashes": hashes,
        "motion_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording-id", default="human-representation-p5-offline")
    args = parser.parse_args(argv)
    result = write_registration_replay(args.bundle, args.output, recording_id=args.recording_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
