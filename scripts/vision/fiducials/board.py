"""Physical-instance routing for the ID-reusing calibration board."""

from __future__ import annotations

import cv2
import numpy as np


def rigid_board_instance_ids(
    tags,
    layout,
    intrinsics,
    distortion,
    *,
    board_ids,
    root_id,
    max_error_m,
):
    """IDs whose corners match the board plane defined by an exclusive root.

    Phase routing establishes that a shot is a board shot.  This second check
    rejects a leaked wrist/palette copy of a reused numeric ID.
    """
    if root_id not in tags or root_id not in layout:
        return []

    def rectify(corners):
        corners = np.asarray(corners, np.float64)
        if distortion is None:
            return corners
        return cv2.undistortPoints(
            corners.reshape(-1, 1, 2), intrinsics, distortion, P=intrinsics
        ).reshape(-1, 2)

    homography, _ = cv2.findHomography(
        rectify(tags[root_id]), np.asarray(layout[root_id])[:, :2]
    )
    if homography is None:
        return []
    accepted = []
    for tag_id in board_ids:
        if tag_id not in tags or tag_id not in layout:
            continue
        plane = cv2.perspectiveTransform(
            rectify(tags[tag_id]).reshape(1, -1, 2), homography
        ).reshape(-1, 2)
        expected = np.asarray(layout[tag_id])[:, :2]
        error = float(np.sqrt(np.mean(np.sum((plane - expected) ** 2, axis=1))))
        if error <= max_error_m:
            accepted.append(tag_id)
    return accepted

