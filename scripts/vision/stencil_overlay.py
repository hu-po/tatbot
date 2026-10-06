"""Measured image correspondences and the nominal clear-center boundary."""

import cv2
import numpy as np

PALETTE = [(70, 230, 130), (70, 180, 255), (230, 110, 230), (255, 220, 90)]


def annotated(image, rows, references, *, camera='', width=1280):
    scale = min(1., width/image.shape[1])
    size = (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale)))
    shown = cv2.resize(image, size)
    shown = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR) if shown.ndim == 2 else shown.copy()
    labels = [camera] if camera else []
    for index, row in enumerate(rows):
        reference = references.get(row.get('pattern_id'), {})
        label = f"{row.get('seed', reference.get('seed', '?'))}: {row['status']}"
        labels.append(label)
        if row.get('image_tracking_valid'):
            _boundaries(shown, row, reference, PALETTE[index % len(PALETTE)], scale)
    for index, label in enumerate(labels):
        origin = (10, 26+26*index)
        cv2.putText(shown, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(shown, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
    return shown


def _boundaries(image, row, reference, color, scale):
    corners = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], np.float64)
    polygon = cv2.perspectiveTransform(corners, np.asarray(row['homography_uv_to_image'], float))[0]*scale
    cv2.polylines(image, [np.rint(polygon).astype(np.int32)], True, color, 2, cv2.LINE_AA)
    if 'clear_center_uv' in reference:
        u0, v0, u1, v1 = reference['clear_center_uv']
        uv = np.array([[[u0, v0], [u1, v0], [u1, v1], [u0, v1]]], np.float64)
        center = cv2.perspectiveTransform(uv, np.asarray(row['homography_uv_to_image'], float))[0]*scale
        cv2.polylines(image, [np.rint(center).astype(np.int32)], True, color, 1, cv2.LINE_AA)
    for landmark in row.get('landmarks', []):
        point = tuple(np.rint(np.asarray(landmark['image_px'])*scale).astype(int))
        cv2.circle(image, point, 2, color, -1)
