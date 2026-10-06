"""Metrics computed after observation; labels cannot influence matching."""

import cv2
import numpy as np


def score_scene(observations, truth):
    if len(observations) != len(truth):
        raise ValueError("truth must contain one label per observed frame")
    if not observations:
        return {}
    patterns = {row["pattern_id"] for row in observations[0]["stencils"]}
    labels = [{row["pattern_id"]: row for row in frame["stencils"]} for frame in truth]
    if any(set(frame) != patterns for frame in labels):
        raise ValueError("scene truth must name every configured pattern in every frame")
    rows = [{row["pattern_id"]: row for row in frame["stencils"]} for frame in observations]
    return {pattern: score([frame[pattern] for frame in rows], [frame[pattern] for frame in labels])
            for pattern in patterns}


def _image_error(observation, truth):
    if not observation["image_tracking_valid"] or "homography_uv_to_image" not in truth:
        return None
    points = np.array([[[.2, .2], [.8, .2], [.8, .8], [.2, .8], [.5, .5]]], float)
    actual = cv2.perspectiveTransform(points, np.array(observation["homography_uv_to_image"]))
    expected = cv2.perspectiveTransform(points, np.array(truth["homography_uv_to_image"], float))
    return float(np.sqrt(np.mean(np.sum((actual-expected)**2, axis=-1))))


def score(observations, truth):
    if len(observations) != len(truth):
        raise ValueError("truth must contain one label per observed frame")
    false_accepts, misses, errors, recovery = [], [], [], []
    recovering = None
    for index, (observation, label) in enumerate(zip(observations, truth, strict=True)):
        visible = label["visible"]
        valid = observation["image_tracking_valid"]
        correct = valid and visible and observation["pattern_id"] == label["pattern_id"]
        if valid and not correct:
            false_accepts.append(index)
        if visible and not correct:
            misses.append(index)
        if correct:
            error = _image_error(observation, label)
            if error is not None:
                errors.append(error)
        if visible and index > 0 and not truth[index-1]["visible"]:
            recovering = index
        if correct and recovering is not None:
            recovery.append(index-recovering)
            recovering = None
    return {"false_accept_frames": false_accepts, "missed_visible_frames": misses,
            "image_error_rmse_px_p95": float(np.percentile(errors, 95)) if errors else None,
            "recovery_delay_frames": recovery, "unrecovered_final_return": recovering is not None,
            "metric_3d_accuracy_evaluated": False}
