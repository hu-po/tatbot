"""Robust local surface fits and conditional spatial-bootstrap uncertainty."""

import cv2
import numpy as np


def design(uv, curved=False):
    u, v = (np.asarray(uv)-.5).T
    columns = [np.ones(len(u)), u, v]
    if curved:
        columns += [u*u, u*v, v*v]
    return np.column_stack(columns)


def fit_model(uv, xyz, curved=False, *, coeff_only=False):
    matrix = design(uv, curved)
    if len(uv) < 2*matrix.shape[1] or np.linalg.cond(matrix) > 100:
        return None
    weights = np.ones(len(xyz))
    for _ in range(5):
        weighted = matrix*np.sqrt(weights[:, None])
        coeff = np.linalg.lstsq(weighted, xyz*np.sqrt(weights[:, None]), rcond=None)[0]
        residual = np.linalg.norm(matrix@coeff-xyz, axis=1)
        weights = np.minimum(1, .003/np.maximum(residual, 1e-9))
    if coeff_only:
        # Bootstrap draws need the same robust coefficients, but never use
        # leave-one-out diagnostics. Skip that matrix inverse for each draw.
        return {"coeff": coeff}
    leverage = np.sum((weighted@np.linalg.pinv(weighted.T@weighted))*weighted, axis=1)
    loo = residual/np.maximum(1-leverage, .05)
    return {"coeff": coeff, "curved": curved, "residual": residual,
            "loo_p95_m": float(np.percentile(loo, 95)),
            "loo_rms_m": float(np.sqrt(np.mean(loo**2)))}


def choose_model(uv, xyz):
    plane = fit_model(uv, xyz)
    if plane is None:
        return None
    curved = fit_model(uv, xyz, True) if len(uv) >= 24 else None
    if curved is not None and curved["loo_rms_m"] < .7*plane["loo_rms_m"] and plane["loo_rms_m"] > .001:
        return curved
    return plane


def center_pose(coeff):
    x, y = coeff[1].copy(), coeff[2].copy()
    if np.linalg.norm(x) < .005:
        return None
    x /= np.linalg.norm(x)
    y -= x*np.dot(x, y)
    if np.linalg.norm(y) < .005:
        return None
    y /= np.linalg.norm(y)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
    pose[:3, 3] = coeff[0]
    return pose


def bootstrap_pose(uv, xyz, model, pose, *, samples=64):
    """Resample spatial groups; this cannot estimate systematic sensor error."""
    groups = np.clip((np.asarray(uv)*4).astype(int), 0, 3)
    groups = groups[:, 0]+4*groups[:, 1]
    unique = np.unique(groups)
    if len(unique) < 6:
        return {"available": False, "reason": "insufficient_spatial_groups"}
    rng = np.random.default_rng(41)
    deltas = []
    for _ in range(samples):
        indices = np.concatenate([np.flatnonzero(groups == group) for group in rng.choice(unique, len(unique))])
        candidate = fit_model(uv[indices], xyz[indices], model["curved"], coeff_only=True)
        transform = center_pose(candidate["coeff"]) if candidate is not None else None
        if transform is None:
            continue
        rotation, _ = cv2.Rodrigues(pose[:3, :3].T@transform[:3, :3])
        deltas.append(np.r_[transform[:3, 3]-pose[:3, 3], rotation.ravel()])
    if len(deltas) < 24:
        return {"available": False, "reason": "degenerate_bootstrap_samples"}
    covariance = np.cov(np.asarray(deltas).T)
    return {"available": True, "samples": len(deltas), "spatial_groups": len(unique),
            "covariance_6x6": covariance.tolist(),
            "translation_std_m": np.sqrt(np.maximum(np.diag(covariance)[:3], 0)).tolist(),
            "rotation_std_deg": np.degrees(np.sqrt(np.maximum(np.diag(covariance)[3:], 0))).tolist(),
            "convention": "translation in camera XYZ meters; local rotation vector in stencil XYZ radians",
            "method": "4x4 UV spatial-block bootstrap of observed correspondences",
            "calibrated": False, "systematic_depth_error_m": None,
            "excludes": ["depth bias", "intrinsic/extrinsic error", "image correspondence bias", "unobserved deformation"]}
