"""Pose retained board branches through the same wrist geometry as surface fusion."""
from __future__ import annotations

import hashlib
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np

INPUTS = ('bundle.json', 'registration.json', 'golden.json', 'config/arms.json',
          'config/workspace.yaml', 'urdf/tatbot.urdf', 'vision.toml')


def fingerprints(root):
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in INPUTS}


class WristInputs:
    """Explicit retained configuration; no live configuration or device fallback."""

    def __init__(self, root, manifests):
        from stencil_observer import WristPoser

        self.root = Path(root)
        self.hashes = fingerprints(self.root)
        if not manifests or any(not isinstance(m.get('wrist'), dict) for m in manifests):
            raise ValueError('posed board captures need measured wrist metadata')
        identities = {(m.get('wrist', {}).get('arm'), m.get('wrist', {}).get('camera')) for m in manifests}
        if len(identities) != 1 or any(not value for value in next(iter(identities))):
            raise ValueError('posed board captures need one measured wrist camera and arm')
        self.arm, self.camera = next(iter(identities))
        self.bundle = json.loads((self.root/'bundle.json').read_bytes())
        self.golden = json.loads((self.root/'golden.json').read_bytes())
        binding = {'wrist': {'registrations': {self.arm: str(self.root/'registration.json')},
                            'robot_world_golden': str(self.root/'golden.json'),
                            'vision_config': str(self.root/'vision.toml')}}
        self.poser = WristPoser(binding, self.bundle, repo=self.root)

    def pose(self, path, manifest, color_sensor):
        from stencil_observer import wrist_view

        if color_sensor != self.camera+'_color':
            raise ValueError('posed board sensor differs from its measured wrist')
        camera = self.poser.by_camera.get(self.camera)
        if camera is None or camera['arm'] != self.arm:
            raise ValueError('posed board camera differs from retained registry')
        for stream in ('color', 'depth'):
            meta = manifest['frames'][self.camera+'_'+stream]['metadata']
            attributes, profile, expected = meta['attributes'], meta['profile'], camera[stream]
            if (attributes.get('device_serial') != str(camera['serial'])
                    or attributes.get('capture_owner_role') != camera['owner_role']):
                raise ValueError('posed board device identity differs from retained registry')
            converted = (expected['format'] == 'yuyv' and profile['format'] == 'rgb8'
                         and attributes.get('bus_source_format') == 'yuyv')
            if (any(profile[k] != expected[k] for k in ('stream', 'width', 'height', 'fps_num', 'fps_den'))
                    or profile['format'] != expected['format'] and not converted):
                raise ValueError('posed board profile differs from retained registry')
        _, (_, matrix), provenance = wrist_view(path, self.bundle, self.golden, self.poser)
        if json.loads(path.read_bytes()) != manifest:
            raise ValueError('retained capture changed during wrist pose preparation')
        if fingerprints(self.root) != self.hashes:
            raise ValueError('retained wrist configuration changed during witness preparation')
        from urdf_kinematics import driver_joint_names
        prefix = self.poser.arms[self.arm].urdf_prefix
        _, camera_frame = self.poser.camera_binding(self.arm, self.camera)
        flange_frame = prefix+'/link_6'
        wrist = manifest['wrist']
        corrected = np.r_[wrist['joints'], wrist['carriage_m']] + self.poser.joint_offsets[self.arm]
        values = dict(zip(driver_joint_names(prefix, 7), corrected.tolist(), strict=True))
        chain = self.poser.chain
        flange_from_camera = np.linalg.inv(chain.link_pose(flange_frame, values)) @ chain.link_pose(camera_frame, values)
        return {'reference_root_from_camera': matrix.tolist(), 'provenance': provenance,
                'flange_frame': flange_frame, 'nominal_flange_from_camera': flange_from_camera.tolist(),
                'reference_root_from_flange': (matrix @ np.linalg.inv(flange_from_camera)).tolist()}


def branch_pose(camera_pose, branch):
    matrix = np.eye(4)
    matrix[:3, :3] = branch['camera_from_tag_rotation']
    matrix[:3, 3] = branch['camera_from_tag_translation_m']
    return np.asarray(camera_pose) @ matrix


def consistency(records):
    """All pair/branch discrepancies, conditional on each individual tag staying fixed."""
    from fiducials.geometry import rotation_distance_deg

    rows = []
    for (ai, a), (bi, b) in combinations(enumerate(records), 2):
        left = {(t['family'], t['id']): t for t in a['tags']}
        right = {(t['family'], t['id']): t for t in b['tags']}
        for family, identifier in sorted(left.keys() & right.keys()):
            for aj, ab in enumerate(left[family, identifier]['pose_branches']):
                at = branch_pose(a['wrist_pose']['reference_root_from_camera'], ab)
                for bj, bb in enumerate(right[family, identifier]['pose_branches']):
                    bt = branch_pose(b['wrist_pose']['reference_root_from_camera'], bb)
                    rows.append({'capture_indices': [ai, bi], 'family': family, 'id': identifier,
                                 'branch_indices': [aj, bj],
                                 'translation_difference_m': float(np.linalg.norm(at[:3, 3]-bt[:3, 3])),
                                 'rotation_difference_deg': rotation_distance_deg(at[:3, :3], bt[:3, :3])})
    return {'comparisons': rows, 'branch_selection': 'none', 'mount_error_bound_m': None,
            'scope': 'conditional static individual-tag consistency; no hand-eye solution, adoption or absolute accuracy bound'}


def fit_mount(records, indices):
    """Unadopted per-tag hand-eye candidates, with explicit excluded-view checks."""
    import cv2
    from fiducials.geometry import rotation_distance_deg

    if not hasattr(cv2, 'calibrateHandEye'):
        raise ValueError('mount fitting needs an existing OpenCV interpreter with calibrateHandEye; no dependencies are installed')
    if (len(set(indices)) != len(indices) or not 3 <= len(indices) <= 8
            or any(type(i) is not int or not 0 <= i < len(records) for i in indices)
            or len(indices) == len(records)):
        raise ValueError('mount fit needs 3–8 distinct capture indices and excluded validation captures')
    tables = [{(t['family'], t['id']): t for t in r['tags']} for r in records]
    shared = set.intersection(*(set(tables[i]) for i in indices))
    results = []
    for family, identifier in sorted(shared):
        branches = [tables[i][family, identifier]['pose_branches'] for i in indices]
        for chosen in product(*(range(len(b)) for b in branches)):
            grippers = [np.asarray(records[i]['wrist_pose']['reference_root_from_flange']) for i in indices]
            targets = [branch_pose(np.eye(4), b[j]) for b, j in zip(branches, chosen, strict=True)]
            try:
                rot, translation = cv2.calibrateHandEye(
                    [m[:3, :3] for m in grippers], [m[:3, 3] for m in grippers],
                    [m[:3, :3] for m in targets], [m[:3, 3] for m in targets], method=cv2.CALIB_HAND_EYE_PARK)
                candidate = np.eye(4)
                candidate[:3, :3], candidate[:3, 3] = rot, translation.ravel()
                if (not np.isfinite(candidate).all() or np.linalg.det(rot) < 0
                        or not np.allclose(rot.T@rot, np.eye(3), atol=1e-6)):
                    raise ValueError('hand-eye returned a non-rigid or nonfinite candidate')
            except (cv2.error, ValueError) as error:
                results.append({'family': family, 'id': identifier, 'fit_branch_indices': list(chosen),
                                'status': 'unavailable', 'reason': str(error)})
                continue
            anchor = grippers[0] @ candidate @ targets[0]
            validation = []
            for i, table in enumerate(tables):
                if (family, identifier) not in table:
                    continue
                for j, branch in enumerate(table[family, identifier]['pose_branches']):
                    actual = np.asarray(records[i]['wrist_pose']['reference_root_from_flange']) @ candidate @ branch_pose(np.eye(4), branch)
                    validation.append({'capture_index': i, 'branch_index': j, 'used_for_fit': i in indices,
                                       'translation_difference_m': float(np.linalg.norm(actual[:3, 3]-anchor[:3, 3])),
                                       'rotation_difference_deg': rotation_distance_deg(anchor, actual)})
            nominal = np.asarray(records[indices[0]]['wrist_pose']['nominal_flange_from_camera'])
            results.append({'family': family, 'id': identifier, 'fit_branch_indices': list(chosen), 'status': 'candidate',
                            'flange_from_camera': candidate.tolist(), 'validation': validation,
                            'nominal_translation_difference_m': float(np.linalg.norm(candidate[:3, 3]-nominal[:3, 3])),
                            'nominal_rotation_difference_deg': rotation_distance_deg(nominal, candidate)})
    return {'fit_capture_indices': indices, 'candidates': results, 'branch_selection': 'none',
            'mount_error_bound_m': None, 'calibration_adopted': False,
            'scope': 'stationary individual-tag hand-eye candidates; excluded views are consistency checks, not absolute physical qualification'}
