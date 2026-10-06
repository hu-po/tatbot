"""Bind Session-owned stencil observation to the accepted scan artifact.

The runtime selects the exact preview and acceptance receipts after scan
registration. Standalone diagnostics may explicitly follow previews instead.
An accepted geometry binding does not accept a later candidate material pose.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import schemas
from stencil_state import load_scan
from view_assets import bound_read, read


class ScanBinding:
    def __init__(self, run, *, from_session=False):
        self.run = Path(run)
        self.bound = None
        self.states = {}
        self.originals = {}
        self.error = None
        self.from_session = from_session
        self.selection = None

    def select(self, selection):
        """Consume the Session's explicit selection, never whichever file is newest."""
        if not self.from_session:
            raise ValueError('scan selection requires Session ownership')
        if selection == self.selection:
            return
        try:
            if schemas.is_schema(selection, schemas.SURFACE, 'stencil-origin'):
                self._origin(selection)
                return
            sequence = selection['scan_sequence']
            if (not schemas.is_schema(selection, schemas.SURFACE, 'stencil-selection')
                    or type(sequence) is not int or sequence < 0
                    or (self.bound is not None and sequence <= self.bound[0])):
                raise ValueError('Session scan selection changed or regressed')
            prefix = f'surface-{sequence}'
            for key, name in [('preview', 'registration-preview.json'), ('acceptance', 'registration-accepted.json')]:
                if selection[key]['path'] != f'{prefix}/{name}':
                    raise ValueError('Session scan receipt path differs from selected scan')
            preview_bytes = bound_read(self.run, selection['preview']['path'], selection['preview']['sha256'])
            acceptance = json.loads(bound_read(self.run, selection['acceptance']['path'], selection['acceptance']['sha256']))
            preview = json.loads(preview_bytes)
            if (not schemas.is_schema(preview, schemas.SURFACE, 'registration-preview')
                    or not schemas.is_schema(acceptance, schemas.SURFACE, 'registration-confirmation')
                    or acceptance.get('stationary_confirmed') is not True
                    or acceptance.get('preview_sha256') != hashlib.sha256(preview_bytes).hexdigest()):
                raise ValueError('Session scan acceptance does not bind the selected preview')
            self._apply(sequence, preview)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            self.states, self.error = {}, str(error)
            return
        self.selection = json.loads(json.dumps(selection))

    def _origin(self, selection):
        from stencil_origin import load_origin
        if self.bound is not None:
            raise ValueError('compiled original cannot replace an existing Session scan binding')
        states, sha = load_origin(self.run, selection)
        self.states, self.originals = states, dict(states)
        self.bound, self.error = (-1, sha), None
        self.selection = json.loads(json.dumps(selection))

    def refresh(self):
        if self.from_session:
            return
        candidates = []
        for path in self.run.glob('surface-*/registration-preview.json'):
            try:
                sequence = int(path.parent.name.removeprefix('surface-'))
            except ValueError:
                continue
            candidates.append((sequence, path))
        if not candidates:
            return
        sequence, path = max(candidates)
        if self.bound is not None and sequence <= self.bound[0]:
            return
        try:
            preview = json.loads(read(path, 1024*1024))
            self._apply(sequence, preview)
        except (OSError, ValueError, KeyError, TypeError) as error:
            # A new scan must never leave an older scan looking current after
            # its binding failed. Retry the atomic producer artifact next time.
            self.states = {}
            self.error = str(error)
            return

    def _apply(self, sequence, preview):
        relative = f'surface-{sequence}/stroke-check-{preview["first_chunk"]}/surface.npz'
        payload = bound_read(self.run, relative, preview['surface_sha256'])
        states = load_scan(payload, preview['surface_sha256'])
        if len(self.originals.keys() | states.keys()) > 8:
            raise ValueError('Session original stencil count exceeds eight')
        for pattern, incoming in states.items():
            if pattern in self.originals:
                # Validate every candidate before committing any changes.
                import copy
                candidate = copy.deepcopy(self.originals[pattern])
                candidate.incorporate_scan(incoming)
                states[pattern] = candidate
        self.states, self.bound, self.error = states, (sequence, preview['surface_sha256']), None
        self.originals.update(states)

    def observe(self, cameras, now_ns):
        self.refresh()
        mode = 'session-compiled' if self.bound and self.bound[0] == -1 else 'session-accepted'
        return {**schemas.stamp(schemas.SURFACE, 'stencil-state'), 'error': self.error,
                'scan_binding': mode if self.from_session else 'diagnostic-preview',
                'selection': self.selection,
                'scan_sequence': self.bound[0] if self.bound else None,
                'surfaces': [state.observe(cameras, now_ns) for state in self.states.values()],
                'motion_authority': False}

    def display(self, observation, references=None):
        """Expose the same retained surface under its estimated material pose."""
        rows = []
        budget = max(1, 30000//max(1, len(observation['surfaces'])))
        for value in observation['surfaces']:
            state = self.states[value['pattern_id']]
            points, colors = state.displayed_points(value)
            if not len(points):
                continue
            stride = max(1, int(np.ceil(len(points)/budget)))
            rows.append({'pattern_id': value['pattern_id'], 'source': 'registered_scan',
                         'seed': state.binding.get('seed', value['pattern_id']),
                         'capture_ns': value['pose_capture_ns'], 'geometry_capture_ns': value['geometry_capture_ns'],
                         'surface_sha256': value['surface_sha256'],
                         'geometry_revision_sha256': value['geometry_revision_sha256'],
                         'points': points[::stride].tolist(), 'colors': colors[::stride].tolist(),
                         **state.displayed_outline(value, (references or {}).get(value['pattern_id']))})
        return rows
