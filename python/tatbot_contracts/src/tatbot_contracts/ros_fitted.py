"""The single resource of a one-pen program: the arm's fitted tool as configured, its ink not named.

The acquired pen's RGB is artwork metadata and does not become an ink; the operator loads whatever cartridge the
drawing wants before it starts. A program with several pens binds each to a resource explicitly (tatbot-inks/3).
"""
from __future__ import annotations

import copy


def owner_fitted(resource):
    return resource.get('identity_source') == 'owner_fitted'


def fitted_resource(tool, pen_mode):
    return {'id': 'fitted', 'identity_source': 'owner_fitted', 'tool': copy.deepcopy(tool),
            'ink_id': 'none' if tool['ink'] == 'none' else 'fitted', 'rgb': None, 'pen_mode': pen_mode,
            'activation': {'method': 'manual', **tool.get('cartridge_activation', {'action': 'exchange'})},
            'slot': None, 'dip': None}
