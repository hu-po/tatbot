"""Acquired DBV3 pens bound to physical resources (tatbot-inks/3): which tool and ink draws each pen, and for a
dipping tool its palette cap, the dip settings coming from the tool's datasheet. A one-pen drawing needs no file: the
arm's fitted tool draws it (ros_fitted)."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import yaml
from tatbot_contracts.dip import resolve_dip, resolve_rinse
from tatbot_contracts.ros_fitted import fitted_resource
from tatbot_contracts.ros_program import riding_keys

from tatbot_ink.errors import CompileError
from tatbot_ink.tools import load_tool


class _UniqueLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep=deep)
        if len(mapping) != len(node.value):
            raise CompileError('duplicate mapping key in resource document')
        return mapping


def _read_yaml(raw):
    try:
        return yaml.load(raw, Loader=_UniqueLoader)
    except yaml.YAMLError as error:
        raise CompileError(f'invalid resource document: {error}') from error


def _id(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}', value):
        raise CompileError(f'{name} requires a bounded identifier')
    return value


def _number(value, name, *, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise CompileError(f'{name} requires a finite {"positive" if positive else "nonnegative"} value')
    return value


def acquired_pens(design):
    return {(art['content_sha256'], layer['ink_id'])
            for item in design['placements'] for art in [design['artworks'][item['artwork_id']]]
            for layer in art['program']['layers']}


def fitted_bindings(design, *, repo, arm, tool_id, pen_mode):
    """One acquired pen constrains the configured assembly without declaring its pigment. A dipping tool drawn
    this way names no cap, so it never dips: a dry pass (a hover rehearsal, a diagnostic)."""
    pens = acquired_pens(design)
    if len(pens) != 1:
        raise CompileError('multiple acquired pens require explicit tatbot-inks/3 resource bindings')
    tool = load_tool(repo, arm, tool_id, allow_dip=True)
    tool.pop('dip_block', None)
    if pen_mode == 'ride':
        for key in riding_keys(tool):
            _number(tool[key], 'riding tool '+key)
    resource = fitted_resource(tool, pen_mode)
    return [resource], {next(iter(pens)): resource['id']}, {'mode': 'owner_fitted', 'file_sha256': None, 'substrate': None}


def _activation(value, tool):
    if not isinstance(value, dict) or value.get('method') != 'manual' or value.get('action') not in ('exchange', 'select'):
        raise CompileError('activation requires a manual exchange or selection')
    expected = {'method', 'action'} | ({'channel'} if value['action'] == 'select' else set())
    if set(value) != expected:
        raise CompileError('manual selection requires a channel; exchange does not take one')
    if 'channel' in value:
        _id(value['channel'], 'selected channel')
    if {key: val for key, val in value.items() if key != 'method'} != tool['cartridge_activation']:
        raise CompileError('activation differs from the selected tool configuration; a channel needs its own datasheet')
    return dict(value)


def _ink(tool, ink_id, catalog):
    ink = {'rgb': [0, 0, 0]} if tool['ink'] == 'none' and ink_id == 'none' else catalog.get(ink_id)
    if not isinstance(ink, dict):
        raise CompileError('physical ink is absent from config/inks.yaml')
    if tool['ink'] == 'none' and ink_id != 'none':
        raise CompileError('a tool without an ink supply cannot bind a physical ink')
    rgb = ink.get('rgb')
    if not isinstance(rgb, list) or len(rgb) != 3 or any(type(n) is not int or not 0 <= n <= 255 for n in rgb):
        raise CompileError('physical ink requires three 8-bit RGB channels')
    return rgb


def _resource(value, repo, arm, substrate, catalog, palette):
    base = {'id', 'tool_id', 'ink_id', 'pen_mode', 'activation'}
    if not isinstance(value, dict) or not base <= value.keys() or value.keys() - base - {'slot'}:
        raise CompileError('resource requires id, tool_id, ink_id, pen_mode and activation')
    identity = _id(value['id'], 'resource id')
    tool = load_tool(repo, arm, _id(value['tool_id'], 'tool id'), allow_dip=True)
    block = tool.pop('dip_block')
    supported = tool['substrates']
    if not isinstance(supported, list) or any(not isinstance(name, str) for name in supported) or substrate not in supported:
        raise CompileError(f'{identity}: tool does not support substrate {substrate}')
    ink_id = _id(value['ink_id'], 'physical ink id')
    rgb = _ink(tool, ink_id, catalog)
    if value['pen_mode'] not in ('press', 'ride', 'hover'):
        raise CompileError('resource pen_mode must be press, ride or hover')
    if value['pen_mode'] == 'ride':
        for key in riding_keys(tool):
            _number(tool[key], 'riding tool '+key)
    result = {'id': identity, 'tool': tool, 'ink_id': ink_id, 'rgb': rgb, 'pen_mode': value['pen_mode'],
              'activation': None, 'slot': None, 'dip': None}
    if tool['ink'] == 'dip' and value.get('slot') is not None:   # without a cap it draws dry: no dips
        slot = _id(value.get('slot'), 'canonical palette slot')
        if slot not in (palette.get('slots') or {}) or palette['slots'][slot].get('arm') != arm:
            raise CompileError('dipping requires a canonical palette slot assigned to the drawing arm')
        try:
            result.update(slot=slot, dip=resolve_dip(block, ink_id))
        except ValueError as error:
            raise CompileError(f'{identity}: {error}') from error
    elif value.get('slot') is not None:
        raise CompileError('cartridge/none supplies cannot declare dips or palette slots')
    activation = value['activation']
    rinse = isinstance(activation, dict) and activation.get('method') == 'rinse'
    result['activation'] = (_rinse(activation, result, block, arm, catalog, palette) if rinse
                            else _activation(activation, tool))
    return result


def _rinse(value, resource, block, arm, catalog, palette):
    """A dipping cartridge that takes this ink by rinsing the last one out in a cap of water: no landing."""
    if resource['dip'] is None or set(value) != {'method', 'slot', 'ink_id'}:
        raise CompileError(f"{resource['id']}: a rinse names its cap and liquid (slot, ink_id), for a resource that dips")
    slot = _id(value['slot'], 'rinse slot')
    if slot not in (palette.get('slots') or {}) or palette['slots'][slot].get('arm') != arm or slot == resource['slot']:
        raise CompileError(f"{resource['id']}: a rinse needs a palette slot of the drawing arm other than its ink's")
    ink_id = _id(value['ink_id'], 'rinse liquid')
    _ink(resource['tool'], ink_id, catalog)
    try:
        return resolve_rinse(block, slot, ink_id)
    except ValueError as error:
        raise CompileError(f"{resource['id']}: {error}") from error


def _bindings(value, pens, resources):
    bindings = {}
    if not isinstance(value, list):
        raise CompileError('resource bindings must be a list')
    for binding in value:
        if not isinstance(binding, dict) or set(binding) != {'artwork_sha256', 'pen_id', 'resource_id'}:
            raise CompileError('binding requires artwork_sha256, pen_id and resource_id')
        if not isinstance(binding['artwork_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', binding['artwork_sha256']):
            raise CompileError('binding requires an exact artwork digest')
        if not isinstance(binding['pen_id'], str) or not binding['pen_id']:
            raise CompileError('binding requires an acquired pen identifier')
        _id(binding['resource_id'], 'bound resource id')
        key = (binding['artwork_sha256'], binding['pen_id'])
        if key not in pens or key in bindings or binding['resource_id'] not in resources:
            raise CompileError('duplicate or unresolved acquired-pen/resource binding')
        bindings[key] = binding['resource_id']
    if set(bindings) != pens:
        raise CompileError('every acquired pen requires an exact resource binding')
    return bindings


def bind_resources(path, design, *, repo, arm, speed_m_s):
    """tatbot-inks/3: one substrate, explicit physical resources and exact artwork/pen bindings."""
    raw = Path(path).read_bytes()
    document = _read_yaml(raw)
    if not isinstance(document, dict) or set(document) != {'format', 'version', 'substrate', 'resources', 'bindings'}:
        raise CompileError('expected tatbot-inks/3 with substrate, resources and bindings')
    if document['format'] != 'tatbot-inks' or document['version'] != 3:
        raise CompileError('expected tatbot-inks/3')
    _number(speed_m_s, 'working speed')
    substrate = _id(document['substrate'], 'substrate')
    substrates = yaml.safe_load((repo / 'config/substrates.yaml').read_text())
    if substrate not in substrates or not isinstance(substrates[substrate], dict) or substrates[substrate].get('shape', 'pad') != 'pad':
        raise CompileError('ROS plane drawing requires a configured flat substrate')
    catalog = yaml.safe_load((repo / 'config/inks.yaml').read_text())
    palette = yaml.safe_load((repo / 'config/palette.yaml').read_text())
    if not isinstance(document['resources'], list) or not 1 <= len(document['resources']) <= 100:
        raise CompileError('expected 1..100 physical resources')
    resources = {}
    for value in document['resources']:
        resource = _resource(value, repo, arm, substrate, catalog, palette)
        if resource['id'] in resources:
            raise CompileError('resource IDs must be unique')
        resources[resource['id']] = resource
    bindings = _bindings(document['bindings'], acquired_pens(design), resources)
    return list(resources.values()), bindings, {'mode': 'explicit', 'file_sha256': hashlib.sha256(raw).hexdigest(),
                                               'substrate': substrate}
