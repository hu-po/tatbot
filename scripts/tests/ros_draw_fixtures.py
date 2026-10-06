"""Complete resource program using the real configured datasheet, for private SDK fixtures."""
import copy
from pathlib import Path

from ros_program_fixtures import program
from tatbot_contracts.canonical import canonical_digest
from tatbot_ink.tools import load_tool

REPO = Path(__file__).resolve().parents[2]


def bound_program(resource):
    tool = load_tool(REPO, 'right', resource['tool']['id'], allow_dip=True)
    tool.pop('dip_block')
    resource.update(tool=tool, rgb=[0, 0, 0], slot=None, dip=None)
    resource['content_sha256'] = canonical_digest(resource)
    value = program()
    value.update(resources=[resource], substrate='paper_pad')
    value['preparation']['ink_binding'] = {'mode': 'explicit', 'file_sha256': 'e'*64, 'substrate': 'paper_pad'}
    value['preparation']['pen_bindings'][0]['resource_id'] = resource['id']
    value['preparation']['content_sha256'] = canonical_digest(value['preparation'])
    value['ops'][0].update(resource_id=resource['id'], activation=resource['activation'])
    value['ops'][1].update(resource_id=resource['id'], ink=resource['ink_id'])
    return value


def transition_program(resource):
    """Two synthetic pigment resources, requiring an actual ordered service boundary."""
    value = bound_program(resource)
    following = copy.deepcopy(resource)
    following.update(id='fixture-blue', ink_id='blue', rgb=[0, 0, 255])
    following['content_sha256'] = canonical_digest(following)
    value['resources'].append(following)
    stroke = copy.deepcopy(value['ops'][1])
    stroke.update(id='s0004', resource_id=following['id'], ink=following['ink_id'])
    stroke['src']['pen'] = 'blue-pen'
    value['ops'].extend([{'op': 'tool_change', 'id': 't0001', 'initial': False,
                         'activation': following['activation'], 'resource_id': following['id']}, stroke])
    value['preparation']['pen_bindings'].append({'artwork_sha256': stroke['src']['artwork_sha256'],
                                               'pen_id': stroke['src']['pen'], 'resource_id': following['id']})
    value['preparation']['content_sha256'] = canonical_digest(value['preparation'])
    return value
