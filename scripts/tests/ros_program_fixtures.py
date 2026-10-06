"""A small one-pen tatbot-program for the CLI and contract tests."""
from tatbot_contracts.ros_fitted import fitted_resource


def program():
    tool = {'id': 'ball', 'ink': 'cartridge', 'datasheet_sha256': 'a'*64,
            'line_width_m': .0005, 'line_width_status': 'measured', 'substrates': ['paper'],
            'stroke_mm': None, 'tip_out_at_top_mm': None, 'cartridge_activation': {'action': 'exchange'}}
    resource = fitted_resource(tool, 'press')
    preparation = {'schema': 'tatbot-preparation/2', 'adapter': 'dbv3-paths-to-ros/2', 'motion_sha256': 'b'*64,
                   'max_chunk_s': 60, 'ink_binding': {'mode': 'owner_fitted', 'file_sha256': None, 'substrate': None},
                   'pen_bindings': [{'artwork_sha256': 'c'*64, 'pen_id': 'pen', 'resource_id': 'fitted'}]}
    return {'format': 'tatbot-program', 'version': 2, 'arm': 'right', 'substrate': None,
            'page': {'kind': 'stencil', 'size_m': [.1, .15], 'clear_m': [.062, .112]},
            'resources': [resource], 'draw_speed_m_s': .0035, 'preparation': preparation,
            'design': {'name': 'fixture'}, 'ops': [
                {'op': 'tool_change', 'id': 't0000', 'initial': True, 'activation': resource['activation'], 'resource_id': 'fitted'},
                {'op': 'stroke', 'id': 's0003', 'resource_id': 'fitted', 'ink': 'fitted', 'closed': False, 'continues': False,
                 'generation_width_m': .0005, 'planned_drawing_s': 1., 'points_m': [[0., 0.], [.005, 0.]],
                 'src': {'placement': 'p', 'artwork_sha256': 'c'*64, 'program_sha256': 'd'*64, 'layer': 'layer',
                         'path': 'path', 'pen': 'pen', 'closed': False, 'arc_m': [0., .005]}}]}
