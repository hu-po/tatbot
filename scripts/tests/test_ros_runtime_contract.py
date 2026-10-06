"""Stable runtime source snapshots and shared native provenance without ROS imports."""
import json
import os
import shutil
from pathlib import Path

import pytest
from tatbot_contracts import ros_runtime as runtime
from tatbot_contracts.process import alive, process_stamp


@pytest.fixture
def snapshot(tmp_path):
    roots = {'shared': str(tmp_path/'shared'), 'owner': str(tmp_path/'owner')}
    for root in roots.values():
        path = Path(root)
        path.mkdir()
        (path/'source.py').write_text('VALUE = 1\n')
    return {'source_roots': roots, 'sources_sha256': runtime.source_digest(roots)}


def test_source_identity_ignores_checkout_location_and_detects_dependency_bytes(snapshot, tmp_path):
    assert runtime.source_status(snapshot)['complete']
    copy = {name: str(tmp_path/'copy'/name) for name in snapshot['source_roots']}
    for name, root in snapshot['source_roots'].items():
        shutil.copytree(root, copy[name])
    assert runtime.source_digest(copy) == snapshot['sources_sha256']
    (Path(copy['owner'])/'source.py').write_text('VALUE = 2\n')
    assert runtime.source_digest(copy) != snapshot['sources_sha256']
    assert runtime.source_status(snapshot)['complete']


@pytest.mark.parametrize('mutation', ['changed', 'added', 'removed', 'missing', 'empty'])
def test_source_drift_never_borrows_startup_identity(snapshot, mutation):
    root = Path(snapshot['source_roots']['owner'])
    source = root/'source.py'
    if mutation == 'changed':
        source.write_text('VALUE = 2\n')
    elif mutation == 'added':
        (root/'another.py').write_text('VALUE = 1\n')
    elif mutation == 'removed':
        source.unlink()
    elif mutation == 'missing':
        shutil.rmtree(root)
    else:
        source.unlink()
        (root/'data.json').write_text('{}')
    observed = runtime.source_status(snapshot)
    assert not observed['complete'] and observed['reason']


@pytest.mark.parametrize('mutation', ['changed', 'added', 'removed'])
def test_change_during_collection_cannot_be_published_as_stable(snapshot, mutation, monkeypatch):
    source = Path(snapshot['source_roots']['shared'])/'source.py'
    original = Path.read_bytes
    changed = False

    def read(path):
        nonlocal changed
        value = original(path)
        if not changed:
            changed = True
            if mutation == 'changed':
                source.write_text('VALUE = 2\n')
            elif mutation == 'added':
                (source.parent/'another.py').write_text('VALUE = 1\n')
            else:
                source.unlink()
        return value

    monkeypatch.setattr(Path, 'read_bytes', read)
    observed = runtime.source_status(snapshot)
    assert not observed['complete'] and observed['reason']


@pytest.mark.parametrize('roots', [None, {}, {'owner': '.'}, {'owner': True}])
def test_absent_or_invalid_source_manifest_is_unknown(roots):
    assert not runtime.source_status({'source_roots': roots, 'sources_sha256': 'a'*64})['complete']


def test_data_artifacts_do_not_change_code_identity(snapshot):
    (Path(snapshot['source_roots']['owner'])/'data.json').write_text('{}')
    assert runtime.source_status(snapshot)['complete']


@pytest.fixture
def controller(tmp_path):
    proc = tmp_path/'proc'
    process = proc/'123'
    process.mkdir(parents=True)
    boot = proc/'sys/kernel/random/boot_id'
    boot.parent.mkdir(parents=True)
    boot.write_text('boot-one\n')
    (process/'stat').write_text('123 (controller with ) in name) S '+'0 '*18+'456 0\n')
    binary = tmp_path/'ros2_control_node'
    binary.write_bytes(b'controller executable')
    library = tmp_path/'libcontroller.so'
    library.write_bytes(b'loaded plugin')
    (process/'exe').symlink_to(binary)
    maps = []
    for path in (binary, library):
        stat = path.stat()
        maps.append(f'1000-2000 r-xp 0 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} {stat.st_ino} {path}')
    maps.append('2000-3000 r-xp 0 00:00 0 [vdso]')
    (process/'maps').write_text('\n'.join(maps)+'\n')
    reference = tmp_path/'controller-process.json'
    reference.write_text(json.dumps(process_stamp(123, proc=proc)))
    return proc, reference, binary, library



def test_loaded_executable_and_plugin_identity_is_repeatable(controller):
    proc, reference, binary, library = controller
    identity = runtime.controller_identity(reference, proc=proc)
    assert identity['complete'], identity
    assert {item['path'] for item in identity['files']} == {str(binary), str(library)}
    assert identity == runtime.controller_identity(reference, proc=proc)
    library.write_bytes(b'changed plugin')
    changed = runtime.controller_identity(reference, proc=proc)
    assert changed['complete'] and changed['files_sha256'] != identity['files_sha256']



@pytest.mark.parametrize('change', ['pid-reuse', 'reboot', 'zombie', 'deleted', 'replaced', 'anonymous', 'wrong-executable'])
def test_incomplete_native_evidence_cannot_substitute_for_loaded_code(controller, change):
    proc, reference, binary, library = controller
    if change == 'pid-reuse':
        stat = proc/'123/stat'
        stat.write_text(stat.read_text().replace('456', '457'))
    elif change == 'reboot':
        (proc/'sys/kernel/random/boot_id').write_text('boot-two\n')
    elif change == 'zombie':
        stat = proc/'123/stat'
        stat.write_text(stat.read_text().replace(') S ', ') Z '))
    elif change == 'deleted':
        maps = proc/'123/maps'
        maps.write_text(maps.read_text().replace(str(library), str(library)+' (deleted)'))
    elif change == 'replaced':
        replacement = library.with_suffix('.new')
        replacement.write_bytes(library.read_bytes())
        replacement.replace(library)
    elif change == 'anonymous':
        with (proc/'123/maps').open('a') as stream:
            stream.write('3000-4000 r-xp 0 00:00 0\n')
    else:
        (proc/'123/exe').unlink()
        (proc/'123/exe').symlink_to(library)
    observed = runtime.controller_identity(reference, proc=proc)
    assert not observed['complete'] and observed['files_sha256'] is None and observed['reason']
    assert binary.exists()



def test_controller_receipt_records_only_process_identity(tmp_path):
    reference = tmp_path/'controller-process.json'
    runtime.record_controller(reference, os.getpid())
    receipt = json.loads(reference.read_text())
    assert alive(receipt)
    assert set(receipt) == {'pid', 'process_start', 'boot_id'}
    # A live unrelated process is not accepted as the controller.
    assert not runtime.controller_identity(reference)['complete']
    assert not list(tmp_path.glob('*.pending'))



def test_completed_pair_software_identity_ignores_process_and_inode_but_not_code(controller):
    import copy

    proc, reference, _, _ = controller
    original = {'schema': 'tatbot.ros-runtime/2', 'sources_sha256': 'a'*64, 'configuration_sha256': 'c'*64, 'python': '3.12', 'numpy': '2.0',
                'pid': 11, 'controller': runtime.controller_identity(reference, proc=proc)}
    restarted = copy.deepcopy(original)
    restarted.update(pid=22, boot_id='another-boot', revision_at_start={'sha': 'documentation-only'})
    restarted['controller']['process']['pid'] += 1
    for item in restarted['controller']['files']:
        item['inode'] += 1
    assert runtime.software_digest(original) == runtime.software_digest(restarted)
    restarted['controller']['files'][0]['sha256'] = 'b'*64
    assert runtime.software_digest(original) != runtime.software_digest(restarted)
    with pytest.raises(ValueError, match='missing research'):
        runtime.software_digest({'schema': 'tatbot.ros-runtime/2', 'controller': {'complete': True}})



def test_launch_inputs_exclude_evidence_paths_but_keep_geometry_and_controller_settings():
    template = '<robot><ros2_control><hardware><param name="flight_path">{path}</param><param name="limit">2</param></hardware></ros2_control><link name="tool"/></robot>'
    stack = {'rt': {'priority': 80}, 'flight_path': '/run-one', 'controller_process_file': '/run-one/controller.json',
             'runtime_record': '/workspace/runtime.json'}
    first = runtime.configuration_digest(stack, template.format(path='/run-one'), b'gain: 1')
    moved = {**stack, 'flight_path': '/run-two', 'controller_process_file': '/run-two/controller.json',
             'config_path': '/loaded-stack', 'runtime_record': '/test-tmp/runtime.json', 'runtime_configuration_sha256': 'old',
             'runtime_workspace_sha256': 'a'*64, 'service_refresh_id': 'b'*64}
    assert first == runtime.configuration_digest(moved, template.format(path='/run-two'), b'gain: 1')
    assert first != runtime.configuration_digest(stack, template.format(path='/run-one'), b'gain: 2')
    assert first != runtime.configuration_digest(stack, template.format(path='/run-one').replace('>2<', '>3<'), b'gain: 1')
    assert first != runtime.configuration_digest({**stack, 'rt': {'priority': 90}}, template.format(path='/run-one'), b'gain: 1')



def test_launched_geometry_carries_workspace_bytes_and_refuses_an_edit_during_render(tmp_path):
    import hashlib

    path = tmp_path/'config/workspace.yaml'
    path.parent.mkdir()
    path.write_bytes(b'right: {}\n')
    assert runtime.workspace_description(tmp_path, lambda: '<robot/>') == ('<robot/>', hashlib.sha256(path.read_bytes()).hexdigest())

    def edit():
        path.write_bytes(b'left: {}\n')
        return '<robot/>'

    with pytest.raises(ValueError, match='workspace changed'):
        runtime.workspace_description(tmp_path, edit)
