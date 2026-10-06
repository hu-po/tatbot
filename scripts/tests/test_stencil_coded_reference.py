"""A coded print's reference binds its coded.json: exported, loaded, installed and bundled
together, and refused apart."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts/lib')]
import stencil_coded  # noqa: E402
import stencil_reference  # noqa: E402
import stencils  # noqa: E402

pytest.importorskip('PIL')


@pytest.fixture(scope='module')
def prints(tmp_path_factory):
    root = tmp_path_factory.mktemp('coded')
    return {seed: stencil_coded.generate(seed, root/seed) for seed in ('ref-a', 'ref-b')}


def copy(source, destination):
    shutil.copytree(source, destination)
    return destination/'tracking.json'


def test_constants_match_the_generator():
    assert stencil_coded.VERSION in stencil_reference.CODED_GENERATORS
    assert stencil_coded.SCHEME == stencil_reference.CODED_SCHEME


def test_a_coded_manifest_binds_its_code_and_print_id(prints):
    manifest, _ = stencil_reference.load(prints['ref-a']/'tracking.json')
    code = (prints['ref-a']/'coded.json').read_bytes()
    print_id = json.loads(code)['print_id']
    assert manifest['coded'] == {'file': 'coded.json', 'sha256': hashlib.sha256(code).hexdigest(),
                                 'scheme': stencil_coded.SCHEME, 'print_id': print_id}
    assert manifest['physical_instance_encoded'] is True
    assert manifest['physical_instance_id'] == print_id == stencil_coded.print_id_for('ref-a')
    assert 'instance_mark' not in manifest
    settings = json.loads((prints['ref-a']/'settings.json').read_text())
    assert settings['files']['coded.json'] == manifest['coded']['sha256']


def test_a_changed_missing_or_foreign_code_is_refused(prints, tmp_path):
    changed = copy(prints['ref-a'], tmp_path/'changed')
    code = json.loads(changed.with_name('coded.json').read_text())
    code['edges'][0][3] *= -1
    changed.with_name('coded.json').write_text(json.dumps(code))
    with pytest.raises(ValueError, match='coded.json hash mismatch'):
        stencil_reference.load(changed)
    missing = copy(prints['ref-a'], tmp_path/'missing')
    missing.with_name('coded.json').unlink()
    with pytest.raises(ValueError, match='without its coded.json'):
        stencil_reference.load(missing)
    foreign = copy(prints['ref-a'], tmp_path/'foreign')
    shutil.copy(prints['ref-b']/'coded.json', foreign.with_name('coded.json'))
    with pytest.raises(ValueError, match='coded.json hash mismatch'):
        stencil_reference.load(foreign)
    with pytest.raises(ValueError, match='artwork hash mismatch: coded.json'):
        stencil_reference.export(foreign.with_name('settings.json'))


def unbound(path):
    """A coded manifest as exported before coded.json was bound."""
    manifest = json.loads(path.read_text())
    for key in ('coded', 'physical_instance_id'):
        manifest.pop(key)
    manifest['physical_instance_encoded'] = False
    manifest['reference_id'] = stencil_reference.reference_digest(manifest)
    path.write_text(json.dumps(manifest))
    settings = json.loads(path.with_name('settings.json').read_text())
    settings['files'].pop('coded.json')
    path.with_name('settings.json').write_text(json.dumps(settings))


def test_an_unbound_coded_manifest_is_refused_until_re_exported(prints, tmp_path):
    old = copy(prints['ref-a'], tmp_path/'old')
    unbound(old)
    with pytest.raises(ValueError, match='re-export'):
        stencil_reference.load(old)
    reference = stencil_reference.export(old.with_name('settings.json'))
    assert stencil_reference.load(old)[0] == reference
    assert reference['pattern_id'] == stencil_reference.load(prints['ref-a']/'tracking.json')[0]['pattern_id']
    # An unrecorded code is bound only when it describes this artwork's print.
    other = copy(prints['ref-a'], tmp_path/'other')
    unbound(other)
    shutil.copy(prints['ref-b']/'coded.json', other.with_name('coded.json'))
    with pytest.raises(ValueError, match='does not describe this artwork'):
        stencil_reference.export(other.with_name('settings.json'))


def test_a_coded_identity_must_be_consistent(prints, tmp_path):
    path = copy(prints['ref-a'], tmp_path/'identity')
    manifest = json.loads(path.read_text())
    manifest['physical_instance_id'] = stencil_coded.print_id_for('ref-b')
    manifest['reference_id'] = stencil_reference.reference_digest(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='invalid coded print identity'):
        stencil_reference.load(path)


def test_install_and_bundle_carry_the_code(prints, tmp_path):
    installed = stencil_reference.install(prints['ref-a']/'tracking.json', root=tmp_path/'root')
    assert sorted(p.name for p in installed.iterdir()) == ['coded.json', 'settings.json', 'stencil.png', 'tracking.json']
    assert (installed/'settings.json').read_bytes() == (prints['ref-a']/'settings.json').read_bytes()
    assert stencil_reference.load(installed/'tracking.json')[0]['coded']['print_id']
    before = (installed/'coded.json').stat().st_mtime_ns
    assert stencil_reference.install(prints['ref-a']/'tracking.json', root=tmp_path/'root') == installed
    assert (installed/'coded.json').stat().st_mtime_ns == before
    bundle = stencils.bundle([str(prints['ref-a']/'tracking.json')])
    paths = stencils.materialize(bundle, tmp_path/'bundle')
    assert (paths[0].with_name('coded.json')).read_bytes() == (prints['ref-a']/'coded.json').read_bytes()
    assert stencil_reference.load(paths[0])[0]['coded'] == bundle['references'][0]['reference']['coded']


def test_border_inner_edges_are_measured_from_the_ink(prints):
    """Knots sit on junctions by their centres: the default layout's inner ink edges differ
    from the nominal clear centre, unevenly by side, and the edges they bound are ink-free."""
    settings = json.loads((prints['ref-a']/'settings.json').read_text())
    x0, y0, x1, y1 = settings['border_inner_mm']
    assert (x0, y0, x1, y1) != (19., 19., 81., 131.)
    assert all(abs(value-nominal) < 1.5 for value, nominal in zip((x0, y0, x1, y1), (19., 19., 81., 131.), strict=True))
    from PIL import Image
    ink = ~np.asarray(Image.open(prints['ref-a']/'stencil.png'), bool)
    scale = settings['dpi']/25.4
    rows = slice(round(40*scale), round(110*scale))
    assert not ink[rows, round(x0*scale)+1:round(x1*scale)-1].any()
    assert ink[rows, :round(x0*scale)+1].any() and ink[rows, round(x1*scale)-1:].any()


def test_an_installed_print_gives_the_drawing_stack_its_page(prints, tmp_path):
    """The drawing stack's page (tatbot_session.config.print_page) is the installed print's: a half-skin
    print's size and clear centre from its manifest, each side's innermost border ink from its settings."""
    half = stencil_coded.generate('half', tmp_path/'half', width_mm=83, height_mm=127, optimize_rounds=5)
    root = tmp_path/'root'
    manifest = json.loads((half/'tracking.json').read_text())
    assert stencil_reference.installed_page(manifest['pattern_id'], root) is None
    stencil_reference.install(half/'tracking.json', root=root)
    page = stencil_reference.installed_page(manifest['pattern_id'], root)
    x0, y0, x1, y1 = json.loads((half/'settings.json').read_text())['border_inner_mm']
    assert page == {'pattern_id': manifest['pattern_id'], 'size_m': [0.083, 0.127], 'clear_m': [0.045, 0.089],
                    'inner_edges_m': {'left': round(x0/1000-0.0415, 9), 'right': round(x1/1000-0.0415, 9),
                                      'bottom': round(0.0635-y1/1000, 9), 'top': round(0.0635-y0/1000, 9)}}
    edges = page['inner_edges_m']
    assert all(abs(abs(edges[side])-half_clear) < 0.0015 for side, half_clear in
               (('left', 0.0225), ('right', 0.0225), ('bottom', 0.0445), ('top', 0.0445)))
    # Another print's settings beside the manifest do not describe its border.
    shutil.copyfile(prints['ref-a']/'settings.json', root/'stencils/references'/manifest['pattern_id']/'settings.json')
    assert 'inner_edges_m' not in stencil_reference.installed_page(manifest['pattern_id'], root)
