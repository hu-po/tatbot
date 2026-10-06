"""Untrusted NPY headers must be refused before NumPy allocates arrays."""
import io
import warnings
import zipfile
import zlib

import numpy as np
import pytest
from bounded_npz import validate_npz_payload


def archive(members):
    result = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED) as target:
        warnings.simplefilter('ignore', UserWarning)
        for name, data in members:
            target.writestr(name, data)
    return result.getvalue()


def header(shape=(2, 3), dtype='<f8', data=b'', version=1):
    result = io.BytesIO()
    writer = {1: np.lib.format.write_array_header_1_0, 2: np.lib.format.write_array_header_2_0}[version]
    writer(result, {'descr': dtype, 'fortran_order': False, 'shape': shape})
    return result.getvalue()+data


def test_numeric_arrays_and_scalar_surface_metadata(tmp_path):
    import surface_model as ds
    surface = ds.HeightFieldSurface(ds.PlaneChart(np.zeros(3), np.eye(3)), np.zeros((4, 4)), .02, .02)
    surface.count[:] = 1
    surface.anchor_uv, surface.anchor_point = np.zeros(2), np.zeros(3)
    path = tmp_path / 'surface.npz'
    surface.to_npz(path)
    payload = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        assert validate_npz_payload(payload) == sum(member.file_size for member in source.infolist())
    assert ds.HeightFieldSurface.from_npz(path).height.shape == (4, 4)
    assert ds.HeightFieldSurface.from_npz(io.BytesIO(payload)).height.shape == (4, 4)


@pytest.mark.parametrize('shape,dtype,data', [
    ((10**12, 3), '<f8', b''), ((0, 10**100), '<f8', b''),
    ((2, 3), '<f8', b'\0'*8), ((1,), '|O', b'\0'*8),
    ((1, 1, 1, 1, 1), '|u1', b'\0'), ((10**12,), '|V0', b'')])
def test_forged_shapes_and_object_dtype_refuse_before_decoder(monkeypatch, shape, dtype, data):
    def forbidden(*args, **kwargs):
        pytest.fail('NumPy decoder reached during header validation')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='before allocation'):
        validate_npz_payload(archive([('array.npy', header(shape, dtype, data))]))


def test_decompression_budget_precedes_member_open(monkeypatch):
    payload = archive([('array.npy', header((100000,), '|u1', b'\0'*100000))])
    assert len(payload) < 1000
    def forbidden(*args, **kwargs):
        pytest.fail('compressed member opened before budget rejection')
    monkeypatch.setattr(zipfile.ZipFile, 'open', forbidden)
    with pytest.raises(ValueError, match='decompression budget'):
        validate_npz_payload(payload, max_decoded_bytes=1000)


@pytest.mark.parametrize('members,match', [
    ([], 'member count'), ([('a.npy', header()), ('a.npy', header())], 'duplicate'),
    ([('nested/a.npy', header())], 'unexpected'), ([('data.json', b'{}')], 'unexpected')])
def test_member_inventory_refuses(members, match):
    with pytest.raises(ValueError, match=match):
        validate_npz_payload(archive(members))


def test_member_count_cap_and_version_two():
    valid = header((1,), '|u1', b'\0', version=2)
    assert validate_npz_payload(archive([('a.npy', valid)])) == len(valid)
    with pytest.raises(ValueError, match='member count'):
        validate_npz_payload(archive([('a.npy', valid), ('b.npy', valid)]), max_members=1)


def test_truncated_and_oversized_headers_refuse():
    for payload in [b'not a zip', archive([('a.npy', b'\x93NUMPY\x01\x00\xff\xff')]),
                    archive([('a.npy', header(dtype='U10000')[:10]+b' ' * 12000)])]:
        with pytest.raises(ValueError):
            validate_npz_payload(payload)


def test_corrupt_compression_is_a_value_error(monkeypatch):
    payload = archive([('a.npy', header((1,), '|u1', b'\0'))])
    def corrupt(*args, **kwargs):
        raise zlib.error('corrupt compressed bytes')
    monkeypatch.setattr(zipfile.ZipFile, 'open', corrupt)
    with pytest.raises(ValueError, match='invalid NumPy archive'):
        validate_npz_payload(payload)
