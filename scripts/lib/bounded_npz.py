"""Validate NumPy archive allocation bounds before callers decode any arrays.

Callers must bound the compressed input read first. This helper reads only
bounded NPY headers, never calls np.load, and does not establish data provenance.
"""
from __future__ import annotations

import io
import math
import zipfile
import zlib

import numpy as np


def _member_header(archive, member, limit):
    if ('/' in member.filename or '\\' in member.filename
            or not member.filename.endswith('.npy') or member.flag_bits & 1):
        raise ValueError('unexpected or encrypted NumPy archive member')
    with archive.open(member) as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError('unsupported NumPy header version')
        shape, _, dtype = readers[version](stream, max_header_size=10000)
        size = math.prod(shape) * dtype.itemsize
        if (len(shape) > 4 or any(dimension < 0 or dimension > limit for dimension in shape)
                or dtype.hasobject or dtype.itemsize <= 0 or size > limit
                or size != member.file_size - stream.tell()):
            raise ValueError('NumPy shape or dtype exceeds retained payload before allocation')


def validate_npz_payload(payload: bytes, max_decoded_bytes=128*1024*1024, max_members=64):
    """Return total uncompressed member bytes, or refuse before array allocation.

    Only flat, unique NPY members are accepted, including scalar string metadata
    and ordinary numeric arrays used by measured surface artifacts. Header bytes
    count toward the decoded budget. NPY versions 1 and 2 are supported.
    """
    if (not isinstance(payload, bytes) or type(max_decoded_bytes) is not int
            or max_decoded_bytes <= 0 or type(max_members) is not int or not 1 <= max_members <= 64):
        raise ValueError('invalid bounded NumPy archive request')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if not 0 < len(members) <= max_members:
                raise ValueError('NumPy archive member count exceeds budget')
            if len({member.filename for member in members}) != len(members):
                raise ValueError('duplicate NumPy archive members')
            decoded = sum(member.file_size for member in members)
            if decoded > max_decoded_bytes:
                raise ValueError('NumPy archive exceeds decompression budget')
            for member in members:
                _member_header(archive, member, max_decoded_bytes)
            return decoded
    except (zipfile.BadZipFile, EOFError, OSError, RuntimeError, NotImplementedError, OverflowError, zlib.error) as error:
        raise ValueError(f'invalid NumPy archive: {error}') from error
