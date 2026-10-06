"""Safe, versioned local transport for the simulator policy worker.

The LeRobot server wire has its own compatibility protocol.  This transport is
only for the local process boundary between the LeRobot client environment and
the ManiSkill environment.  Messages are a bounded JSON header followed by
explicit C-contiguous array bytes; executable object formats such as pickle are
deliberately not accepted here.
"""

from __future__ import annotations

import json
import socket
import struct
from collections.abc import Mapping

import numpy as np

PROTOCOL = "tatbot.sim-policy-worker/1"
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 256 << 20
_PREFIX = struct.Struct("!I")


class ProtocolError(ValueError):
    """A peer sent a malformed or incompatible message."""


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ProtocolError(
                f"connection closed with {remaining} of {count} bytes still expected"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _array_descriptor(name: str, value: object, offset: int) -> tuple[dict, np.ndarray]:
    if not isinstance(name, str) or not name or len(name) > 128:
        raise ProtocolError(f"invalid array name {name!r}")
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject or array.dtype.fields is not None:
        raise ProtocolError(f"array {name!r} has unsupported dtype {array.dtype}")
    if array.ndim > 8:
        raise ProtocolError(f"array {name!r} has too many dimensions: {array.ndim}")
    descriptor = {
        "name": name,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "order": "C",
        "offset": offset,
        "nbytes": array.nbytes,
    }
    return descriptor, array


def send_message(
    sock: socket.socket,
    header: Mapping[str, object],
    arrays: Mapping[str, object] | None = None,
) -> None:
    """Send one framed message without accepting implicit object serialization."""

    if not isinstance(header, Mapping):
        raise ProtocolError("message header must be a mapping")
    if "arrays" in header or "protocol" in header:
        raise ProtocolError("protocol and arrays are transport-owned header fields")
    descriptors = []
    contiguous = []
    offset = 0
    for name, value in (arrays or {}).items():
        descriptor, array = _array_descriptor(name, value, offset)
        descriptors.append(descriptor)
        contiguous.append(array)
        offset += array.nbytes
    if offset > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"payload is too large: {offset} bytes")
    document = {"protocol": PROTOCOL, **dict(header), "arrays": descriptors}
    try:
        encoded = json.dumps(
            document, separators=(",", ":"), sort_keys=True, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"header is not finite JSON: {exc}") from exc
    if not encoded or len(encoded) > MAX_HEADER_BYTES:
        raise ProtocolError(f"header size is invalid: {len(encoded)} bytes")
    sock.sendall(_PREFIX.pack(len(encoded)))
    sock.sendall(encoded)
    for array in contiguous:
        sock.sendall(memoryview(array).cast("B"))


def _validate_descriptors(raw: object) -> tuple[list[dict], int]:
    if not isinstance(raw, list):
        raise ProtocolError("arrays must be a list")
    descriptors: list[dict] = []
    names: set[str] = set()
    expected_offset = 0
    for item in raw:
        if not isinstance(item, dict):
            raise ProtocolError("array descriptor must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name or len(name) > 128 or name in names:
            raise ProtocolError(f"invalid or duplicate array name {name!r}")
        names.add(name)
        if item.get("order") != "C" or item.get("offset") != expected_offset:
            raise ProtocolError(f"array {name!r} is not a contiguous C-order payload")
        shape = item.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) > 8
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape)
        ):
            raise ProtocolError(f"array {name!r} has invalid shape {shape!r}")
        try:
            dtype = np.dtype(item.get("dtype"))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"array {name!r} has invalid dtype") from exc
        if dtype.hasobject or dtype.fields is not None:
            raise ProtocolError(f"array {name!r} has unsupported dtype {dtype}")
        nbytes = item.get("nbytes")
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if (
            not isinstance(nbytes, int)
            or isinstance(nbytes, bool)
            or nbytes < 0
            or nbytes != expected_nbytes
        ):
            raise ProtocolError(
                f"array {name!r} byte count {nbytes!r} does not match {expected_nbytes}"
            )
        expected_offset += nbytes
        if expected_offset > MAX_PAYLOAD_BYTES:
            raise ProtocolError(f"payload is too large: {expected_offset} bytes")
        descriptors.append({**item, "_dtype": dtype})
    return descriptors, expected_offset


def recv_message(sock: socket.socket) -> tuple[dict, dict[str, np.ndarray]]:
    """Receive and validate one framed message, returning copied arrays."""

    size = _PREFIX.unpack(_recv_exact(sock, _PREFIX.size))[0]
    if size == 0 or size > MAX_HEADER_BYTES:
        raise ProtocolError(f"header size is invalid: {size} bytes")
    try:
        document = json.loads(_recv_exact(sock, size))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"header is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ProtocolError("header root must be an object")
    if document.get("protocol") != PROTOCOL:
        raise ProtocolError(
            f"protocol mismatch: expected {PROTOCOL!r}, got {document.get('protocol')!r}"
        )
    descriptors, payload_size = _validate_descriptors(document.get("arrays"))
    payload = _recv_exact(sock, payload_size)
    arrays: dict[str, np.ndarray] = {}
    for descriptor in descriptors:
        start = descriptor["offset"]
        stop = start + descriptor["nbytes"]
        arrays[descriptor["name"]] = np.frombuffer(
            payload[start:stop], dtype=descriptor["_dtype"]
        ).copy().reshape(descriptor["shape"])
    header = {key: value for key, value in document.items() if key not in {"protocol", "arrays"}}
    return header, arrays
