from __future__ import annotations

import json
import socket
import struct

import numpy as np
import pytest
from sim_policy_protocol import (  # noqa: E402
    MAX_HEADER_BYTES,
    PROTOCOL,
    ProtocolError,
    recv_message,
    send_message,
)


def test_protocol_round_trips_json_and_explicit_contiguous_arrays():
    left, right = socket.socketpair()
    try:
        source = np.arange(48, dtype=np.float32).reshape(6, 8)[:, ::2]
        assert not source.flags.c_contiguous
        send_message(left, {"op": "step", "finite": 1.25}, {"action": source})
        header, arrays = recv_message(right)
        assert header == {"op": "step", "finite": 1.25}
        assert arrays["action"].flags.c_contiguous
        np.testing.assert_array_equal(arrays["action"], source)
    finally:
        left.close()
        right.close()


def test_protocol_rejects_executable_object_arrays_and_nonfinite_json():
    left, right = socket.socketpair()
    try:
        with pytest.raises(ProtocolError, match="unsupported dtype"):
            send_message(left, {"op": "x"}, {"bad": np.array([object()])})
        with pytest.raises(ProtocolError, match="finite JSON"):
            send_message(left, {"op": "x", "bad": float("nan")})
    finally:
        left.close()
        right.close()


def _send_document(sock: socket.socket, document: dict, payload: bytes = b"") -> None:
    encoded = json.dumps(document, separators=(",", ":")).encode()
    sock.sendall(struct.pack("!I", len(encoded)) + encoded + payload)


def test_protocol_rejects_version_mismatch_and_descriptor_lies():
    left, right = socket.socketpair()
    try:
        _send_document(left, {"protocol": "old", "arrays": []})
        with pytest.raises(ProtocolError, match="protocol mismatch"):
            recv_message(right)
    finally:
        left.close()
        right.close()

    left, right = socket.socketpair()
    try:
        _send_document(left, {
            "protocol": PROTOCOL,
            "arrays": [{
                "name": "x", "dtype": "<f4", "shape": [2], "order": "C",
                "offset": 0, "nbytes": 4,
            }],
        })
        with pytest.raises(ProtocolError, match="does not match"):
            recv_message(right)
    finally:
        left.close()
        right.close()


def test_protocol_rejects_unbounded_header_before_reading_it():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!I", MAX_HEADER_BYTES + 1))
        with pytest.raises(ProtocolError, match="header size"):
            recv_message(right)
    finally:
        left.close()
        right.close()
