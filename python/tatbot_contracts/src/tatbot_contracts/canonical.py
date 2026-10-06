"""Strict JSON and canonical digests shared by CLI and runtime readers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

MAX_SAFE_INTEGER = 2**53 - 1


class ContractError(ValueError):
    """A stable, named refusal for a malformed contract."""

    def __init__(self, code: str, path: str, detail: str):
        super().__init__(f"{code} at {path}: {detail}")
        self.code = code
        self.path = path
        self.detail = detail


def _reject_constant(value: str) -> None:
    raise ContractError("non_finite", "$", value)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate_key", "$", key)
        result[key] = value
    return result


def parse_json(data: str | bytes) -> Any:
    """Parse JSON without duplicate keys or JavaScript numeric extensions."""

    try:
        value = json.loads(
            data,
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs,
        )
    except ContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("invalid_json", "$", str(exc)) from exc
    _walk_unicode(value)
    _walk_numbers(value)
    return value


def _walk_unicode(value: Any, path: str = "$") -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ContractError("invalid_json", path, "lone Unicode surrogate")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_unicode(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _walk_unicode(key, f"{path} key")
            _walk_unicode(item, f"{path}.{key}")


def _walk_numbers(value: Any, path: str = "$", *, canonical: bool = True) -> None:
    """Reject numbers a document may not carry.

    Three rules, and only one of them is about JSON at all. A non-finite float
    is not representable: `json.dumps` writes a bare `NaN`, which no strict
    parser reads back, so it is refused for every caller. The safe-integer bound
    and the negative-zero rule are canonicality -- they keep two equal documents
    from hashing differently, and keep a value from changing as it passes through
    a browser. Those apply to digest material only (`canonical=True`); an artifact
    written for a person to read may legitimately hold -0.0 or a nanosecond
    timestamp, both of which exceed what a digest will accept.
    """

    if isinstance(value, int) and not isinstance(value, bool):
        if canonical and abs(value) > MAX_SAFE_INTEGER:
            raise ContractError("unsafe_integer", path, "outside the IEEE-754 safe-integer domain")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("non_finite", path, repr(value))
        if canonical and value == 0.0 and math.copysign(1.0, value) < 0:
            raise ContractError("negative_zero", path, "-0 is not canonical")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_numbers(item, f"{path}[{index}]", canonical=canonical)
    elif isinstance(value, dict):
        for key, item in value.items():
            _walk_numbers(item, f"{path}.{key}", canonical=canonical)


def _canonical_float(value: float) -> str:
    """Serialize binary64 values with the ECMAScript/RFC 8785 spelling."""

    if value == 0:
        return "0"
    if value < 0:
        return f"-{_canonical_float(-value)}"

    rendered = str(value)
    exponent = 0
    exponent_text = ""
    if "e" in rendered:
        rendered, raw_exponent = rendered.split("e", maxsplit=1)
        exponent = int(raw_exponent)
        exponent_text = f"e{exponent:+d}"

    first, dot, last = rendered.partition(".")
    if last == "0":
        dot = ""
        last = ""

    if 0 < exponent < 21:
        first += last
        last = ""
        dot = ""
        exponent_text = ""
        first += "0" * max(0, exponent - len(first) + 1)
    elif -7 < exponent < 0:
        last = ("0" * (-exponent - 1)) + first + last
        first = "0"
        dot = "."
        exponent_text = ""

    return f"{first}{dot}{last}{exponent_text}"


def _canonical_text(value: Any, path: str = "$") -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        parts = (_canonical_text(item, f"{path}[{index}]") for index, item in enumerate(value))
        return f"[{','.join(parts)}]"
    if isinstance(value, dict):
        try:
            items = sorted(value.items(), key=lambda item: item[0].encode("utf-16be"))
        except (AttributeError, UnicodeEncodeError) as exc:
            raise ContractError("invalid_json", path, "object keys must be Unicode strings") from exc
        parts = (
            f"{json.dumps(key, ensure_ascii=False)}:{_canonical_text(item, f'{path}.{key}')}"
            for key, item in items
        )
        return f"{{{','.join(parts)}}}"
    raise ContractError("wrong_type", path, f"unsupported JSON type {type(value).__name__}")


def canonical_bytes(value: Any, *, omit_digest: bool = False) -> bytes:
    """Return the version-one canonical UTF-8 JSON representation."""

    _walk_numbers(value)
    material = value
    if omit_digest:
        if not isinstance(value, dict):
            raise ContractError("wrong_type", "$", "digest-bearing document must be an object")
        material = {key: item for key, item in value.items() if key != "content_sha256"}
    try:
        return _canonical_text(material).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractError("invalid_json", "$", "input contains a lone surrogate") from exc


def canonical_digest(value: Any) -> str:
    """Hash a document with its top-level content digest omitted."""

    return hashlib.sha256(canonical_bytes(value, omit_digest=True)).hexdigest()


def indented_bytes(value: Any) -> bytes:
    """Readable JSON for an artifact a person opens, under the canonical guards.

    `canonical_bytes` is digest material: compact, and the exact bytes a hash is
    taken over. This is the other half -- indented and key-sorted so a diff is
    legible -- and it exists so a written artifact cannot dodge the checks a
    digested one gets. Nine evidence and audit generators each carried their own
    copy of this line, and a plain `json.dumps` emits `NaN` and `Infinity`, which
    no strict JSON parser will read back; an empty sample set is enough to
    produce one. Here a non-finite number is refused.

    The guard is narrower than the one `canonical_bytes` applies: an artifact a
    person reads may hold -0.0 or an integer past the browser-safe range, and
    neither costs anything here, where both would break a digest.

    The byte format is deliberately identical to those nine copies (indent 2,
    sorted keys, non-ASCII kept literal, one trailing newline), so adopting it
    changes no existing artifact and no recorded digest of one.
    """

    _walk_numbers(value, canonical=False)
    _walk_unicode(value)
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"


def write_json(path: Path | str, value: Any) -> None:
    """Write `indented_bytes(value)` to `path`.

    Nine generators defined this identically. It is two lines, but it is the line
    that decides an artifact's bytes, and keeping it in one place is what makes
    the guard above unskippable rather than a thing each new script re-derives.
    """

    Path(path).write_bytes(indented_bytes(value))
