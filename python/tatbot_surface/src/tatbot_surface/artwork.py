"""Validate one artwork record: its TattooProgram/1 and the shared envelope.

There is no network fallback. Provenance identifiers are descriptive data,
never paths to resolve. An unknown license stays unknown at this boundary.
Making a record or a preview from SVG is source tooling and is not here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tatbot_contracts.artwork import validate_artwork_metadata
from tatbot_contracts.canonical import ContractError

from tatbot_surface.program_contract import validate_tattoo_program


def validate_artwork_record(value: dict[str, Any]) -> dict[str, Any]:
    validate_tattoo_program(value.get("program") if isinstance(value, dict) else None)
    try:
        validate_artwork_metadata(value)
    except ValueError as exc:
        raise ContractError("artwork_record_invalid", "$", str(exc)) from exc
    return deepcopy(value)
