"""Shared artwork validation; SVG paint conversion is source/fixture tooling.

There is no network fallback. Provenance identifiers are descriptive data,
never paths to resolve. An unknown license stays unknown at this boundary.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tatbot_contracts.artwork import validate_artwork_metadata

from tatbot_sim.human_rep.artwork_client import artwork_request
from tatbot_sim.human_rep.contracts import ContractError, validate_contract


def make_artwork_record(
    *, name: str, original_svg: str, source: dict[str, Any], conversion: dict[str, Any],
) -> dict[str, Any]:
    return artwork_request({"operation": "materialize_record", "input": {
        "name": name, "original_svg": original_svg, "source": source, "conversion": conversion,
    }})


def validate_artwork_record(value: dict[str, Any]) -> dict[str, Any]:
    validate_contract(value.get("program") if isinstance(value, dict) else None, expected_schema="tatbot.tattoo-program/1")
    try:
        validate_artwork_metadata(value)
    except ValueError as exc:
        raise ContractError("artwork_record_invalid", "$", str(exc)) from exc
    return deepcopy(value)


def artwork_preview(value: dict[str, Any]) -> str:
    """Derive a visual preview from validated geometry; never materialize it again."""
    record = validate_artwork_record(value)
    return artwork_request({"program": record["program"]})["svg"]
