from __future__ import annotations

import pytest
from tatbot_sim.human_rep.contracts import ContractError, load_contract
from tatbot_sim.human_rep.placement import (
    HardenedPlacement,
    harden_face_distribution,
    make_surface_coordinate,
    make_surface_curve,
    make_surface_placement,
)
from tatbot_sim.repo import repo_root

EXAMPLES = repo_root() / "config" / "human-representation" / "examples"


def test_face_hardening_is_single_supported_and_deterministic():
    hardened = harden_face_distribution(
        [8, 4, 7],
        [0.4, 0.4, 0.2],
        [[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [1, 0, 0]],
        supported_faces={4, 8},
    )
    assert hardened == HardenedPlacement(4, (0.1, 0.4, 0.5), 0.4)
    with pytest.raises(ContractError) as caught:
        harden_face_distribution([8], [1], [[1, 0, 0]], supported_faces={4})
    assert caught.value.code == "anchor_outside_domain"


def test_surface_builders_emit_strict_self_hashed_contracts():
    source = load_contract(EXAMPLES / "surface-placement.json")
    anchor = source["anchor"]
    coordinate = make_surface_coordinate(
        topology_sha256=anchor["topology_sha256"],
        face_index=anchor["face_index"],
        barycentric=anchor["barycentric"],
    )
    assert coordinate["schema"] == "tatbot.surface-coordinate/1"
    placement = make_surface_placement(
        tattoo_program_sha256=source["tattoo_program_sha256"],
        body_identity_sha256=source["body_identity_sha256"],
        rest_surface_sha256=source["rest_surface_sha256"],
        topology_sha256=anchor["topology_sha256"],
        semantic_site=source["semantic_site"],
        laterality=source["laterality"],
        anchor=(anchor["face_index"], anchor["barycentric"]),
        physical_scale_m=tuple(source["physical_scale_m"]),
        rotation_rad=source["rotation_rad"],
        mirrored=source["mirrored"],
        supported_faces=source["supported_domain"]["face_indices"],
        margin_m=source["supported_domain"]["margin_m"],
        review=source["review"],
        provenance=source["provenance"],
    )
    assert placement["anchor"] == source["anchor"]
    curve = make_surface_curve(
        topology_sha256=anchor["topology_sha256"],
        addresses=[(1200, [0.2, 0.3, 0.5]), (1201, [0.1, 0.6, 0.3])],
        rest_points_m=[[0, 0, 0], [0.01, 0, 0]],
        width_m=0.0008,
        deposition=0.7,
        direction="forward",
        source_primitive_sha256="1" * 64,
        compiler_sha256="2" * 64,
    )
    assert curve["rest_surface_arc_length_m"] == pytest.approx(0.01)
