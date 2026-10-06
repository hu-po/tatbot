"""Report capability consumers and evidence for four independent approval packets."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tatbot_contracts.canonical import write_json
from tatbot_contracts.digest import sha256_file

from tatbot_sim.human_rep.contracts import parse_json
from tatbot_sim.human_rep.review import APPROVALS, load_release_gates
from tatbot_sim.repo import git_output

PACKET_DETAILS = {
    "inkmap_deployment": {
        "action": "deployment_enabled",
        "scope": "deploy the MHR/SOMA-only Inkmap consumer",
        "required_capabilities": ["source_contracts", "nominal_body", "inkmap"],
        "review": [
            "clean published single-body revision",
            "model, asset, license, SBOM, atlas, parity, performance, and visual evidence",
            "live target manifest and rollback owner",
        ],
    },
    "public_release": {
        "action": "public_export_enabled",
        "scope": "promote a reviewed sanitized tree to the public repository",
        "required_capabilities": ["source_contracts", "nominal_body", "inkmap"],
        "review": [
            "disclosure, secret, topology, license, and data-boundary checks rerun on the export",
            "no private plan, internal path, fleet topology, sensitive body data, or anatomy pack",
            "one MHR/SOMA implementation with no compatibility body path",
        ],
    },
    "powered_evaluation": {
        "action": "powered_motion_enabled",
        "scope": "perform a separately authorized powered robot evaluation",
        "required_capabilities": ["artwork_compilation", "stroke_lowering", "registration"],
        "review": [
            "qualified instrument and approved non-human phantom registration evidence",
            "current robot, tool, calibration, reach, collision, and E-stop preflight",
            "physically present operator approval for the exact run",
        ],
    },
    "human_facing_study": {
        "action": "human_contact_enabled",
        "scope": "begin any human-facing study or contact",
        "required_capabilities": ["artwork_compilation", "stroke_lowering", "registration", "phantom_mechanics"],
        "review": [
            "separate study, ethics, consent, privacy, clinical, and operational approvals",
            "qualified physical evidence appropriate to the declared use",
            "no inference from phantom, anatomy, or synthetic evidence to human safety",
        ],
    },
}


# Legacy numbers locate immutable historical packets only. They are never
# used to infer current capability status, admission, or physical acceptance.
CAPABILITIES = {
    "source_contracts": (0, "maintained", "body audit and typed readers", "software",
                         "reviewed asset and software bytes", "repeat cache and contract checks on the candidate revision"),
    "nominal_body": (1, "maintained", "Inkmap asset export and simulation rig", "software",
                     "visual review and assigned-device parity", "compare the retained pose assets against the pinned provider"),
    "artwork_compilation": (2, "maintained", "Inkmap, simulation and tatbot_ink", "software",
                            "measured end-to-end placement error", "compare compiled artwork with marks on a stationary plane fixture"),
    "inkmap": (3, "maintained", "browser placement and simulation bundle", "software",
               "body-to-measured-surface mapping", "retain one body placement for a measured phantom mapping experiment"),
    "stroke_lowering": (4, "maintained", "simulation factory references", "software",
                         "current measured inputs and native qualification", "execute the retained design on a supervised plane fixture and measure error"),
    "registration": (5, "experimental", "offline body diagnostics", "physical",
                     "instrument repeatability and independent held-out landmarks", "measure a stationary phantom mapping with uncertainty and held-out points"),
    "differentiable_patch": (2, "frozen", "tests and historical evidence only", "synthetic",
                             "named optimization consumer and measured baseline deficiency", "reopen only for a bounded optimization experiment against the exact compiler"),
    "proposal_training": (6, "frozen", "tests and historical evidence only", "synthetic",
                          "named runtime consumer and baseline deficiency", "reopen only for a measured consumer problem and preregistered comparison"),
    "phantom_mechanics": (7, "frozen", "tests and historical evidence only", "physical",
                          "qualified instrument and held-out non-human force/displacement data", "reopen only after rigid-model residuals justify a compliant fit"),
    "anatomy_mechanics": (8, "frozen", "tests and historical evidence only", "physical",
                         "measured coupling residuals or reviewed source objects and a consumer", "reopen only for a specific measured failure of the simpler model"),
}


def _capability_inventory(root: Path) -> dict[str, Any]:
    result = {}
    revision = git_output("rev-parse", "HEAD")
    clean_checkout = git_output("status", "--porcelain") == ""
    for name, (legacy, lifecycle, consumer, required_type, dependency, experiment) in CAPABILITIES.items():
        path = root / name / "manifest.json"
        historical = False
        if not path.is_file():
            path = root / f"phase-{legacy}" / "manifest.json"
            historical = True
        item = {
            "lifecycle": lifecycle, "consumer": consumer,
            "required_evidence_type": required_type,
            "unresolved_dependency": dependency, "next_experiment": experiment,
            "present": path.is_file(), "status": "missing", "current_revision": False,
            "evidence_type": "none", "hardware_authority": False,
        }
        if path.is_file():
            value = parse_json(path.read_bytes())
            if not isinstance(value, dict):
                raise ValueError(f"{path}: expected evidence object")
            if historical:
                if value.get("phase") != legacy:
                    raise ValueError(f"{path}: mismatched historical evidence")
            elif (value.get("schema") != "tatbot.capability-evidence/1"
                  or value.get("capability") != name
                  or value.get("evidence_type") not in {"software", "synthetic", "physical"}
                  or not isinstance(value.get("status"), str)):
                raise ValueError(f"{path}: expected named capability evidence and explicit evidence type/status")
            current = (not historical and clean_checkout and revision != "unknown"
                       and value.get("base_git_sha") == revision
                       and value.get("dirty_state") == "")
            item.update({
                "path": str(path), "sha256": sha256_file(path),
                "status": "historical" if historical else ("reported" if current else "stale"),
                "reported_status": value.get("status", "unknown"),
                "current_revision": current,
                "evidence_type": "historical-unclassified" if historical else value["evidence_type"],
                "known_gaps": value.get("known_gaps", []),
            })
        result[name] = item
    return result


def _write_review_markdown(
    path: Path, *, title: str, summary: str, capabilities: list[str],
    inventory: dict[str, Any], created_utc: str,
) -> None:
    lines = ["| Capability | Consumer | Evidence | Unresolved dependency | Next experiment |",
             "| --- | --- | --- | --- | --- |"]
    for name in capabilities:
        item = inventory[name]
        lines.append(
            f"| {name} ({item['lifecycle']}) | {item['consumer']} | "
            f"{item['evidence_type']}; {item['status']} | {item['unresolved_dependency']} | {item['next_experiment']} |"
        )
    path.write_text(
        f"# {title}\n\nGenerated {created_utc}.\n\n{summary}\n\n"
        + "\n".join(lines) + "\n\n"
        "Reported evidence is not acceptance. Historical packets do not establish current behavior. "
        "This report grants no deployment, public release, motion, emissions, human contact, or collection authority.\n",
        encoding="utf-8",
    )


def run_evidence(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evidence directory is not empty: {output}")
    (output / "approvals").mkdir(parents=True, exist_ok=True)
    gates = load_release_gates(args.gates)
    evidence_root = getattr(args, "evidence_root", None) or args.phase_root
    inventory = _capability_inventory(evidence_root.resolve())
    write_json(output / "capability-inventory.json", inventory)

    _write_review_markdown(
        output / "mhr-soma-inkmap-cutover.md",
        title="MHR/SOMA Inkmap cutover review",
        summary=(
            "The repository implementation has one nominal MHR-through-SOMA mid surface. "
            "The packet keeps maintainer visual review, explicitly assigned GPU parity, clean publication, and deployment distinct."
        ),
        capabilities=["source_contracts", "nominal_body", "inkmap"],
        inventory=inventory,
        created_utc=args.created_utc,
    )
    _write_review_markdown(
        output / "offline-full-pipeline.md",
        title="Offline full-pipeline review",
        summary=(
            "Typed design, placement, ink, registration, execution, and dense-sample contracts are reviewed independently "
            "from learned proposals and optional mechanics. Physical registration baselines remain mandatory for this claim."
        ),
        capabilities=["artwork_compilation", "stroke_lowering", "registration"],
        inventory=inventory,
        created_utc=args.created_utc,
    )

    _write_review_markdown(
        output / "capabilities.md", title="Human representation consumers and evidence",
        summary="Maintained consumers and frozen experiments; evidence is reported separately from acceptance.",
        capabilities=list(CAPABILITIES), inventory=inventory, created_utc=args.created_utc,
    )

    packet_paths = []
    for approval in APPROVALS:
        detail = PACKET_DETAILS[approval]
        gate = gates["approvals"][approval]
        packet = {
            "schema": "tatbot.human-representation-approval-packet/2",
            "approval": approval,
            "scope": detail["scope"],
            "status": gate["status"],
            "approved": gate["approved"],
            "external_action": detail["action"],
            "external_action_enabled": gates["actions"][detail["action"]],
            "required_capabilities": {name: inventory[name] for name in detail["required_capabilities"]},
            "required_review": detail["review"],
            "reviewer": gate["reviewer"],
            "reviewed_utc": gate["reviewed_utc"],
            "evidence_sha256": gate["evidence_sha256"],
            "generated_utc": args.created_utc,
        }
        path = output / "approvals" / f"{approval}.json"
        write_json(path, packet)
        packet_paths.append(path.relative_to(output).as_posix())

    write_json(output / "release-gates.snapshot.json", gates)
    write_json(output / "rollback.json", gates["rollback"])
    output_names = [
        "capability-inventory.json",
        "capabilities.md",
        "mhr-soma-inkmap-cutover.md",
        "offline-full-pipeline.md",
        "release-gates.snapshot.json",
        "rollback.json",
        *packet_paths,
    ]
    all_external_disabled = not any(gates["actions"].values())
    all_packets_present = all((output / name).is_file() for name in output_names)
    manifest = {
        "schema": "tatbot.capability-review/1",
        "status": "review_packets_generated",
        "created_utc": args.created_utc,
        "plan": None,
        "base_git_sha": git_output("rev-parse", "HEAD"),
        "dirty_state": git_output("status", "--short"),
        "remote_comparison": git_output("rev-list", "--left-right", "--count", "origin/main...HEAD"),
        "command": sys.argv,
        "runtime": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "device": "cpu",
            "gpu": "not_used_not_assigned",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "inputs": {
            "gates": {"path": str(args.gates), "sha256": sha256_file(args.gates)},
            "evidence_root": str(evidence_root.resolve()),
        },
        "outputs": {
            name: {"sha256": sha256_file(output / name), "size": (output / name).stat().st_size}
            for name in output_names
        },
        "accepted": {
            "review_packets": all_packets_present,
            "external_actions_fail_closed": all_external_disabled,
            "inkmap_deployment": False,
            "public_release": False,
            "powered_evaluation": False,
            "human_facing_study": False,
        },
        "known_gaps": [
            "Maintainer visual review and explicitly assigned GPU parity are pending for the nominal-body release claim.",
            "Qualified instrument and approved non-human phantom evidence are pending for the offline full-pipeline claim.",
            "Deployment, public release, powered evaluation, and human-facing study approvals are all pending and separate.",
        ],
        "next_dependency": "the named external reviewer and evidence for each independently scoped approval",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-root", "--phase-root", dest="evidence_root", type=Path, required=True,
                        help="named capability directories; legacy numbered packets are historical only")
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--created-utc", default=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_evidence(args)
    print(json.dumps(manifest, sort_keys=True))
    return (
        0
        if all((manifest["accepted"]["review_packets"], manifest["accepted"]["external_actions_fail_closed"]))
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
