"""`tatbot schema` — the internal JSON tree and public Markdown reference."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from tatbot_cli import EXIT_NAMES, __version__, nodes
from tatbot_cli.aliases import ALIASES, for_command
from tatbot_cli.metadata import operation_effects, requirements
from tatbot_cli.registry import NOUN_SUMMARY, TIER_GATES, TIER_MEANING, TIERS, all_verbs, nouns


def _private_node_re() -> re.Pattern:
    """Node names to redact from public Markdown, from config/nodes.json —
    the deployment's own config, so no fleet hostname is frozen into source.
    Matches nothing when no fleet is described."""
    from tatbot_cli.registry import repo_root

    names = sorted(nodes.load(repo_root()), key=len, reverse=True)
    if not names:
        return re.compile(r"(?!x)x")  # matches nothing
    alt = "|".join(re.escape(n) for n in names)
    return re.compile(rf"(?i)(?<![a-z0-9])({alt})(?![a-z0-9])")


_PRIVATE_NODE = _private_node_re()

def _public_text(value: str) -> str:
    """Remove deployment identities from generated public prose."""
    value = _PRIVATE_NODE.sub("configured host", value)
    value = re.sub(r"(?<![\w/])(?:~/|/)[^ )`]+", "<path>", value)
    value = re.sub(r"\bconfig/training/[^ )`]+", "<manifest>", value)
    value = re.sub(r"--tag(?:=|\s+)[^ )`]+", "--tag <label>", value)
    value = re.sub(r"(?i)server on configured host", "server on a configured host", value)
    value = re.sub(r"(?i)configured host / configured host", "configured training hosts", value)
    value = re.sub(r"(?i)configured host's", "configured host's", value)
    value = re.sub(r"(?i)fleet generator", "local generator", value)
    value = re.sub(r"(?i)what the Space serves", "the generated web bundle", value)
    value = re.sub(r"(?i)run and read trained policies on the arm", "analyze policy rollouts and recorded data", value)
    value = re.sub(r"(?i)the Pico e-stop, bench-checked or simulated", "simulate and inspect e-stop interfaces", value)
    value = re.sub(r"(?i)cameras, calibration, tracking, deploy", "offline vision processing and calibration", value)
    return value


def as_dict(repo: Path, schema_version: int = 2) -> dict:
    nmap = nodes.load(repo)
    data = {
        "version": __version__,
        "tiers": {t: {"meaning": TIER_MEANING[t], "gates": list(TIER_GATES[t])} for t in TIERS},
        "exit_codes": {str(k): v for k, v in EXIT_NAMES.items()},
        "nodes": nmap,
        "nouns": [{"noun": n, "summary": NOUN_SUMMARY.get(n, "")} for n in nouns()],
        "verbs": [{
            "name": v.name, "noun": v.noun, "verb": v.verb, "tier": v.tier, "summary": v.summary,
            "role": v.role, "role_from": v.role_for.describe if v.role_for else None,
            "nodes": nodes.nodes_with(nmap, v.role) if v.role else None,
            "gates": list(v.gates), "needs_tool": v.needs_tool, "launch_id": v.launch_id,
            "ink_hook": v.ink_hook,
            "passthrough": v.passthrough, "wraps": list(v.wraps), "doc": v.doc,
            "invariants": list(v.invariants),
            "example": f"tatbot {v.name} {shlex.join(v.example)}".strip(),
        } for v in all_verbs()],
    }
    if schema_version == 1:
        return data
    if schema_version != 2:
        raise ValueError("supported schema versions are 1 and 2")
    names = {v.name for v in all_verbs()}
    data.update(schema="tatbot.cli-schema/2", schema_version=2,
                aliases=[a.schema() for a in ALIASES if a.canonical_name in names])
    for row, v in zip(data["verbs"], all_verbs(), strict=True):
        row.update(canonical_name=v.canonical_name, deprecated=v.deprecated,
                   aliases=[a.schema() for a in for_command(v.name)], group=v.group,
                   visibility=v.visibility, audience=v.audience, disposition=v.disposition,
                   effects=operation_effects(v),
                   requirements=requirements(v), arguments=v.argument_spec.schema(),
                   conflict_groups=v.argument_spec.groups,
                   output={"result": v.output, "dry_run": "json", "explain": "json",
                           "modes": v.output_modes,
                           "errors": "CLI errors: JSON; backend exit codes/output preserved"},
                   passthrough_policy=("remainder" if v.passthrough and v.args is None else
                                       "explicit" if v.passthrough else "none"))
    return data



def as_disposition(repo: Path) -> str:
    """The whole-CLI command audit, generated so it cannot go stale.

    Every column but the reason is read from the command's own declarations:
    what it wraps, what it can do, where it runs. The reason is authored in
    tatbot_cli/audience.py, and `scripts/check cli` fails when a command has
    none — a placement nobody could justify is a placement nobody reviewed.
    """
    from tatbot_cli.audience import AUDIENCES
    from tatbot_cli.registry import all_verbs as verbs

    rows = list(verbs())
    out = ["# `tatbot` command disposition\n",
           "<!-- GENERATED by `scripts/tatbot schema --disposition`. Do not edit; "
           "`scripts/check cli` fails when stale. -->\n",
           f"Every one of the {len(rows)} commands this checkout provides, and who it is for. "
           "Ordinary `--help` shows the **primary** ones; `--help-all` shows all of them, and "
           "`tatbot schema --json` always has every one. Hiding a command from a help page is not "
           "removing it: removal needs evidence about callers, a migration and a release note, "
           "which this table does not by itself provide.\n",
           "See the [interface contract](cli-contract.md) for the grammar and "
           "[the full reference](cli-full.md) for arguments and examples.\n"]
    for audience in AUDIENCES:
        selected = [v for v in rows if v.audience == audience]
        if not selected:
            continue
        out.append(f"\n## {audience} ({len(selected)})\n")
        out.append("| command | tier | runs on | implemented by | effects | why here |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for v in selected:
            where = ((f"role `{v.role}`" + (", routed" if v.auto_hop else "")) if v.role
                     else v.role_for.describe if v.role_for else "any node")
            owner = ", ".join(f"`{w}`" for w in v.wraps) or "native"
            out.append(f"| `tatbot {v.name}` | `{v.tier}` | {where} | {owner} "
                       f"| {', '.join(sorted(v.effects))} | {v.disposition} |")
    out.append("\n## Compatibility retention\n")
    out.append("Commands marked **compatibility** reach the same handler as the canonical name in "
               "their reason column, so the two cannot drift. They remain for at least two reviewed "
               "releases and 30 days from the date they landed, whichever is longer; removal also "
               "requires consumer review and a release note.\n")
    return "\n".join(out)


def as_json(repo: Path, schema_version: int = 2) -> str:
    return json.dumps(as_dict(repo, schema_version), indent=2)


def as_markdown(repo: Path, full: bool = False) -> str:
    """Render the command reference.

    Public (default): a checked-in page, so it must not expose node inventory,
    SSH targets, local paths, private plans, or internal example arguments —
    only offline verbs of the public nouns. JSON remains the complete
    machine-readable schema.

    ``full``: the complete private table (every noun, tier, node and example,
    unscrubbed) for docs/cli-full.md — regenerated, never hand-edited.
    """
    d = as_dict(repo)
    scrub = (lambda v: v) if full else _public_text
    out = []
    if full:
        out.append("# `tatbot` command reference (complete)\n")
        out.append("<!-- GENERATED by `scripts/tatbot schema --md --full > docs/cli-full.md`. "
                   "Do not edit; `scripts/check cli` fails when stale. The public subset is docs/cli.md. -->\n")
        out.append("Every verb, tier, node and example. Current [interface contract](cli-contract.md); "
                   "[hardening and qualification](../plans/2026-09-05-cli-hardening.md).\n")
    else:
        out.append("# `tatbot` command reference\n")
        out.append("<!-- GENERATED by `scripts/tatbot schema --md > docs/cli.md`. Do not edit; `scripts/check cli` fails when stale. -->\n")
        out.append("Offline CLI workflows for public development and replay. Maintainers can inspect the complete machine-readable schema with `tatbot schema --json`.\n")
    out.append("```\ntatbot [global flags] <noun> [verb] [args] [-- passthrough args]\n```\n")
    if full:
        out.append("Global flags may appear anywhere before the first `--`, as `--flag value` or `--flag=value`, unless a declared command option owns that spelling: `--json`, `--dry-run`, `--on <node>`, `--ee-tool <id>`, `--explain`, `-q`, `-v`. Stating one twice with different values is a usage error.\n")
    else:
        out.append("Global flags may appear anywhere before the first `--`, as `--flag value` or `--flag=value`, unless a declared command option owns that spelling: `--json`, `--dry-run`, `--explain`, `-q`, `-v`. Stating one twice with different values is a usage error.\n")
    out.append("`--json` execution is supported only by commands declaring structured output in `schema --json`; "
               "all commands support JSON plans and explanations. Use `--output FILE` for draw report destinations. "
               "Backend flags on structured commands require `--`; tokens after it are preserved.\n")
    out.append("CLI-owned exit codes are below. Executed backends and SSH retain their own exit codes "
               "and error output; backend failures are not remapped to 1.\n")
    out.append("## Exit codes\n")
    out.append("| code | meaning |\n| --- | --- |")
    for k, v in d["exit_codes"].items():
        out.append(f"| {k} | {v} |")
    out.append("\n## Safety tiers\n")
    out.append("| tier | meaning | gates |\n| --- | --- | --- |")
    for t, rec in d["tiers"].items():
        if t != "offline" and not full:
            continue
        out.append(f"| `{t}` | {rec['meaning']} | {'; '.join(rec['gates']) or '—'} |")
    if full and d["nodes"]:
        out.append("\n## Nodes and roles\n")
        out.append("From `config/nodes.json` (targets are the canonical Tailscale addresses).\n")
        out.append("| node | ssh | arch | roles |\n| --- | --- | --- | --- |")
        for name, rec in d["nodes"].items():
            out.append(f"| `{name}` | `{rec.get('ssh', '')}` | {rec.get('arch', '')} | {', '.join(rec.get('roles', []))} |")
    out.append("\n## Verbs\n")
    cur = None

    def shown(v):
        return full or v["visibility"] == "public"

    for v in d["verbs"]:
        if not shown(v):
            continue
        if v["noun"] != cur:
            cur = v["noun"]
            summ = scrub(next(x["summary"] for x in d["nouns"] if x["noun"] == cur))
            suffix = f" — {summ}" if summ else ""
            out.append(f"\n### `tatbot {cur}`{suffix}\n")
            out.append("| verb | tier | runs on | wraps | summary |\n| --- | --- | --- | --- | --- |")
        where = (", ".join(v["nodes"]) if v["nodes"] else v["role_from"] or "any") if full else "this checkout"
        wraps = ", ".join(f"`{w}`" for w in v["wraps"]) or "native"
        name = f"`{v['name']}`"
        extras = []
        if v["needs_tool"]:
            extras.append("`--ee-tool`")
        if v["launch_id"]:
            extras.append("launch id, `--tag` labels it")
        if v["ink_hook"]:
            extras.append("`--no-ink`")
        if v["passthrough"]:
            extras.append(f"`--` → {v['passthrough']}")
        summ = scrub(v["summary"] + (f" ({'; '.join(extras)})" if extras else ""))
        out.append(f"| {name} | `{v['tier']}` | {where} | {wraps} | {summ} |")
    out.append("\n## Compatibility spellings\n")
    out.append("These deprecated forms translate to the same canonical handler. They warn once on stderr. "
               "They remain for at least 30 days and two reviewed releases after 2026-09-05, whichever is longer; "
               "removal also requires consumer review and a release note.\n")
    out.append("| deprecated form | canonical form |\n| --- | --- |")
    shown_names = {v["name"] for v in d["verbs"] if shown(v)}
    for alias in d["aliases"]:
        if alias["canonical_name"] in shown_names:
            out.append(f"| `tatbot {alias['source']}` | `tatbot {alias['canonical_form']}` |")
    out.append("\n## Bash completion\n")
    out.append("Generate and source completion explicitly (Bash 4+):\n")
    out.append("```sh\ntatbot completion bash > /tmp/tatbot-completion.bash\nsource /tmp/tatbot-completion.bash\n```\n")
    out.append("Generation prints shell code; it does not install anything or edit shell startup files. "
               "Completion is generated from the canonical commands, options and choices.\n")
    out.append("\n## Examples\n")
    out.append("```")
    for v in d["verbs"]:
        if not shown(v):
            continue
        if full:
            out.append(v["example"])
            continue
        # Examples are deliberately schematic: node names, tool IDs, paths,
        # and inventory identifiers are deployment-specific/private.
        args = []
        prefix_len = 1 + len(v["name"].split())
        for arg in shlex.split(v["example"])[prefix_len:]:
            if _PRIVATE_NODE.fullmatch(arg):
                args.append("<configured-node>")
            elif "tool" in arg or (repo / "config" / "tools" / f"{arg}.yaml").is_file():
                args.append("<tool-id>")
            elif arg.startswith("inkcap_") or arg.startswith("nighthawk_"):
                args.append("<inventory-id>")
            elif arg.startswith(("~/", "/", "<")):
                args.append("<path>")
            elif re.match(r"^\d{8}T\d{6}Z-", arg) or "run-id" in arg or _PRIVATE_NODE.search(arg):
                args.append("<run-id>")
            else:
                args.append(_public_text(arg))
        out.append(f"tatbot {v['name']} {shlex.join(args)}".strip())
    out.append("```\n")
    return "\n".join(out)
