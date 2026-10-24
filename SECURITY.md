# Security Policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/hu-po/tatbot/security/advisories/new),
not in a public issue, pull request or discussion. Include the affected path
or command, the release or commit, and how to reproduce it.

Treat as security issues anything that could move an arm unexpectedly or
weaken the e-stop path, the controller limits, the carriage contact cap, the
trip retract or exclusive driver ownership, as well as the usual software
classes: leaked credentials, code execution, unauthenticated network control,
and compromised dependencies or workflows.

## Supported versions

Only the latest release receives fixes. This repository is published from a
private development repository, so a fix lands in the next release rather than
as a merged pull request here.

## Dependencies

The Rust workspace, the LeRobot plugin, the Inkmap editor and the docs build
install from lockfiles, and every GitHub Action in this repository's workflows
is pinned to a full commit SHA. Updates adopt releases that have been public
for at least seven days unless they fix a vulnerability. `scripts/deps_audit.py`
checks every locked and pinned version against [OSV](https://osv.dev) in CI
each week; an advisory we accept is listed in `config/deps-audit-allow.tsv`
with the reason it does not apply and a date to review it again.
