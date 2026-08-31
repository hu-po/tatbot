# Contributing to tatbot

Thanks for your interest in tatbot! Contributions from developers and artists
are welcome.

## How this repository works

This repository is the **public release tree** of the tatbot project. Its
`main` branch advances through curated, reviewed exports from a private
development tree — not through direct merges. In practice:

- **Pull requests are welcome and reviewed here.** Accepted changes are ported
  into the development tree and land on `main` in the next export, with your
  authorship preserved via `Co-authored-by` credit. Your PR may therefore be
  closed as "landed via export" rather than merged with the merge button.
- **Issues and discussions are the fastest way to influence the project.**
  Open an issue before starting large changes so we can agree on direction
  first.

## What makes a good contribution

- Bug reports with a minimal reproduction.
- Fixes and improvements to simulation, replay, dataset tooling, schemas,
  docs, and web code — everything that runs without tatbot hardware.
- New hardware backends or profiles behind the documented extension points.

## Ground rules

- **Safety is not negotiable.** Code that can move hardware must require an
  explicit, complete hardware profile and fail closed when configuration is
  missing or ambiguous. PRs that weaken a safety gate, bypass an e-stop path,
  or turn a measured limit into a guessed default will not be accepted.
- Example and synthetic profiles must not be able to drive real hardware.
- Keep the tree buildable offline: no network access in tests, no vendored
  credentials, no hardcoded personal paths or hostnames.
- Match the style of the surrounding code; run the checks before pushing.

## Building and testing

Each component documents its own build in `docs/`. CI runs on every pull
request; a green check on your PR is the bar.

## Licensing

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).
