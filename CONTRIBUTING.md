# Contributing

Issues and pull requests are welcome. Open an issue before starting large
changes. Accepted changes may be merged by a maintainer rather than the merge
button; authorship is preserved via `Co-authored-by`.

## Ground rules

- **Safety is not negotiable.** Code that can move hardware must require an
  explicit, complete hardware profile and fail closed when configuration is
  missing or ambiguous. PRs that weaken a safety gate, bypass an e-stop path,
  or replace a measured limit with a guessed default will not be accepted.
- Example and synthetic profiles must not be able to drive real hardware.
- Keep the tree buildable offline: no network access in tests, no credentials,
  no hardcoded personal paths or hostnames.
- Match the style of the surrounding code. CI must be green.

## Licensing

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).
