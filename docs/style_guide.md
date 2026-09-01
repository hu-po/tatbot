---
summary: Public documentation style and review rules
tags: [docs, style]
updated: 2026-08-31
audience: [dev, contributor]
---

# Documentation style

Keep pages focused, current, and useful without private infrastructure.

## Page rules

- One H1 and short sections with descriptive H2 headings.
- Put prerequisites and a tested quick start near the top.
- Prefer stable paths, schemas, and commands over copied output.
- Label simulations, measurements, hypotheses, and release behavior separately.
- Use relative links and update the page date when behavior changes.

## Public review

Before publishing, check links, build with warnings as errors, and search for
private paths, hostnames, addresses, identifiers, credentials, business plans,
and human-use claims. If a page needs those details to be useful, keep it under
`internal/` and publish a conceptual summary instead.
