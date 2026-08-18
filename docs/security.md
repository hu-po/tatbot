---
summary: Public repository security and configuration policy
tags: [security, configuration]
---

# Security and configuration

This public repository contains the project archive and public-facing
documentation. Network addresses, hardware identifiers, shared-data paths, and
deployment configuration are represented with placeholders. The private
development repository is the source of truth for active deployment details.

Never commit credentials, API tokens, private keys, camera passwords, or
machine-specific identifiers. Use environment variables for secrets; the
tracked `.env.example` file contains placeholders only.

If a credential is ever exposed, rotate or revoke it first, then remove it
from the affected working tree and report the incident through GitHub's private
security channels.
