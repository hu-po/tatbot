# Tatbot shared contracts

This Python 3.10+ package uses only the standard library. It owns strict JSON
parsing, canonical UTF-8 bytes, the MHR/SOMA model-spec contract, immutable cache
verification, and bootstrap/evidence file operations and digests. It also holds
the shared artwork envelope and the ordered metric path programs that drawing
preparation reads (`artwork.py`, `paths.py`), the ROS executable-program boundary
(`ros_program.py`), its ledger readers (`ros_progress.py`), a dipping tool's
replenishment profile (`dip.py`) and the one-pen fitted resource (`ros_fitted.py`,
which never infers pigment from artwork RGB). It contains no model assets,
network client, deserializer, or robot control code.

The CLI initializes the source path once in `scripts/lib/tatbot_cli/__init__.py`, including
when invoked through `scripts/lib/tatbot_cli/__main__.py`. Bare-clone help, schema, and planning work
without an install. The python/ packages (tatbot_surface and the sim 2.0
packages) declare `tatbot-contracts==0.1.0` and use the sibling
source through their uv configuration during development; a wheel built from
one of them records that versioned dependency, with no checkout or relative
file URL, so install both wheels together when distributing offline. The shared
package has no `TATBOT_REPO` import requirement. Public publication is a separate review.

`load_spec`/`validate_spec` verify structure and the declared hash.
`require_reviewed_digest` applies the production model pin: both the CLI commands
and the simulator spec loader call it. Synthetic tests can exercise structural
and cache validation with small assets without claiming a reviewed body.
`verify_cache` revalidates the spec and reads every allowlisted byte immediately
before the runtime constructs SOMA. Verification is repeated at bootstrap and
runtime boundaries; it does not cache a previous pass.

Context adapters retain their refusal interfaces. Shared validation combines
both prior validators' restrictions: distribution filenames must be relative,
geometry identities must match the reviewed pins, and cache paths must have no
symlink ancestors or Git repository ancestors. Cache files must be regular,
read-only, listed, correctly sized and hashed, and accompanied by canonical
metadata. Assets must remain protected from concurrent mutation after the
point-in-time check; verification does not replace filesystem ownership.
