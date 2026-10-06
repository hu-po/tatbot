# Tatbot shared contracts

This Python 3.10+ package uses only the standard library. It owns strict JSON
parsing, canonical UTF-8 bytes, the MHR/SOMA model-spec contract, immutable cache
verification, and bootstrap/evidence file operations and digests. It also holds
the shared artwork envelope and the ordered metric path programs that drawing
preparation reads (`artwork.py`, `paths.py`), the follower's named observation
channels (`observations.py`), the ROS executable-program boundary
(`ros_program.py`), its ledger readers (`ros_progress.py`), a dipping tool's
replenishment profile (`replenishment.py`), the one-pen fitted resource
(`ros_fitted.py`, which never infers pigment from artwork RGB), and integer
sampling of a slower sensor on a controller clock (`timing.py`). It contains no model assets, network client,
deserializer, or robot control code.

The CLI initializes the source path once in `scripts/lib/tatbot_cli/__init__.py`, including
when invoked through `scripts/lib/tatbot_cli/__main__.py`. Bare-clone help, schema, and planning work
without an install. The simulator declares `tatbot-contracts==0.1.0` and uses the
sibling source through its uv configuration and lockfile during development.

For distribution, build both packages into one wheel directory and install the
simulator wheel with that directory available to the dependency resolver:

```sh
uv build --project python/tatbot_contracts --out-dir /tmp/tatbot-wheels
uv build --project python/tatbot_sim --wheel --out-dir /tmp/tatbot-wheels
uv pip install --find-links /tmp/tatbot-wheels /tmp/tatbot-wheels/tatbot_sim-0.1.0-py3-none-any.whl
```

The built wheel records a versioned dependency, with no checkout or relative file
URL. Install both wheels together when distributing offline. The simulator's
other runtime dependencies still apply. Full simulator imports also retain their
existing `TATBOT_REPO` requirement for the URDF, tool registry, and configuration
when installed outside an editable checkout; set it to the resource checkout,
independently of the working directory. The shared package has no such import
requirement. Public publication is a separate review.

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
