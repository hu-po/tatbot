# Public simulator test profile

These fixtures contain synthetic pigments, an empty freestanding six-slot
palette (the `urdf/palette.urdf` body at a synthetic scene pose), an analytic
arm seed, and a **nominal, unqualified** ballpoint workspace. Pigment IDs
match the checked-in example programs; they do not identify products or
stock. There is no measured touch-off, field pose bank, or hardware permission.

Run from the repository root with Python 3.12, uv, and official Node 22:

```sh
npm ci --prefix web/inkmap
test -f "${MS_ASSET_DIR:-$HOME/.maniskill}/data/robots/widowxai/wxai_follower.urdf" || \
  uv run --project python/tatbot_sim python -m mani_skill.utils.download_asset widowxai -y
uv run --project python/tatbot_sim --with pytest --with pytest-timeout \
  pytest -ra --timeout=120 python/tatbot_sim/tests --sim-profile public
```

Pytest copies code/assets and an allowlist of portable configuration into a
temporary root, applies these fixtures there, and redirects simulator imports
and child processes to it. Installed JavaScript dependencies are reused. The
original checkout's configuration is never overwritten. This also works in a
private checkout, without inheriting its measured workspace or inventories.

The portable run reports render-device, locked SOMA-cache, and field-evidence
tests as skips. Add `--sim-render` on a configured graphics host to run texture,
example-dataset, and GPU-root tests; the GPU test additionally requires CUDA.
The example dataset must retain nominal/unqualified provenance and its geometry
warning. A passing test run is not evidence of a measured or qualified tool.

Omit `--sim-profile public` to retain the normal deployment suite, including
field-calibration assertions. `scripts/check sim` chooses the public profile
when a deployment workspace or arm profile is absent, and otherwise keeps the
normal checkout behavior.
