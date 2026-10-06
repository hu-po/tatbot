# Inkmap

Inkmap is a static browser app for placing vector designs on a 3D body preview.
It writes a versioned placement JSON file and never commands a robot.

Inkmap consumes [InkLang](../../docs/inklang.md), the one system that turns a
placement description plus a named body into a face+barycentric anchor on that
body's canonical rest surface. Placement-only phrases are preferred. Missing
sides and other interactive ambiguities produce candidate buttons and block
placement or generation until someone chooses; legacy full tattoo sentences
remain compatible.

## Local development

```bash
npm ci
npm run check
npm run dev
```

Open `http://127.0.0.1:4180/?showcase=1` for the shared-artwork pose gallery. It replays generated scenario fixtures with the same renderer,
rig, decals, and compiled surface anchors used by the editor and simulator.
`npm test` regenerates missing showcase scenarios from the tracked manifest using
`python/tatbot_sim/.venv/bin/python`. Set up that environment with
`uv sync --project python/tatbot_sim` from the repository root first. The
pretest step preserves the manifest and its source provenance; generated
scenarios remain ignored build output. It establishes no reach or physical
qualification.

The app must run against local fixtures. Service endpoints and deployment
credentials are supplied by the deployment environment and are not committed.

`npm run check` runs typecheck, unit tests, the build, and the browser suite
twice (dev and production). Set `INKMAP_E2E_EVIDENCE=<dir>` to keep
screenshots and reports outside the repository.

The placement schema lives at `config/inkmap/placement.schema.json`; see the
public [design format](../../docs/design-format.md) for the artist-facing
contract.

The picker offers three genuine native DBV3 acquisitions: orbit, sprout and
ridges. Every ID binds acquired JSON, its source and a portable version 3
recipe. **Import artwork** accepts acquired `artwork.json`; source SVG/raster
images go through DBV3 first. Legacy saved drawings require regeneration.
The Generate panel explains the installed native acquisition path.

The body preview uses the shared SOMA rig and named tattoo-session poses. See [Inkmap documentation](../../docs/inkmap.md#named-body-poses) for
the source-of-truth files, regeneration command, and numerical gates.
