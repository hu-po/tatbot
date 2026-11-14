# Inkmap

Inkmap is a static browser app for placing tattoo art on a 3D body preview.
A design is its source image (PNG or JPEG), sized freely; the finished
placement exports as a tattoo brief that the art pipeline separates into robot
strokes at the chosen width. Inkmap never commands a robot.

Inkmap consumes [InkLang](../../docs/inklang.md), the one system that turns a
placement description plus a named body into a face+barycentric anchor on that
body's canonical rest surface. Placement-only phrases are preferred. Missing
sides and other interactive ambiguities produce candidate buttons and block
placement until someone chooses; legacy full tattoo sentences remain
compatible.

## Local development

```bash
npm ci
npm run check
npm run dev
```

Open `http://127.0.0.1:4180/?showcase=1` for the pose gallery.
It places stock art from the tracked `public/showcase/manifest.json`
on five named poses with the editor's own renderer, rig and decals. It
establishes no reach or physical qualification.

The app must run against local fixtures. Service endpoints and deployment
credentials are supplied by the deployment environment and are not committed.

`npm run check` runs typecheck, unit tests, the build, and the browser suite
twice (dev and production). Set `INKMAP_E2E_EVIDENCE=<dir>` to keep
screenshots and reports outside the repository.

The source-art, placement and brief schemas live in `config/inkmap/`; see the
public [design format](../../docs/design-format.md) for the artist-facing
contract.

The library offers three flash designs from `public/designs/`, each an image
with the inks it was separated into. **Import art** takes PNG and JPEG files,
with an optional `tatbot.art-generation/1` sidecar of the same name. Inkmap
holds no strokes: `tatbot art separate --brief` makes them from the brief and
gives back a picture of the predicted ink, which **Load render** shows on the
body.

The body preview uses the shared SOMA rig and named tattoo-session poses. See [Inkmap documentation](../../docs/inkmap.md#named-body-poses) for
the source-of-truth files, regeneration command, and numerical gates.
