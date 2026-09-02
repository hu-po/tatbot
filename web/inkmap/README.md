# Inkmap

Inkmap is a static browser app for placing vector designs on a 3D body preview.
It writes a versioned placement JSON file and never commands a robot.

## Local development

```bash
npm ci
npm run check
npm run dev
```

Open `http://127.0.0.1:4180/?showcase=1` for the guided procedural-body
milestone view. It replays checked-in scenario fixtures with the same renderer,
rig, decals, and compiled surface anchors used by the editor and simulator.

The app must run against local fixtures. Service endpoints and deployment
credentials are supplied by the deployment environment and are not committed.

The placement schema lives at `config/inkmap/placement.schema.json`; see the
public [design format](../../docs/design-format.md) for the artist-facing
contract.

The two body previews also carry a shared generated humanoid rig and named
poses. See [Inkmap documentation](../../docs/inkmap.md#named-body-poses) for
the source-of-truth files, regeneration command, and numerical gates.
