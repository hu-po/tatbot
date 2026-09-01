# Inkmap

Inkmap is a static browser app for placing vector designs on a 3D body preview.
It writes a versioned placement JSON file and never commands a robot.

## Local development

```bash
npm ci
npm run check
npm run dev
```

The app must run against local fixtures. Service endpoints and deployment
credentials are supplied by the deployment environment and are not committed.

The placement schema lives at `config/inkmap/placement.schema.json`; see the
public [design format](../../docs/design-format.md) for the artist-facing
contract.
