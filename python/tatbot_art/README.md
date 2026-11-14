# tatbot-art

The art studio's library: what each ink looks like (`inks`), the colours a palette load can draw by tint
and two-ink hatching (`gamut`), and a predicted render of a `tatbot-program` on paper, silicone or skin
(`render`), and the separation of a flat-colour image into per-ink DBV3 layers and a program
(`separate`, `acquire`, `pipeline`), and the voting loop's scoring and proposals (`loop`), the wall's map
(`atlas`) and promotion pyramid (`pyramid`) behind the Flash Wall page (`gallery.html`). Run it as `tatbot art`; behaviour and limits are in `docs/art.md`.

`tatbot art profile PROGRAM --out DIR` explains a saved program's time as a terminal table and standalone
HTML/JSON report. It uses the recorded compiler components and explicit workflow allowances shared with
the render; no planning, arm or camera access.
