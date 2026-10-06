# Native multicolor fixture

Generated RGB contours in `source.png`, acquired with the recorded DrawingBotV3
Premium 1.6.22 jar and bridge through its normal batch SVG exporter. The fixture
contains no application binaries, activation data or robot measurements.

`fixture.json` contains the exact requested job, effective native settings,
software identity and successful normalized replay comparison. Native names
carry logical pen IDs; all five pens share a requested human label. Two black
pens have different widths, a red pen has a third width, and the disabled and
enabled zero-weight pens emit no paths. Export order is red, wide black, black,
different from the requested table's order. `normalized.svg` is the admitted
native transport used by the decoder test; paths are not hand-authored.
