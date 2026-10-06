---
summary: Sanitized minimal and full replacement-cost bills of materials for Tatbot
tags: [hardware, bom, reference]
updated: 2026-09-29
audience: [dev, artist, contributor]
---

# Bills of materials

These are sanitized planning BOMs for the current Tatbot hardware class. They
answer “what is the minimum useful paper-development cell?” and “what replaces
the full current system?” without publishing private asset identities, network
topology, calibration, live state, or safety-acceptance evidence.

The 2026-09-03 replacement estimates are **about \$7.5k USD** for the minimal
cell and **about \$34.6k USD** for the full system, before tax, shipping, labor,
software/cloud services, facilities, and maintenance spares. Current matching
prices are used where possible; obsolete or unidentified SKUs use documented
successors or explicit allowances. These are planning costs, not historical
purchase prices.

The project now demos and develops against the **demo stack**
([Demo stack](demo-stack.md) lists its hardware): two WidowX AI arms, two
Jetson Thor computers, one wrist D405 per arm, one overhead D555, a
Raspberry Pi 5 palette station carrying the one e-stop button and the touch
probe, an unmanaged switch, a PoE source and a portable power station. It has
no PoE scene cameras, no dedicated viewer or camera computer, and no audio.
Both estimates below price the earlier five-camera cell as of 2026-09-03 and
have not been re-priced for the demo stack.

## Minimal BOM — $7,507.65

The minimal build supports guarded single-arm, non-needle ballpoint work on
paper with two wrist D405s and three PoE scene cameras. One D405, the arm
controller, standard gripper, and arm power supply are included in the follower
package.

| Group | Planning cost | What it covers |
| --- | ---: | --- |
| Tool arm | $4,995.95 | One [WidowX AI follower package](https://www.trossenrobotics.com/widowx-ai), including controller, power supply, gripper, one D405, and camera mount |
| Additional wrist sensing | $288.75 | One additional [RealSense D405](https://www.digikey.com/en/product-highlight/i/intel-realsense/d405-depth-camera) |
| Scene cameras | $203.97 | Three matching 5 MP PoE cameras |
| Scene-camera mounts and data cabling | $120.00 | Three mounts and Cat6 runs |
| Headless control computer | $1,499.00 | Arm control and sensor attachment |
| Ballpoint tool and second-camera mount | $169.98 | One custom tool/second-D405 mount, one rotary machine/battery kit, and one ballpoint-cartridge pack |
| Hardware E-stop | $45.00 | Latching stop assembly and controller |
| PoE network and USB attachment | $135.00 | TP-Link TL-SG1210MP (eight PoE+ ports), powered hub, and short cables |
| Basic power distribution | $50.00 | Power strip and cell leads |
| **Minimal BOM total** | **$7,507.65** | Single-arm paper-development and capture cell |

This assumes an existing operator computer/display, table, room uplink, and
ordinary paper supplies. It excludes the physical leader arm, two of the full
system's five scene cameras, reconstruction computer, dedicated viewer, audio,
backup batteries, tattoo-practice materials, and local training systems.

The PoE switch model is TP-Link TL-SG1210MP: eight gigabit PoE+ ports,
123 W power budget, two gigabit uplinks (one RJ45/SFP combo), and 16 KB
jumbo-frame support. See [manufacturer specifications](https://www.tp-link.com/us/business-networking/soho-switch-poe/tl-sg1210mp/v1/).
The model detail was updated on 2026-09-07; the original switch cost allowance
and BOM totals are unchanged. Each uplink remains limited to 1 Gbps.

## Full BOM — $34,620.54

The full build reproduces the current operating cell and dedicated local
training allocation. It includes the shared training battery at full
replacement cost; deduct $649.99 when that shared resource is already present.

### Operating-cell breakdown

| Group | Planning cost | What it covers |
| --- | ---: | --- |
| Robot pair | $9,681.90 | One [WidowX AI leader and one follower](https://www.trossenrobotics.com/widowx-ai), including controllers, power supplies, one D405, and standard end effectors |
| Additional wrist sensing | $288.75 | One additional [RealSense D405](https://www.digikey.com/en/product-highlight/i/intel-realsense/d405-depth-camera) |
| Scene cameras | $339.95 | Five matching 5 MP PoE cameras |
| Mechanical, fiducial, camera-mount, and data-cable allowances | $245.00 | Custom tool mount, printed fiducials, five camera mounts, and Cat6 |
| Cell compute, viewer, display, and connectivity | $6,167.00 | Control/vision computers, viewer computer and touchscreen, PoE/data switches, and USB attachment |
| Hardware E-stop and process audio | $259.00 | Latching E-stop assembly, USB audio interface, piezo pickup, and leads |
| Dedicated arm backup power | $849.99 | Portable power station with solar panel plus cell power distribution |
| Practice tooling and on-hand material families | $441.96 | Two rotary machines, ballpoint and needle cartridges, inks, caps, and three practice-skin forms |
| **Operating-cell subtotal** | **$18,273.55** | Complete current cell |

### Full reconciliation

| Boundary | Planning cost | Treatment |
| --- | ---: | --- |
| Operating-cell subtotal | $18,273.55 | Included |
| Dedicated local training hardware | $15,697.00 | Current training-system allocation; not required to operate the cell |
| Shared training power allocation | $649.99 | Portable power station also serving non-Tatbot loads |
| **Full BOM total** | **$34,620.54** | Full current-system replacement view |

Shared lab networking, storage, control-plane services, operator/development
computers, fabrication tools, and facilities remain excluded from both BOMs.

The current non-needle paper tool is an
[Inlumino ballpoint cartridge](https://inluminoheartink.com/products/10-ballpoint-pen-cartridges?variant=46567418921240)
in an [Ambition Lutin](https://www.ambition-tattoo.com/collections/ambition-rotary-tattoo-pen-series).
Consumable quantities are product-family placeholders, not a shelf-verified
stock count. The identified but unqualified handheld laser product is excluded
from this BOM and is not a procurement or use recommendation.

Prices change, so collect dated quotes before buying. The minimal BOM is not a
safety reduction: hardware presence and cost do not authorize powered use;
follow the repository's safety contracts and operator gates.
