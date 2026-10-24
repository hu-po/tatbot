# Human representation: consumers and evidence

MHR-through-SOMA supplies the nominal body Inkmap places designs on. A nominal
body is not a measured robot target: the robot's compile (`tatbot ros
compile`, `ros/README.md` section 4.2) accepts plane charts only, and body
placement stays out of scope until a measured body-to-surface address mapping
exists.

## Maintained core

| Capability | Consumer | Evidence type and limit | Unresolved dependency | Next experiment |
| --- | --- | --- | --- | --- |
| Pinned body assets and contracts | Body cache audit (`tatbot body bootstrap`, `tatbot body audit`), typed readers (`python/tatbot_surface`) | Software/cache integrity; no physical geometry acceptance | Reviewed bytes, visual review and device parity | Compare retained poses with the pinned provider |
| Nominal body and placement | Inkmap browser and its pose showcase | Browser/geometry checks; nominal only | Measured mapping to a physical target | Map one retained placement onto a fixed body-shaped phantom |
| Artwork and material compilation | Inkmap and `tatbot_ink` | Deterministic software tests | Measured placement and deposition error | Compare one retained design with marks on a stationary plane fixture |
| Release review | Inkmap deployment and public-export gates (`config/human-representation/release-gates.json`) | Explicit reviewer-bound approvals | Evidence for each independently scoped action | Review the exact candidate revision and its evidence |

The ManiSkill simulator carried the rest: the synthetic stroke expert, the
registration diagnostics, the evidence reporter, and the frozen
proposal-training, phantom-mechanics and anatomy experiments. They were
deleted with it on 2026-10-18; their packets are historical. Body placement
must not be converted to a plane or cylinder by discarding its address or
synthesizing a registration, and offline plans grant no hardware authority.

## The next physical result

One retained design on a measured stationary plane fixture, with independent
held-out landmarks and observed-versus-planned mark error, scale, shape and
uncertainty. Set acceptance limits from task requirements and instrument
repeatability before collection. A plane result does not qualify the body
bridge; that needs a separate body-shaped phantom mapping experiment.
