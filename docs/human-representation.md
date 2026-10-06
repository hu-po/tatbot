# Human representation: consumers and evidence

MHR-through-SOMA supplies the nominal body used by Inkmap and simulation. A
nominal body is not a measured robot target: the robot's compile
(`tatbot ros compile`, `ros/README.md` section 4.2) accepts plane charts only,
and body placement stays out of scope until a measured body-to-surface address
mapping exists.

## Maintained core and frozen experiments

| Capability | Consumer | Evidence type and limit | Unresolved dependency | Next experiment |
| --- | --- | --- | --- | --- |
| Pinned body assets and contracts | Body cache audit, asset exporter, typed readers | Software/cache integrity; no physical geometry acceptance | Reviewed bytes, visual review and device parity | Compare retained poses with the pinned provider |
| Nominal body and placement | Inkmap browser and simulation bundles | Browser/geometry checks; nominal only | Measured mapping to a physical target | Map one retained placement onto a fixed body-shaped phantom |
| Artwork and material compilation | Inkmap, simulation and `tatbot_ink` | Deterministic software and synthetic coverage tests | Measured placement and deposition error | Compare one retained design with marks on a stationary plane fixture |
| Execution lowering | Simulation factory references | Shared Cartesian compiler, C++ joint planner and FK evidence | Current measured inputs and existing native qualification | Measure placement, scale and trajectory error on that fixture |
| Registration diagnostics | Offline body diagnostics | Synthetic diagnostics or explicitly supplied measured observations; unqualified | Instrument repeatability, independent held-out landmarks and a body mapping reader | Measure phantom registration error and uncertainty |
| Proposal training and differentiable patch optimization — frozen | Tests and historical evidence only | Synthetic proposal/gradient checks | Named consumer and measured baseline deficiency | Reopen only for a specific consumer problem and comparison |
| Phantom mechanics — frozen | Tests and historical evidence only | Synthetic fits; no accepted physical model | Qualified instrument and held-out force/displacement measurements | Reopen only when measured rigid-model residuals justify compliance |
| Coupled mechanics and anatomy — frozen | Tests and historical evidence only | Synthetic interfaces; no admitted anatomy or coupled model | Measured coupling failure or reviewed source objects and a consumer | Reopen only for a measured failure of the simpler model |
| Release review | Export/deployment gates and review packets | Explicit reviewer-bound approvals; report generation grants none | Evidence for each independently scoped action | Review the exact candidate revision and its evidence |

Frozen means maintenance and reproduction only: no new features, runtime
consumers, dependencies, dataset campaigns or recurring evidence jobs. A deliberate
scope change must identify the consumer, measured baseline deficiency, experiment
and acceptance criteria. Existing tests remain; a checked import boundary prevents
maintained runtime code from acquiring a frozen research dependency.
`scripts/check body-single-path` runs that stdlib check even without the simulator.

## One retained-design lowering path

Simulation lowers a retained design through `human_rep.ink_program`,
`pen_path` and the C++ `path_plan_check` planner. The separate body
`ExecutionProgram` sample compiler and its standalone evidence generator have
been retired; their schema and example records remain readable for historical
inspection only. Body placement must not be converted to a plane or cylinder by
discarding its address or synthesizing a registration. Offline plans grant no
hardware authority.

## Evidence reporting

Report consumer, evidence type, unresolved dependency and next experiment.
`review_evidence --evidence-root` reads named capability directories and writes
`capability-inventory.json` and independent approval packets. Each new input
`manifest.json` uses `tatbot.capability-evidence/1`, names its `capability`,
`evidence_type` (`software`, `synthetic` or `physical`), `status`, `base_git_sha`
and `dirty_state`. Only an exact clean revision match is reported as current;
a reported result still does not establish acceptance. Missing evidence is
missing, and old numbered packets are historical regardless of their pass label.
The body audit and synthetic registration suite emit named capability manifests.
The legacy root flag remains an alias for reading those immutable records.

The next physical result is one retained design on a measured stationary plane
fixture, with independent held-out landmarks and observed-versus-planned mark
error, scale, shape and uncertainty. Set acceptance limits from task requirements
and instrument repeatability before collection. A plane result does not qualify
the body bridge; that needs a separate body-shaped phantom mapping experiment.
