# Human representation maintenance scope

Keep artwork/placement/contracts/material compilation and release gates used by
Inkmap and the simulator (the Python session preparation was deleted 2026-09-26). Read `docs/human-representation.md` for consumers
and evidence limits before changing this package.

`training.py`, `torch_patch.py`, `mechanics.py`, `coupled_mechanics.py` and
`anatomy.py` are frozen experiments. Maintain existing behavior and tests, but do
not expand features, dependencies, runtime consumers or evidence campaigns.
Reopening requires a deliberate scope change naming a consumer, a measured
failure of the simpler baseline, an experiment and acceptance criteria. These
requirements are not a request to ask for approval on ordinary maintenance.

Do not restore a separate body-to-samples compiler. The robot draws plane
designs through `tatbot_ink` in `ros/`; simulation lowers retained designs
through the shared compiler and C++ planner checks. Body mapping is an
unresolved input boundary; nominal geometry cannot fill observed holes or
authorize execution. Preserve historical schema readers without presenting
them as active inputs.

Describe progress using consumer, evidence type, unresolved dependency and next
experiment. Numbered historical packets remain provenance, not readiness.
