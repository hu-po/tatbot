# DBV3 paired drawing research

`tatbot research` freezes candidates, prepares both sides, and calls the same
`tatbot ros draw` used by an operator. It has no compiler selector or legacy
backend. The ROS ledger remains the execution authority. Install `uv`; commands
resolve the [locked drawing environment](drawingbot.md#prepare-and-review)
before running, including Python, NumPy and PyYAML. ROS image inspection uses
OpenCV in the deployed ROS environment. The CLI grammar and dry runs remain
available with stdlib Python.

## Frozen inputs

Create a source manifest with schema `tatbot.dbv3-corpus/1` and a `cases` array.
Every case has `id`, `family`, `split` (`train`, `validation`, or `test`), `file`
(relative to the manifest), `sha256`, `size_mm: [width, height]`, and source
`provenance`. Keep every seed and derivative of one source family in the same
split. All three splits are required. Test sources are excluded from routine
candidate creation and preparation.

```sh
tatbot research init /path/corpus.json --out /path/study
```

The study copies and verifies source bytes. Its default layout has four rows
of two 26 × 26 mm slots inside the stencil's 62 × 112 mm clear region.
Pass `--layout /path/layout.json` to specify different slot dimensions and centres:

```json
{"slot_mm": [24, 32], "rows_mm": [36, 0, -36], "columns_mm": [-15, 15]}
```

Coordinates are millimetres from the page centre: x right, y toward the top of
the print. Columns must name two centres from left to right; row indices follow
the supplied order. Every rectangular slot must fit the clear region, with a
positive gap between rows and columns. Touching or overlapping slots are refused
before creating a study or reserving a trial. The frozen study records metres;
changing the layout requires a new study and candidates. Existing default-layout
studies keep their identities.

Both canvas dimensions and the deposited stroke footprint must fit a slot.
A larger drawing requires a new acquisition; preparation never scales it.

Acquire each train and validation case with `tatbot drawingbot generate` at
its stated size. Create an acquisitions JSON object mapping case IDs to
completed acquisition directories. All cases in a candidate use one recipe
policy, including the same DBV3 application, decoder and generation runtime.
Source identity and physical size come from each corpus case.

```sh
tatbot research candidate /path/study /path/acquisitions.json --out /path/baseline --speed 3.5
```

Use the standard `--ee-tool ID` global or `TATBOT_EE_TOOL` to select the
candidate's physical tool; otherwise preparation reads the configured fitted
right-arm tool. An explicit tool on `research run` must match both frozen
programs. It cannot alter a prepared trial's tool at dispatch.

The candidate copies acquired artwork and native replay bundles, verifies
their hashes, and freezes preparation code, tool, motion and scorer identities.
It is immutable. A motion candidate can reuse the same acquisitions with a
new speed or maximum chunk duration. A recipe candidate requires new acquired
output. The first implementation admits one changed policy leaf per pair;
PFM-family changes with several different parameter leaves need a separately
designed study, not a falsely labelled one-factor trial.

## Review training acquisitions in Inkmap

Review can precede candidate creation. Each acquisitions JSON maps source case
IDs to completed acquisition directories, using the same format as `candidate`,
but covering only cases in one selected split. Supply several maps to compare
recipe variants. Default admission is training only; `--split validation` is
explicit, and held-out test sources are never admitted. Validation acquisitions
are not required merely to preview training alternatives.

```sh
tatbot research review /path/study /path/recipe-one.json /path/recipe-two.json \
  --title 'Line density study' --speed 3.5 --out /path/review
```

This uses production ROS preparation at the canvas centre. It writes frozen
artwork, exact prepared programs and `review.json`, and registers the bundle in
`STUDY/reviews/`. Tool selection uses the standard `--ee-tool` global. A review
is not a reserved physical slot or a prepared trial; later placement produces
its own program identity. Up to 64 entries and 20 MiB fit one review bundle.

Open `review.json` using **File → Review study…** in Inkmap. Cards show acquired
paths at generation width, physical dimensions, tool-width evidence, path/chunk
counts, contact length and the production duration estimate. Estimate scope and
unmeasured costs remain visible. **Add to library** uses the existing artwork
import and leaves the current placement draft intact.

Like, Dislike and Clear append observations locally in the browser. Reloading
and reopening the same bundle restores them; other tabs merge records without
overwriting them. Save failures keep votes in memory and offer retry and a
feedback download. Download feedback before clearing browser data or moving
to another browser. No preference or bundle is sent to a public worker.

```sh
tatbot research feedback /path/study /path/inkmap-feedback.json
```

Import verifies the registered bundle, study, artwork and preparation identities,
validates each observation and preserves immutable receipts. Re-importing the
same file is idempotent; conflicting observation IDs are refused. The client
records its renderer version and the hash of the actual generated SVG. These
are human review records, not independently attested rendering or physical
execution evidence. Old votes do not transfer to regenerated artwork. Multiple
observations remain in history, including cleared votes; importing them does
not automatically promote a candidate or change a research baseline.

## Prepare, establish the page, execute

```sh
tatbot research prepare /path/study --a /path/baseline --b /path/baseline \
  --case flower --page sheet-001 --row 0 --kind aa \
  --hypothesis 'Measure repeatability with identical DBV3 artwork and motion'
```

Preparation is the compile-only operation. It writes both `program.json`
files, previews, candidate snapshots and an immutable trial manifest before
reserving both slots. Physical sides alternate each iteration; execution
order alternates every two iterations. A failed preparation keeps its
artifacts and journal event. Retrying creates a new trial directory.

A page identifier denotes one physical sheet. Record the concrete observation
establishing a fresh sheet after preparing its first pair:

```sh
tatbot research page /path/study sheet-001 \
  --observation 'Operator replaced the completed print with a fresh blank sheet'
tatbot research run /path/study 0000
```

This observation must describe an actual placement. Tracking loss/reacquisition
does not establish a page change. Confirmation never clears occupied slots.
The study lock prevents overlapping iterations; the execution owner separately
claims each physical slot before drawing. A reserved, partial, completed,
unreadable or unresolved claim remains occupied. Other workflows drawing into
these slots invalidate the experiment; the index is not a paper sensor.

`run` reconciles each side with `tatbot ros evidence --page ID --slot ROW:COL`.
It reads the existing program, ledger, motion configuration, runtime and latest
inspection. A new, unclaimed pair calls `tatbot ros ready` before freezing its
runtime, so the ordinary wake-up of a landed arm happens before the pair starts.
Both sides then use `ros draw --no-wake`: the first holds at rest after inspection,
and the second retains ordinary inspection and landing. Before each side the runner
waits, between attempts, for the cameras to measure the page again (`page_awaited`
in the journal), so time the arm spent hiding the print is not counted as drawing. The order is the frozen
trial order, including trials that execute B first. The pair is marked drawn
only after ROS reports the final landing; a failed or uncertain landing needs
reconciliation, never another drawing. Startup attempts retain their own logs.

A lost client cannot redraw a completed side. An interrupted run
with a known abort can continue via `research run ... --resume`, using the same
ROS run and ledger. A crash-sent operation, still-running run, missing inspection
or changed runtime stops the iteration for reconciliation. No retry deletes
artifacts or treats uncertainty as blank paper. Existing ROS cancellation,
e-stop, landing, controller limits and driver ownership remain unchanged.

Runtime identity includes the session's imported Python sources and the exact
controller process identified by the launcher, with boot/start identity and
hashes of its file-backed executable mappings (including loaded controller and
hardware plugins). Missing, replaced or unidentifiable loaded files refuse a
research comparison before slot admission. Readback checks the current mappings
again; it does not substitute newly deployed files for an older loaded inode.
This is runtime provenance, not an attestation of process memory.

An actual ROS process restart or loaded-code change makes an unfinished pair
inconclusive; its slots remain used. Ordinary drawing retains its admission
rules when this research evidence is unavailable.
Completed pairs from different operating windows may be compared when the
Python source digest, Python/NumPy versions, loaded native file paths/hashes,
launch configuration, robot description and controller settings match. The
launcher snapshots the controller settings it actually supplies. Run-log paths
and the runtime record's location are excluded from the launch-input digest;
geometry, limits and settings are included. Tool, registration and draw execution context must also agree.
Process IDs, boot/start IDs, file inodes and unrelated checkout revisions
remain recorded evidence, but do not make identical software different. Full
runtime identity must still match within each pair. A recorded completed pair
remains readable after its runtime stops.
This first version supports right-arm, single-pen trials. Explicit multi-pen
preparation is available through the ordinary ROS workflow, not this study
runner. Cartridge dipping is not implemented.

## Score and decide

```sh
tatbot research score /path/study 0000
tatbot research decide /path/study qualify 0000 0001 \
  --reason 'Two DBV3 A/A pairs inspected; repeatability accepted'
```

A baseline starts unqualified. Qualification requires two scored A/A pairs on
one exact candidate, real-hardware ink evidence and a positive measured
drawing-session duration for every side. Mock runs only qualify software.
Qualification and promotion both require complete timing, including the A/A
noise pairs. A result with missing timing can still be retained as inconclusive.
Ordinary pairs must use that qualified candidate as A.
For example, a speed trial declares `--kind paired --factor preparation.speed_m_s`.
Candidate promotion requires two A/A noise pairs, two training pairs and at
least one held-out validation pair. Both sides must share the actual deployed
runtime and scorer. The gain must exceed both the declared minimum and the
largest A/A difference on every repeat; missing-ink or spill regressions beyond
A/A spread prevent automatic promotion.

```sh
tatbot research decide /path/study promote 0000 0001 0002 0003 0004 \
  --metric draw_session_s --max-time-ratio 1.0 \
  --reason 'Faster traversal reduced measured time beyond A/A variation without fidelity loss'
```

The decision is an immutable write-ahead record; replaying the command repairs
an interrupted baseline update without undoing a later promotion. Failed or
unfinished trials can be recorded with `decide ... inconclusive` and a reason,
even without a score. The event journal retains failures and attempts. The
scheduler should invoke one prepared iteration at a time, then score and record
a decision before choosing the next hypothesis; it must stop when a new physical
sheet or unresolved ROS state requires operator input.

Inspection first registers the printed page border, then measures only the
reserved slot. It reports placement separately from aligned missing ink,
unwanted ink, overlap and line-width diagnostics. Ink is judged against the
local paper brightness, so the arm's shadow is not ink; a slot whose paper is
too dark to separate from ink (deep shadow or a smear wider than 5 mm) is
refused. Alignment searches the arm's measured placement spread (8 mm), and
missing/unwanted ink use a 0.5 mm tolerance, one native wrist pixel plus the
border registration sigma. Registration failure, ambiguous alignment, a
refused view and an incompletely observed slot are inconclusive. The
width diagnostic is a distance-transform proxy with a stated pixel resolution,
not a calibrated tool-width measurement. Images and overlays remain in the
ROS inspection directory; receipts preserve their hashes. No before-image
subtraction is claimed: recorded fresh-page placement and exclusive slot use
are required to rule out prior marks inside a slot.

Cost fields distinguish the planned motion estimate, ROS drawing-session time,
and client wall time (which includes inspection and landing). The ROS session
records monotonic start/end timing for every draw-action attempt in its existing
event log. `draw_session_s` sums completed attempts, including setup, planning,
execution, in-action waits and bag shutdown. It excludes idle time between
attempts, inspection and landing. Missing, unfinished or damaged timing evidence
makes the full duration unknown; the last attempt's `meta.duration_s` is never
substituted. Old runs without these events also have unknown research timing.
Measured pen-down time remains null until separately instrumented; the
runner never substitutes metadata, planned length/speed or planned phase timestamps.
Artistic preference remains unset unless a human supplies it.
These measurements are research criteria, not new motion safety interlocks.
