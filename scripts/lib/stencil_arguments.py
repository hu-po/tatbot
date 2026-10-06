"""Stdlib argument contract shared by CLI planning and the observer backend."""

from pathlib import Path


def common(parser):
    parser.add_argument("--reference", action="append", required=True, help="tracking.json; repeat for competing known patterns")
    parser.add_argument("--instance", required=True, help="instance label; namespace when tracking all references")
    parser.add_argument("--output", required=True, help="new evidence directory outside the repository")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--all-references", action="store_true", help="track one independent instance of every supplied pattern")
    parser.add_argument("--surface", action="store_true", help="estimate advisory RGB-D surface or explicit nominal-print RGB pose")
    parser.add_argument("--rerun", action="store_true", help="save capped RGB-D and stencil geometry to output/replay.rrd")
    parser.add_argument("--connect", help="stream to the existing fleet Rerun proxy; never starts a viewer")
    parser.add_argument("--recording-id", help="Rerun recording for this single camera; defaults to the run id")
    parser.add_argument("--region", type=pixel_box, metavar="X0,Y0,X1,Y1",
                        help="decode coded prints only in this pixel box (a large overhead frame); "
                             "default the whole image")


def pixel_box(text):
    """'X0,Y0,X1,Y1' pixels -> a box with positive width and height."""
    try:
        box = tuple(int(value) for value in text.split(","))
    except ValueError:
        raise ValueError(f"expected X0,Y0,X1,Y1 in pixels, got {text!r}") from None
    if len(box) != 4 or min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"expected X0,Y0,X1,Y1 with X1 > X0 and Y1 > Y0, got {text!r}")
    return box


def replay(parser):
    common(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames", help="JSONL image manifest: image, timestamp_ns, optional depth_m NPY")
    source.add_argument("--recording", help="one sensor's visiond frames.jsonl")
    parser.add_argument("--truth", help="separate per-frame evaluation labels; never supplied to tracker")


def observe(parser):
    common(parser)
    parser.add_argument("--socket", required=True, help="existing visiond frame-owner Unix socket")
    parser.add_argument("--sensor", required=True, help="exact RGB sensor name")
    parser.add_argument("--duration-s", type=float, default=60.)


def validate(args, repo):
    if not 1 <= len(args.reference) <= 8 or not 1 <= args.max_frames <= 10000:
        raise ValueError("require 1–8 references and 1–10000 frames")
    if not args.instance or len(args.instance) > 160:
        raise ValueError("require a nonempty instance label of at most 160 characters")
    if args.all_references and len(args.instance) > 80:
        raise ValueError("multi-stencil instance namespace must be at most 80 characters")
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_relative_to(Path(repo).resolve()):
        raise ValueError("output must be a new directory outside the repository")
    if hasattr(args, "duration_s") and not 0 < args.duration_s <= 3600:
        raise ValueError("duration must be finite and within 0–3600 seconds")
    return output
