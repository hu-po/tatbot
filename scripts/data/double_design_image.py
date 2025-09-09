#!/usr/bin/env python3
"""Generate horizontally concatenated image pairs from a directory.

This script scans an input directory for images and produces a specified number of
outputs, each being a horizontal concatenation of two randomly chosen images. A CSV
manifest is also written containing the UUID filename and the two source filenames.

Features:
- Reproducible randomness via ``--seed``
- Control output count via ``--num`` (defaults to number of found images)
- Optional recursive search with ``--recursive``
- Avoid pairing the same image unless ``--allow-same`` is provided
- No resizing: smaller image is padded to match the larger per-axis
  dimensions and kept centered vertically and horizontally

Example:
  # Uses defaults for input/output and writes pairs.csv manifest
  python scripts/data/double_design_image.py --num 500 --seed 42 --recursive

Note:
- If the output directory is within the input directory, generated images are
  excluded from the input set automatically.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
import csv
import uuid
from pathlib import Path
from typing import Iterable, Sequence


def _require_pillow() -> None:
    try:
        import PIL  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime guard
        print(
            "Pillow (PIL) is required. Install with: uv pip install pillow",
            file=sys.stderr,
        )
        raise


def list_image_files(
    directory: Path,
    *,
    recursive: bool = False,
    include_exts: Sequence[str] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".gif",
        ".tif",
        ".tiff",
    ),
    exclude_under: Iterable[Path] | None = None,
) -> list[Path]:
    """List image files in ``directory`` filtered by extensions.

    Args:
      directory: Directory to scan.
      recursive: If True, scan subdirectories.
      include_exts: File extensions to include (case-insensitive).
      exclude_under: Paths to treat as excluded roots (skip any files under them).

    Returns:
      Sorted list of image file paths.
    """
    directory = directory.resolve()
    include = {ext.lower() for ext in include_exts}
    excluded = [p.resolve() for p in (exclude_under or [])]

    def is_excluded(p: Path) -> bool:
        for root in excluded:
            try:
                if p.resolve().is_relative_to(root):
                    return True
            except AttributeError:
                # For Python <3.9 this would not exist; but we target 3.11
                pass
        return False

    it = directory.rglob("*") if recursive else directory.glob("*")
    files = [
        p
        for p in it
        if p.is_file()
        and p.suffix.lower() in include
        and not is_excluded(p)
    ]
    files.sort()
    return files


@dataclass(frozen=True)
class Pair:
    a: Path
    b: Path


def choose_pairs(
    files: Sequence[Path],
    *,
    count: int,
    rng: random.Random,
    allow_same: bool = False,
) -> list[Pair]:
    """Choose ``count`` pairs from ``files``.

    Args:
      files: Candidate image paths.
      count: Number of pairs to produce.
      rng: Random generator instance.
      allow_same: If True, allow a pair to contain the same image twice.

    Returns:
      List of ``Pair`` objects.
    """
    if not allow_same and len(files) < 2:
        raise ValueError("Need at least 2 images unless --allow-same is set")
    if allow_same and len(files) < 1:
        raise ValueError("No images found to pair")

    pairs: list[Pair] = []
    for _ in range(count):
        if allow_same:
            a = rng.choice(files)
            b = rng.choice(files)
        else:
            a, b = rng.sample(files, 2)
        pairs.append(Pair(a=a, b=b))
    return pairs


def concat_horizontal_no_resize(a_path: Path, b_path: Path):
    """Open two images and concatenate horizontally without resizing.

    When images differ in size, use the larger image's dimensions (per-axis)
    and pad the smaller image so that each side is centered both vertically
    and horizontally. The final output is two equally sized panels placed
    side-by-side on a white background (padding is white, not transparent).
    Returns a Pillow Image in RGBA mode with opaque white padding.
    """
    from PIL import Image, ImageOps

    with Image.open(a_path) as img_a, Image.open(b_path) as img_b:
        img_a = ImageOps.exif_transpose(img_a)
        img_b = ImageOps.exif_transpose(img_b)

        img_a = img_a.convert("RGBA")
        img_b = img_b.convert("RGBA")

        # Target panel dimensions are the per-axis maxima
        panel_w = max(img_a.width, img_b.width)
        panel_h = max(img_a.height, img_b.height)

        # Build centered panels for each image by padding around the content
        def make_centered_panel(img: Image.Image) -> Image.Image:
            # White, opaque background for padding
            panel = Image.new("RGBA", (panel_w, panel_h), (255, 255, 255, 255))
            off_x = (panel_w - img.width) // 2
            off_y = (panel_h - img.height) // 2
            # Paste using the image as mask to respect alpha
            panel.paste(img, (off_x, off_y), img)
            return panel

        panel_a = make_centered_panel(img_a)
        panel_b = make_centered_panel(img_b)

        out_w = panel_w * 2
        out_h = panel_h
        # White, opaque background for the final canvas as well
        canvas = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))
        canvas.paste(panel_a, (0, 0), panel_a)
        canvas.paste(panel_b, (panel_w, 0), panel_b)
        return canvas


def main(argv: Sequence[str] | None = None) -> int:
    _require_pillow()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=Path("/nfs/tatbot/designs/double/raw"),
        type=Path,
        help="Directory containing source images (default: /nfs/tatbot/designs/double/raw)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=Path("/nfs/tatbot/designs/double/combined"),
        type=Path,
        help="Directory to write concatenated images (default: /nfs/tatbot/designs/double/combined)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Number of output images to generate (default: number of input images)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when searching for images",
    )
    parser.add_argument(
        "--allow-same",
        action="store_true",
        help="Allow a pair to use the same image twice",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs (default: skip if file exists)",
    )
    # Output is always PNG
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_image_files(
        input_dir,
        recursive=args.recursive,
        exclude_under=[output_dir] if output_dir.resolve().is_relative_to(input_dir.resolve()) else [],
    )
    if not files:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 2

    count = args.num if args.num is not None else len(files)
    if count <= 0:
        print("--num must be positive", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    pairs = choose_pairs(files, count=count, rng=rng, allow_same=args.allow_same)

    from PIL import Image

    # Prepare CSV manifest
    csv_path = output_dir / "pairs.csv"
    csv_mode = "w" if args.overwrite or not csv_path.exists() else "a"
    write_header = csv_mode == "w"
    csv_file = open(csv_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["uuid", "filename1", "filename2"])

    saved = 0
    try:
        for i, pair in enumerate(pairs):
            uid = uuid.uuid4().hex
            out_name = f"{uid}.png"
            out_path = output_dir / out_name

            if out_path.exists() and not args.overwrite:
                # Skip existing to make the tool resumable without clobbering
                continue

            try:
                img = concat_horizontal_no_resize(pair.a, pair.b)

                # Always save PNG to preserve transparency
                img.save(out_path, format="PNG")
                saved += 1
                # Write CSV row with UUID (without extension) and original file names
                csv_writer.writerow([uid, pair.a.name, pair.b.name])
            except Exception as exc:  # pragma: no cover - runtime path
                print(f"Failed to process {pair.a} + {pair.b}: {exc}", file=sys.stderr)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{count} pairs...", file=sys.stderr)
    finally:
        try:
            csv_file.close()
        except Exception:
            pass

    print(f"Done. Saved {saved} images to {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
