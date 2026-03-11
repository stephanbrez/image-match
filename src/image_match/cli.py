"""CLI layer for image-match: argparse setup, path resolution, batch loop."""

import argparse
import pathlib
import sys

import image_match.matching

SUPPORTED_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
}


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for the image-match CLI.
    """
    parser = argparse.ArgumentParser(
        prog="image-match",
        description=(
            "Match exposure and color balance of destination images"
            " to a reference image."
        ),
    )
    parser.add_argument(
        "source",
        type=pathlib.Path,
        help="Reference image to match against.",
    )
    parser.add_argument(
        "dest",
        nargs="+",
        type=pathlib.Path,
        help="Image file(s) or directory to process.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        default=None,
        help="Write outputs to this directory (created if missing).",
    )
    parser.add_argument(
        "--suffix",
        "-s",
        default="_matched",
        help='Suffix before extension when no --output-dir (default: "_matched").',
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-file progress to stderr.",
    )
    return parser


def resolve_images(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Expand directories and filter to supported image extensions.

    Parameters
    ----------
    paths : list[pathlib.Path]
        File paths and/or directories provided by the user.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of resolved image file paths.
    """
    images: list[pathlib.Path] = []
    for p in paths:
        if p.is_dir():
            images.extend(
                f
                for f in sorted(p.iterdir())
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(p)
    return images


def compute_output_path(
    path: pathlib.Path,
    output_dir: pathlib.Path | None,
    suffix: str,
) -> pathlib.Path:
    """Determine the output path for a matched image.

    Parameters
    ----------
    path : pathlib.Path
        Original image path.
    output_dir : pathlib.Path | None
        If set, outputs go here with original filenames.
    suffix : str
        Suffix inserted before extension when output_dir is None.

    Returns
    -------
    pathlib.Path
        The computed output file path.
    """
    if output_dir is not None:
        return output_dir / path.name
    return path.with_stem(path.stem + suffix)


def run() -> None:
    """Parse arguments and run the batch matching loop."""
    parser = build_parser()
    args = parser.parse_args()

    # ─── Validate source ───
    source_path: pathlib.Path = args.source.resolve()
    if not source_path.is_file():
        print(f"🚨 ERROR: Source image not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    # ─── Resolve destination images ───
    dest_images = resolve_images(
        [p.resolve() for p in args.dest],
    )
    if not dest_images:
        print("🚨 ERROR: No valid destination images found.", file=sys.stderr)
        sys.exit(1)

    # ─── Create output directory if needed ───
    output_dir: pathlib.Path | None = args.output_dir
    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Load reference image ───
    if args.verbose:
        print(f"📝 NOTE: Loading reference: {source_path}", file=sys.stderr)
    reference = image_match.matching.load_image(source_path)

    # ─── Batch loop ───
    failures = 0
    for img_path in dest_images:
        out_path = compute_output_path(img_path, output_dir, args.suffix)
        try:
            if args.verbose:
                print(f"🔍 Processing: {img_path}", file=sys.stderr)
            img = image_match.matching.load_image(img_path)
            matched = image_match.matching.match_to_reference(img, reference)
            image_match.matching.save_image(matched, out_path)
            if args.verbose:
                print(f"✅ SUCCESS: {out_path}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(
                f"⚠️ WARNING: Failed to process {img_path}: {exc}",
                file=sys.stderr,
            )

    if failures == len(dest_images):
        print("🚨 ERROR: All images failed to process.", file=sys.stderr)
        sys.exit(1)
    if failures > 0:
        print(
            f"⚠️ WARNING: {failures}/{len(dest_images)} images failed.",
            file=sys.stderr,
        )
        sys.exit(2)
