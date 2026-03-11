# image-match

Match the exposure and color balance of images to a reference image.

Given a source/reference image and one or more destination images (or a
directory), `image-match` computes a unique per-image transformation so each
destination's exposure and color balance matches the reference. The goal is
visual consistency across a set — each image gets its own correction, not one
uniform transform.

## How It Works

Images are converted to **L\*a\*b\* color space**, which separates luminance (L)
from color (a\*=green-red, b\*=blue-yellow). Each channel is matched
independently, avoiding the cross-channel artifacts that RGB matching can
produce.

- **L channel (exposure):** mean/std linear transfer — corrects overall
  brightness while preserving the original tonal shape and highlight detail.
- **a\*/b\* channels (color balance):** histogram matching via quantized CDF —
  accurately aligns color distributions.

A **detail-preserving transfer** separates each channel into tone (low
frequency) and texture (high frequency), applies the correction to the tone
layer only, then recombines. This prevents the flattening of local contrast
that naive histogram matching can cause. The `--strength` flag lets you blend
between the original and matched result for finer control.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
git clone smbgit:stephanbrez/image-match.git
cd image-match
uv sync
```

## Usage

```
image-match SOURCE DEST [DEST ...] [-o DIR] [-s STR] [--strength F] [-j N] [-v]
```

| Argument            | Default       | Description                                    |
|---------------------|---------------|------------------------------------------------|
| `SOURCE`            | required      | Reference image to match against               |
| `DEST`              | required      | Image file(s) or directory to process          |
| `--output-dir -o`   | None          | Write outputs here (created if missing)        |
| `--suffix -s`       | `_matched`    | Suffix before extension (no `--output-dir`)    |
| `--strength`        | `1.0`         | Blend factor: 0.0 = no change, 1.0 = full match |
| `--jobs -j`         | CPU count     | Number of parallel workers                     |
| `--verbose -v`      | off           | Per-file progress to stderr                    |

### Examples

```bash
# Match a single image
uv run image-match reference.jpg target.jpg

# Match all images in a directory, write to an output folder
uv run image-match reference.jpg ./photos/ -o ./output -v

# Custom suffix instead of output directory
uv run image-match reference.jpg a.jpg b.jpg -s _corrected

# Dial back matching strength to preserve more of the original look
uv run image-match reference.jpg ./photos/ -o ./output --strength 0.7

# Use 4 parallel workers
uv run image-match reference.jpg ./photos/ -o ./output -j 4
```

### Output Naming

- **Without `--output-dir`:** `photo.jpg` → `photo_matched.jpg` (same directory)
- **With `--output-dir`:** `photo.jpg` → `OUTPUT_DIR/photo.jpg` (original name)

### Supported Formats

`.jpg` `.jpeg` `.png` `.tiff` `.tif` `.bmp` `.webp`

### Exit Codes

| Code | Meaning                              |
|------|--------------------------------------|
| 0    | All images processed successfully    |
| 1    | Fatal error (missing source, no valid destinations) |
| 2    | Partial failure (some images failed) |

## Dependencies

- [scikit-image](https://scikit-image.org/) — histogram matching and color space conversion
- [Pillow](https://python-pillow.org/) — image I/O
- [NumPy](https://numpy.org/) — array operations
