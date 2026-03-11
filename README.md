# image-match

Match the exposure and color balance of images to a reference image.

Given a source/reference image and one or more destination images (or a
directory), `image-match` computes a unique per-image transformation so each
destination's exposure and color balance matches the reference. The goal is
visual consistency across a set — each image gets its own CDF-based lookup
table, not one uniform transform.

## How It Works

Images are converted to **L\*a\*b\* color space**, which separates luminance (L)
from color (a\*=green-red, b\*=blue-yellow). Per-channel histogram matching via
scikit-image aligns each channel independently, handling both exposure and color
balance without the cross-channel artifacts that RGB matching can produce.

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
image-match SOURCE DEST [DEST ...] [-o DIR] [-s STR] [-v]
```

| Argument          | Default       | Description                                    |
|-------------------|---------------|------------------------------------------------|
| `SOURCE`          | required      | Reference image to match against               |
| `DEST`            | required      | Image file(s) or directory to process          |
| `--output-dir -o` | None          | Write outputs here (created if missing)        |
| `--suffix -s`     | `_matched`    | Suffix before extension (no `--output-dir`)    |
| `--verbose -v`    | off           | Per-file progress to stderr                    |

### Examples

```bash
# Match a single image
uv run image-match reference.jpg target.jpg

# Match all images in a directory, write to an output folder
uv run image-match reference.jpg ./photos/ -o ./output -v

# Custom suffix instead of output directory
uv run image-match reference.jpg a.jpg b.jpg -s _corrected
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
