"""Core image matching logic using histogram matching in L*a*b* color space."""

import numpy as np
import pathlib

import PIL.Image
import skimage.color
import skimage.exposure


def load_image(path: pathlib.Path) -> np.ndarray:
    """Load an image from disk as a uint8 RGB NumPy array.

    Parameters
    ----------
    path : pathlib.Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image as a uint8 array with shape (H, W, 3).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be opened as an image.
    """
    if not path.is_file():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    img = PIL.Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def match_to_reference(
    image: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match the exposure and color balance of an image to a reference.

    Converts both images to L*a*b* color space and performs per-channel
    histogram matching. This independently aligns luminance (L),
    green-red (a*), and blue-yellow (b*) distributions.

    Parameters
    ----------
    image : np.ndarray
        Source image to transform (uint8 RGB).
    reference : np.ndarray
        Reference image to match against (uint8 RGB).

    Returns
    -------
    np.ndarray
        Matched image as uint8 RGB.
    """
    # ─── Convert to L*a*b* ───
    lab_image = skimage.color.rgb2lab(image)
    lab_reference = skimage.color.rgb2lab(reference)

    # ─── Per-channel histogram matching ───
    matched_lab = skimage.exposure.match_histograms(
        lab_image,
        lab_reference,
        channel_axis=-1,
    )

    # ─── Convert back to RGB uint8 ───
    matched_rgb = skimage.color.lab2rgb(matched_lab)
    return np.clip(matched_rgb * 255, 0, 255).astype(np.uint8)


def save_image(image: np.ndarray, path: pathlib.Path) -> None:
    """Save a uint8 RGB array to disk.

    Parameters
    ----------
    image : np.ndarray
        Image as a uint8 array with shape (H, W, 3).
    path : pathlib.Path
        Output file path. Parent directories must exist.
    """
    PIL.Image.fromarray(image).save(path)
