"""Core image matching logic using mean/std transfer in L*a*b* color space."""

import warnings

import numpy as np
import pathlib

import PIL.Image
import scipy.ndimage
import skimage.color


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


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 RGB image to L*a*b* using float32 precision.

    Parameters
    ----------
    image : np.ndarray
        Image as a uint8 array with shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Image in L*a*b* color space as float32.
    """
    float_image = image.astype(np.float32) / 255.0
    return skimage.color.rgb2lab(float_image).astype(np.float32)


def _mean_std_transfer(
    source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Transfer the mean and standard deviation from reference to source.

    Applies a linear transform that shifts and scales the source distribution
    to match the reference's first two moments, preserving the original
    distribution shape.

    Parameters
    ----------
    source : np.ndarray
        Source channel (2D float32).
    reference : np.ndarray
        Reference channel (2D float32).

    Returns
    -------
    np.ndarray
        Transferred channel as float32.
    """
    src_mean, src_std = source.mean(), source.std()
    ref_mean, ref_std = reference.mean(), reference.std()

    if src_std > 0:
        return (source - src_mean) * (ref_std / src_std) + ref_mean
    return source + (ref_mean - src_mean)


def match_lab(
    source_lab: np.ndarray,
    reference_lab: np.ndarray,
) -> np.ndarray:
    """Match Lab images using mean/std transfer on all channels.

    Uses a linear mean/std transform on each channel to correct exposure and
    color balance while preserving the original distribution shapes.
    Histogram matching is too aggressive — it forces exact CDF alignment,
    which over-corrects and destroys gradients.

    Parameters
    ----------
    source_lab : np.ndarray
        Source image in L*a*b* (float32, shape H×W×3).
    reference_lab : np.ndarray
        Reference image in L*a*b* (float32, shape H×W×3).

    Returns
    -------
    np.ndarray
        Matched L*a*b* image as float32.
    """
    result = np.empty_like(source_lab)

    for ch in range(3):
        result[:, :, ch] = _mean_std_transfer(
            source_lab[:, :, ch],
            reference_lab[:, :, ch],
        )

    # Clamp L to valid range
    np.clip(result[:, :, 0], 0.0, 100.0, out=result[:, :, 0])

    return result


def _detail_preserving_transfer(
    source_lab: np.ndarray,
    matched_lab: np.ndarray,
    sigma: float = 50.0,
) -> np.ndarray:
    """Combine matched global tone with source local detail.

    Extracts the local detail (high-frequency) from the source and the
    global tone (low-frequency) from the matched result, then recombines
    them. This preserves local contrast and texture from the original.

    For each Lab channel:
        detail   = source - blur(source)        # local texture
        tone     = blur(matched)                 # global color shift
        result   = tone + detail                 # recombined

    Parameters
    ----------
    source_lab : np.ndarray
        Original image in L*a*b* (float32).
    matched_lab : np.ndarray
        Color-matched image in L*a*b* (float32).
    sigma : float
        Gaussian blur radius for tone/detail separation.

    Returns
    -------
    np.ndarray
        Detail-preserved L*a*b* image as float32.
    """
    result = np.empty_like(source_lab)
    for ch in range(3):
        source_ch = source_lab[:, :, ch]
        matched_ch = matched_lab[:, :, ch]

        source_smooth = scipy.ndimage.gaussian_filter(
            source_ch, sigma=sigma,
        )
        matched_smooth = scipy.ndimage.gaussian_filter(
            matched_ch, sigma=sigma,
        )

        detail = source_ch - source_smooth
        result[:, :, ch] = matched_smooth + detail

    return result


def match_to_reference(
    image: np.ndarray,
    lab_reference: np.ndarray,
    strength: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """Match the exposure and color balance of an image to a reference.

    Converts the source image to L*a*b* color space and applies a mean/std
    transfer on each channel to match the reference. Local detail from
    the source is preserved by separating tone and texture, applying the
    correction to the tone layer only, then recombining.

    Parameters
    ----------
    image : np.ndarray
        Source image to transform (uint8 RGB).
    lab_reference : np.ndarray
        Reference image already converted to L*a*b* (float32).
    strength : float
        Blend factor between original (0.0) and fully matched (1.0).
    verbose : bool
        If False, suppress gamut-clipping warnings from lab2rgb.

    Returns
    -------
    np.ndarray
        Matched image as uint8 RGB.
    """
    lab_image = rgb_to_lab(image)

    matched_lab = match_lab(lab_image, lab_reference)
    matched_lab = _detail_preserving_transfer(lab_image, matched_lab)

    if strength < 1.0:
        matched_lab = lab_image + strength * (matched_lab - lab_image)

    with warnings.catch_warnings():
        if not verbose:
            warnings.filterwarnings("ignore", message=".*negative Z values.*")
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
