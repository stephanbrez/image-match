"""Core image matching logic using histogram matching in L*a*b* color space."""

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


def _match_channel_quantized(
    source: np.ndarray,
    reference: np.ndarray,
    bins: int = 256,
) -> np.ndarray:
    """Match the histogram of a single channel using quantized bincount.

    Bins float values into integer levels, computes CDFs via np.bincount
    (O(n) instead of the np.unique O(n log n) path), then maps source
    values through the matched CDF.

    Parameters
    ----------
    source : np.ndarray
        Source channel (2D float32).
    reference : np.ndarray
        Reference channel (2D float32).
    bins : int
        Number of quantization levels.

    Returns
    -------
    np.ndarray
        Matched channel as float32 with same shape as source.
    """
    s_min = min(source.min(), reference.min())
    s_max = max(source.max(), reference.max())
    value_range = s_max - s_min

    if value_range == 0:
        return source.copy()

    scale = (bins - 1) / value_range

    # Quantize to integer bin indices
    src_idx = np.clip(((source - s_min) * scale).ravel(), 0, bins - 1).astype(np.int32)
    ref_idx = np.clip(((reference - s_min) * scale).ravel(), 0, bins - 1).astype(np.int32)

    # Build CDFs via bincount (fast O(n) path)
    src_counts = np.bincount(src_idx, minlength=bins)
    ref_counts = np.bincount(ref_idx, minlength=bins)

    src_cdf = np.cumsum(src_counts).astype(np.float64)
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)

    src_cdf /= src_cdf[-1]
    ref_cdf /= ref_cdf[-1]

    # Build lookup: for each source bin, find the closest reference bin by CDF
    lookup = np.searchsorted(ref_cdf, src_cdf, side="left")
    lookup = np.clip(lookup, 0, bins - 1)

    # Map source bins through lookup and convert back to float range
    matched_idx = lookup[src_idx]
    matched = matched_idx.astype(np.float32) / scale + s_min

    return matched.reshape(source.shape)


def match_histograms_quantized(
    source_lab: np.ndarray,
    reference_lab: np.ndarray,
) -> np.ndarray:
    """Match per-channel histograms of Lab images using quantized bincount.

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
        result[:, :, ch] = _match_channel_quantized(
            source_lab[:, :, ch],
            reference_lab[:, :, ch],
        )
    return result


def _detail_preserving_transfer(
    source_lab: np.ndarray,
    matched_lab: np.ndarray,
    sigma: float = 50.0,
) -> np.ndarray:
    """Combine matched global tone with source local detail.

    Histogram matching can collapse many distinct values into a narrow range,
    destroying local contrast (detail). This function extracts the local
    detail (high-frequency) from the source and the global tone
    (low-frequency) from the matched result, then recombines them.

    For each Lab channel:
        detail   = source - blur(source)        # local texture
        tone     = blur(matched)                 # global color shift
        result   = tone + detail                 # recombined

    Parameters
    ----------
    source_lab : np.ndarray
        Original image in L*a*b* (float32).
    matched_lab : np.ndarray
        Histogram-matched image in L*a*b* (float32).
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
) -> np.ndarray:
    """Match the exposure and color balance of an image to a reference.

    Converts the source image to L*a*b* color space and performs per-channel
    histogram matching against a pre-converted reference. Local detail from
    the source is preserved by separating tone and texture, applying the
    match to the tone layer only, then recombining.

    The strength parameter controls blending between the original and
    matched image.

    Parameters
    ----------
    image : np.ndarray
        Source image to transform (uint8 RGB).
    lab_reference : np.ndarray
        Reference image already converted to L*a*b* (float32).
    strength : float
        Blend factor between original (0.0) and fully matched (1.0).

    Returns
    -------
    np.ndarray
        Matched image as uint8 RGB.
    """
    lab_image = rgb_to_lab(image)

    matched_lab = match_histograms_quantized(lab_image, lab_reference)
    matched_lab = _detail_preserving_transfer(lab_image, matched_lab)

    if strength < 1.0:
        matched_lab = lab_image + strength * (matched_lab - lab_image)

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
