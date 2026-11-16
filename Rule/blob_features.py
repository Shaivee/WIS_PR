"""
Shared blob segmentation and feature extraction.
=============================================================================
Used by build_blob_dataset.py and classify_blob_folders.py so the exact
same logic (and any future bugfixes) applies everywhere, instead of two
copies silently drifting apart.

Computes the 4 calibration-free features the model expects:
    aspect_ratio, PERC_BRIGHT, PERC_DARK, A_to_roi_ratio

All 4 are computed from the SEGMENTED BLOB's own bounding box — never from
the size of the source image/crop itself. A defect in a tightly-cropped
40x40px image and the same defect in a loosely-cropped 300x300px image
produce identical features, AS LONG AS segmentation correctly finds the
blob in both cases (see the coverage check below for what happens when it
can't).
"""

from pathlib import Path
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def segment_blob(gray: np.ndarray):
    """
    Split one image into (blob_mask, background_mask) using Otsu's method.

    Assumes the defect is the SMALLER of the two Otsu regions — true for
    the large majority of blob crops, where the defect sits inside a
    visible background margin rather than filling the whole frame.

    Otsu assumes two roughly comparable-sized populations. If the blob
    occupies only a tiny fraction of a heavily-padded image, Otsu can fail
    and split background noise in half instead of finding the real blob.
    Empirically this only breaks down once the blob is under ~0.3% of the
    image area (tested down to that limit) — comfortably below any normal
    crop margin, but the check below catches it explicitly rather than
    silently returning garbage if it ever happens.
    """
    thresh = threshold_otsu(gray)
    mask_a = gray > thresh
    mask_b = ~mask_a
    if mask_a.sum() <= mask_b.sum():
        blob_mask, background_mask = mask_a, mask_b
    else:
        blob_mask, background_mask = mask_b, mask_a

    h, w = gray.shape
    ys, xs = np.where(blob_mask)
    touches_all_edges = (
        len(xs) > 0 and xs.min() == 0 and xs.max() == w - 1
        and ys.min() == 0 and ys.max() == h - 1
    )
    coverage = blob_mask.sum() / gray.size
    if touches_all_edges and coverage > 0.30:
        raise ValueError(
            "Segmentation failed: detected region spans the entire image — "
            "the blob is likely too small relative to background padding "
            "for Otsu to find reliably (this only happens with extreme "
            "padding, under ~0.3% blob-to-image area). Use a tighter crop."
        )

    return blob_mask, background_mask


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the single largest connected region (drops speckle/noise)."""
    labeled = label(mask)
    if labeled.max() == 0:
        return mask
    props = regionprops(labeled)
    largest = max(props, key=lambda p: p.area)
    return labeled == largest.label


def compute_features_for_image(image_path, save_mask_dir=None) -> dict:
    """
    Load one blob image (any size, any amount of background padding within
    reason) and compute the 4 model-ready features. Scale-free by
    construction — computed from the blob's own bounding box, never from
    the source image's dimensions.

    save_mask_dir : optional. If given, saves two QA images per input:
        {name}_mask.png     - binary mask (white = detected blob, black = background)
        {name}_overlay.png  - original image with the detected blob tinted red,
                               so you can visually confirm segmentation is
                               finding the actual defect and not the background
    """
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.float64)

    blob_mask, background_mask = segment_blob(gray)
    blob_mask = largest_connected_component(blob_mask)

    ys, xs = np.where(blob_mask)
    if len(xs) == 0:
        raise ValueError("no blob detected after segmentation")

    width_px = xs.max() - xs.min() + 1
    height_px = ys.max() - ys.min() + 1
    roi_area = width_px * height_px
    blob_pixel_count = int(blob_mask.sum())

    blob_pixels = gray[blob_mask]
    background_pixels = gray[background_mask]
    bg_mean = background_pixels.mean()
    bg_std = background_pixels.std() + 1e-6

    if save_mask_dir is not None:
        save_mask_and_overlay(image_path, gray, blob_mask, save_mask_dir)

    return {
        "PERC_BRIGHT": float((blob_pixels > (bg_mean + bg_std)).mean()),
        "PERC_DARK": float((blob_pixels < (bg_mean - bg_std)).mean()),
        "A_to_roi_ratio": float(blob_pixel_count / roi_area),
        # reference columns, not used as model inputs
        "aspect_ratio": float(height_px / width_px),
        "blob_width_px": int(width_px),
        "blob_height_px": int(height_px),
        "blob_pixel_count": blob_pixel_count,
        "image_width_px": gray.shape[1],
        "image_height_px": gray.shape[0],
    }


def save_mask_and_overlay(image_path, gray: np.ndarray, blob_mask: np.ndarray, output_dir) -> None:
    """
    Save the binary segmentation mask and a red-tinted overlay for visual
    QA — lets you actually look at what Otsu + connected-components decided
    was "the defect" for any image, rather than trusting the numbers blind.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    # Binary mask: white blob on black background
    mask_img = Image.fromarray((blob_mask * 255).astype(np.uint8))
    mask_img.save(output_dir / f"{stem}_mask.png")

    # Overlay: original grayscale, blob region tinted red, for quick visual QA
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    overlay = rgb.copy()
    overlay[blob_mask, 0] = 255                      # boost red channel
    overlay[blob_mask, 1] = (rgb[blob_mask, 1] * 0.3).astype(np.uint8)   # dim green
    overlay[blob_mask, 2] = (rgb[blob_mask, 2] * 0.3).astype(np.uint8)   # dim blue
    Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")
