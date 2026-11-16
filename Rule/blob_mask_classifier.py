"""
Rule-based defect classifier: input a blob image together with its mask,
get back the ISRA class it matches best.
=============================================================================
No trained model and no segmentation. The mask you supply says where the
defect is; the rules in class_rules.csv decide which ISRA class it matches.

HOW THE MASK IS USED
    1. The mask's tight bounding box is the blob's ROI (region of interest).
    2. Every pixel OUTSIDE the mask is the background reference (mean, std).
       This works even when the blob image is cropped tightly around the
       defect, as long as some non-defect pixels remain in the image.

THE FOUR PARAMETERS (exactly as defined for the rules)
    aspect_ratio    = box_h / box_w
    PERC_BRIGHT     = share of ROI pixels brighter than background (mean + 1 std)
    PERC_DARK       = share of ROI pixels darker than background (mean - 1 std)
    A_to_roi_ratio  = share of ROI pixels that are bright OR dark
                      (how much of the ROI differs from the background)

    The ROI is the whole bounding box, not only the mask pixels. For an
    irregular blob the box contains background, so PERC_BRIGHT/PERC_DARK are
    diluted and A_to_roi_ratio acts as a fill measure. This is the same
    definition the rules were built and validated with. Computing the
    percentages over mask pixels only would make A_to_roi_ratio close to 1
    for every blob and the ISRA ranges would no longer apply.

THE RULE
    For each class, a defect fits a feature if it lies inside that class's
    [lower, upper] range from class_rules.csv. Values outside a range are
    penalised by how far outside they fall (scaled by the range width). The
    predicted class has the smallest total penalty; ties go to the class
    whose range centre the defect sits closest to.

INPUT REQUIREMENTS
    blob : image file or array (grayscale or colour; colour is converted to
           grayscale with standard luminance weights)
    mask : image file or array, SAME height and width as the blob.
           The brighter of two values is the defect (255/0 or 1/0 both work).
           A mask with many grey levels is thresholded at half its maximum.
           If your mask is inverted (defect black), invert it first.
           It must contain exactly one connected defect region.

ORIENTATION
    aspect_ratio assumes the image's vertical axis is the direction ISRA
    calls "machine direction" (its length / width ratio). If your blobs
    come rotated by 90 degrees, pass swap_axes=True (--swap-axes).

FILES NEEDED IN THE SAME FOLDER
    classify_rules.py, class_rules.csv, subclass_rules.csv (optional)

REQUIREMENTS
    pip install numpy pandas pillow scipy

USAGE (command line)
    python blob_mask_classifier.py --blob blob.png --mask mask.png --explain

    # whole folders; masks matched by filename (blob.png <-> blob.png or
    # blob_mask.png); sub-folders are mirrored under the masks folder
    python blob_mask_classifier.py --blobs BLOBS_DIR --masks MASKS_DIR --out results.csv

USAGE (Python)
    from blob_mask_classifier import MaskBlobClassifier
    clf = MaskBlobClassifier()
    result = clf.classify("blob.png", "mask.png")
    print(result["predicted_class"], result["confidence"], result["warnings"])
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from classify_rules import RuleBasedClassifier, FEATURE_COLS

DEFAULT_RULES = "class_rules.csv"
DEFAULT_SUBCLASS_RULES = "subclass_rules.csv"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Safety limits. Below MIN_BACKGROUND_PIXELS the background mean/std are too
# noisy to use, so classification is refused. The WARN_* values only add a
# warning to the result. They are sensible defaults, not values fitted to data.
MIN_BACKGROUND_PIXELS = 50
WARN_BACKGROUND_PIXELS = 200
WARN_MIN_BOX_SIDE_PX = 8


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_gray(image) -> np.ndarray:
    """Path or array -> 2-D float grayscale array."""
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float64)
        if arr.ndim == 3:
            if arr.shape[2] >= 3:
                arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
            else:
                arr = arr[..., 0]
        elif arr.ndim != 2:
            raise ValueError(f"blob array must be 2-D or 3-D, got {arr.ndim}-D")
        return arr
    return np.array(Image.open(image).convert("L"), dtype=np.float64)


def load_mask(mask, expected_shape) -> np.ndarray:
    """Path or array -> boolean array, True where the defect is."""
    if isinstance(mask, np.ndarray):
        m = mask
        if m.ndim == 3:
            m = m.max(axis=2)
    else:
        m = np.array(Image.open(mask).convert("L"))

    if m.shape != tuple(expected_shape):
        raise ValueError(
            f"mask is {m.shape[1]}x{m.shape[0]} (width x height) but the blob is "
            f"{expected_shape[1]}x{expected_shape[0]}. They must be the same size; "
            f"nothing is resized automatically."
        )

    if m.dtype == bool:
        fg = m
    else:
        values = np.unique(m)
        if len(values) == 1:
            fg = np.full(m.shape, values[0] > 0)
        elif len(values) == 2:
            fg = m == values[1]                      # the brighter value is the defect
        else:
            fg = m > 0.5 * m.max()                   # soft / anti-aliased mask

    if not fg.any():
        raise ValueError("mask is empty: it contains no defect pixels")
    return fg


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_mask_features(gray, mask, swap_axes=False, on_multiple="error"):
    """
    Returns (features, diagnostics) for one blob.

    on_multiple : what to do if the mask has more than one separate region.
        "error"   - refuse (default; several regions give a meaningless box)
        "largest" - keep the largest region and record a warning
    """
    height, width = gray.shape
    warnings = []

    labeled, n_regions = ndimage.label(mask)
    if n_regions > 1:
        if on_multiple == "largest":
            sizes = ndimage.sum(mask, labeled, index=range(1, n_regions + 1))
            mask = labeled == (int(np.argmax(sizes)) + 1)
            warnings.append(f"mask had {n_regions} separate regions; kept the largest")
        else:
            raise ValueError(
                f"mask has {n_regions} separate regions; this classifier expects "
                f"one defect per mask. Split them into one mask per defect, or "
                f"keep only the largest region (on_multiple='largest' / --keep-largest)."
            )

    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    box_h, box_w = y2 - y1, x2 - x1

    # An inverted mask (everything EXCEPT the defect marked as defect) has a
    # bounding box equal to the whole image and would otherwise be classified
    # as if the image itself were the blob. A genuine defect does not cover
    # most of the image while touching all four edges.
    if y1 == 0 and x1 == 0 and y2 == height and x2 == width and mask.mean() > 0.5:
        raise ValueError(
            f"mask covers {mask.mean():.0%} of the image and touches all four edges; "
            f"it looks inverted (defect should be the bright value) or the blob has "
            f"no margin of surrounding material."
        )

    background = gray[~mask]
    if background.size < MIN_BACKGROUND_PIXELS:
        raise ValueError(
            f"only {background.size} pixel(s) lie outside the mask (need at least "
            f"{MIN_BACKGROUND_PIXELS}) so there is no usable background reference. "
            f"Use a blob image with some surrounding material, or check that the "
            f"mask is not inverted (defect should be the bright value)."
        )
    if background.size < WARN_BACKGROUND_PIXELS:
        warnings.append(f"small background sample ({background.size} px): brightness features are less reliable")

    bg_mean = float(background.mean())
    bg_std = float(background.std() + 1e-6)

    box_pixels = gray[y1:y2, x1:x2]
    bright = box_pixels > (bg_mean + bg_std)
    dark = box_pixels < (bg_mean - bg_std)

    aspect = (box_w / box_h) if swap_axes else (box_h / box_w)
    features = {
        "aspect_ratio": aspect,
        "PERC_BRIGHT": float(bright.mean()),
        "PERC_DARK": float(dark.mean()),
        "A_to_roi_ratio": float((bright | dark).mean()),
    }

    touches_border = (y1 == 0) or (x1 == 0) or (y2 == height) or (x2 == width)
    if touches_border:
        warnings.append("mask touches the image border: the blob may be cut off or the crop too tight")
    if min(box_h, box_w) < WARN_MIN_BOX_SIDE_PX:
        warnings.append(f"very small blob ({box_w}x{box_h} px): ratios are coarse and noisy")

    diagnostics = {
        "box_width_px": box_w,
        "box_height_px": box_h,
        "mask_area_px": int(mask.sum()),
        "mask_fill_of_box": float(mask.sum() / (box_w * box_h)),
        "background_pixels": int(background.size),
        "background_mean": bg_mean,
        "background_std": bg_std,
        "touches_border": bool(touches_border),
        "image_width_px": width,
        "image_height_px": height,
    }
    return features, diagnostics, warnings


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

class MaskBlobClassifier:
    def __init__(self, rules_csv=DEFAULT_RULES, subclass_rules_csv=DEFAULT_SUBCLASS_RULES,
                 confidence_threshold=0.35):
        self.rules = RuleBasedClassifier(
            rules_csv,
            confidence_threshold=confidence_threshold,
            subclass_rules_csv=subclass_rules_csv,
        )
        self.classes = self.rules.classes

    def classify_features(self, features: dict) -> dict:
        """Classify directly from the four parameter values."""
        missing = [f for f in FEATURE_COLS if f not in features]
        if missing:
            raise ValueError(f"missing feature(s): {missing}")
        return self._result(features, {}, [])

    def classify(self, blob, mask, swap_axes=False, on_multiple="largest") -> dict:
        """Classify one blob given its image and mask (paths or arrays)."""
        gray = load_gray(blob)
        fg = load_mask(mask, gray.shape)
        features, diagnostics, warnings = compute_mask_features(gray, fg, swap_axes, on_multiple)
        return self._result(features, diagnostics, warnings)

    def explain(self, features: dict) -> str:
        """Feature-by-feature check against the predicted class's ranges."""
        return self.rules.explain({f: features[f] for f in FEATURE_COLS})

    def classify_folder(self, blobs_dir, masks_dir, swap_axes=False, on_multiple="error"):
        """
        Classify every blob under blobs_dir. The mask for blobs_dir/<sub>/<name>.<ext>
        is masks_dir/<sub>/<name>.<ext> or masks_dir/<sub>/<name>_mask.<ext>
        (any image extension). Returns (DataFrame, list of (item, reason) problems).
        """
        blobs_dir, masks_dir = Path(blobs_dir), Path(masks_dir)
        if not blobs_dir.is_dir():
            raise FileNotFoundError(f"blobs folder not found: {blobs_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"masks folder not found: {masks_dir}")
        same_dir = blobs_dir.resolve() == masks_dir.resolve()

        rows, problems = [], []
        for blob_path in sorted(blobs_dir.rglob("*")):
            if not blob_path.is_file() or blob_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if same_dir and blob_path.stem.endswith("_mask"):
                continue                      # a mask stored next to its blob, not a blob
            rel = blob_path.relative_to(blobs_dir)
            mask_path = self._find_mask(masks_dir / rel.parent, blob_path.stem)
            if mask_path is None:
                problems.append((str(rel), "no matching mask found"))
                continue
            try:
                row = {"folder": str(rel.parent) if str(rel.parent) != "." else "",
                       "blob": blob_path.name, "mask": mask_path.name}
                row.update(self.classify(blob_path, mask_path, swap_axes, on_multiple))
                rows.append(row)
            except Exception as e:
                problems.append((str(rel), str(e)))
        return pd.DataFrame(rows), problems

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _find_mask(folder: Path, stem: str):
        if not folder.is_dir():
            return None
        by_stem = {}
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                by_stem.setdefault(p.stem, p)
        return by_stem.get(stem) or by_stem.get(stem + "_mask")

    def _result(self, features, diagnostics, warnings):
        pred = self.rules.predict(features)
        x = np.array([features[f] for f in FEATURE_COLS], dtype=float)
        penalty, _ = self.rules._scores(x)
        total_penalty = float(penalty[self.classes.index(pred["group"])])

        if not pred["accept"]:
            warnings = warnings + [f"low confidence ({pred['confidence']:.2f}): review manually"]

        top = pred["top_3"]
        out = {
            "predicted_class": pred["group"],
            "confidence": pred["confidence"],
            "low_confidence": not pred["accept"],
            "fits_all_ranges": pred["in_range"],
            "total_penalty": round(total_penalty, 4),
            "specific_match": pred.get("subclass"),
            "specific_match_reliability": pred.get("subclass_reliability"),
            "runner_up": top[1][0] if len(top) > 1 else None,
            "warnings": "; ".join(warnings),
        }
        out.update({k: features[k] for k in FEATURE_COLS})
        out.update(diagnostics)
        return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _print_result(result, clf, show_explain):
    print(f"  Predicted class : {result['predicted_class']}"
          f"   (confidence {result['confidence']:.2f})")
    if result.get("specific_match"):
        rel = result.get("specific_match_reliability")
        rel_txt = f" (right ~{rel:.0%} of the time when the family is correct)" if rel else ""
        print(f"  Closest specific class : {result['specific_match']}{rel_txt}")
    print(f"  Runner-up       : {result['runner_up']}")
    print(f"  Fits every range of the predicted class: {'yes' if result['fits_all_ranges'] else 'no'}"
          f"   (total penalty {result['total_penalty']})")
    print("  Parameters      : " + ", ".join(f"{f}={result[f]:.3f}" for f in FEATURE_COLS))
    if "box_width_px" in result:
        print(f"  Blob ROI        : {result['box_width_px']}x{result['box_height_px']} px, "
              f"background {result['background_pixels']} px "
              f"(mean {result['background_mean']:.1f}, std {result['background_std']:.1f})")
    if result["warnings"]:
        print(f"  Warnings        : {result['warnings']}")
    if show_explain:
        print()
        print("  " + clf.explain(result).replace("\n", "\n  "))


def _print_batch_summary(df):
    print("\n" + "=" * 78)
    print("RESULTS BY FOLDER")
    print("=" * 78)
    for folder, group in df.groupby("folder"):
        print(f"\n'{folder or '(top level)'}'  ({len(group)} blobs)")
        for cls, n in group["predicted_class"].value_counts().head(4).items():
            print(f"    {cls:30s} {n:4d}  ({n / len(group):.0%})")
    print(f"\nLow-confidence predictions : {int(df['low_confidence'].sum())} of {len(df)}")
    print(f"Blobs with warnings        : {int((df['warnings'] != '').sum())} of {len(df)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Rule-based defect blob classifier (blob + mask)")
    p.add_argument("--rules", default=DEFAULT_RULES)
    p.add_argument("--subclass-rules", default=DEFAULT_SUBCLASS_RULES)
    p.add_argument("--blob", help="blob image")
    p.add_argument("--mask", help="mask for --blob (same size)")
    p.add_argument("--blobs", help="folder of blob images (batch)")
    p.add_argument("--masks", help="folder of masks (batch)")
    p.add_argument("--out", default="classification_results.csv", help="batch output CSV")
    p.add_argument("--explain", action="store_true", help="show the range check for the predicted class")
    p.add_argument("--swap-axes", action="store_true", help="aspect_ratio = width/height (blobs rotated 90 degrees)")
    p.add_argument("--keep-largest", action="store_true", help="if a mask has several regions, keep the largest")
    args = p.parse_args(argv)

    on_multiple = "largest" if args.keep_largest else "error"
    try:
        clf = MaskBlobClassifier(args.rules, args.subclass_rules)

        if args.blobs and args.masks:
            # df, problems = clf.classify_folder(args.blobs, args.masks, args.swap_axes, on_multiple)
            df, problems = clf.classify_folder('/home/shaivee/PycharmProjects/WIS_RuleClassification/Rule/defect_crops/grey-spot/1_png.rf.b9351c5e9cff00f1272392034ac548ac_1.png', '/home/shaivee/PycharmProjects/WIS_RuleClassification/Rule/masks/grey-spot/1_png.rf.b9351c5e9cff00f1272392034ac548ac_1.png', args.swap_axes, on_multiple)
            df.to_csv(args.out, index=False)
            print(f"Classified {len(df)} blobs -> {args.out}")
            if problems:
                print(f"\n{len(problems)} item(s) skipped:")
                for item, why in problems:
                    print(f"  {item}: {why}")
            if len(df):
                _print_batch_summary(df)
            return 0

        if args.blob and args.mask:
            print(f"Blob: {args.blob}   Mask: {args.mask}")
            result = clf.classify(args.blob, args.mask, args.swap_axes, on_multiple)
            _print_result(result, clf, args.explain)
            return 0
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    p.error("provide --blob with --mask, or --blobs with --masks")


if __name__ == "__main__":
    sys.exit(main())
