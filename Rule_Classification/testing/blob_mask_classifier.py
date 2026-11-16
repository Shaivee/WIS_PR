"""
Rule-based defect classifier for ISRA classes.

Two ways to use it:

    TEST MODE        classify(blob, mask)
                     Give a blob image and its mask. The four parameters are
                     calculated from them, then classified.

    PRODUCTION MODE  classify_parameters(parameters, blob=None)
                     Give the four parameters, already calculated, with the
                     original blob. Nothing is calculated from the image; the
                     blob is not opened, only recorded in the result as blob_id.
                     This mode needs only numpy and pandas.

The four parameters
    aspect_ratio    = box_h / box_w
    PERC_BRIGHT     = share of ROI pixels brighter than background (mean + 1 std)
    PERC_DARK       = share of ROI pixels darker than background (mean - 1 std)
    A_to_roi_ratio  = share of ROI pixels that are bright or dark
    The ROI is the mask's bounding box. The background (mean, std) is every
    pixel outside the mask.

Mask requirements
    Same height and width as the blob; the defect is the brighter of two values
    (255/0 or 1/0); exactly one defect region; some non-defect pixels left in the
    image. A mask with many grey levels is thresholded at half its maximum.

The rule tables are read from rules/ (class_rules.csv, subclass_rules.csv).

Command line (run from any folder)
    python blob_mask_classifier.py --blob blob.png --mask mask.png [--explain]
    python blob_mask_classifier.py --blobs BLOBS_DIR --masks MASKS_DIR [--out results.csv]
    python blob_mask_classifier.py --params-csv parameters.csv [--id-col blob] [--out results.csv]

    In folder mode, the mask for blobs_dir/<sub>/name.ext is masks_dir/<sub>/name.ext
    or name_mask.ext (any image extension).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from classify_rules import FEATURE_COLS, RuleBasedClassifier

RULES_DIR = Path(__file__).resolve().parent / "rules"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

MIN_BACKGROUND_PIXELS = 50      # fewer than this: refuse, the background statistics are unusable
WARN_BACKGROUND_PIXELS = 200    # fewer than this: warn
WARN_MIN_BOX_SIDE_PX = 8        # blob narrower than this: warn
IDENTITY_TOLERANCE = 0.005      # slack for parameters that were rounded upstream


# ---------------------------------------------------------------------------
# Test mode: blob + mask -> parameters
# ---------------------------------------------------------------------------

def load_gray(image):
    """Image path or array -> 2-D float grayscale array."""
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float64)
        if arr.ndim == 3:
            if arr.shape[2] >= 3:
                return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
            return arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"blob array must be 2-D or 3-D, got {arr.ndim}-D")
        return arr
    from PIL import Image
    return np.array(Image.open(image).convert("L"), dtype=np.float64)


def load_mask(mask, expected_shape):
    """Mask path or array -> boolean array, True on the defect."""
    if isinstance(mask, np.ndarray):
        m = mask.max(axis=2) if mask.ndim == 3 else mask
    else:
        from PIL import Image
        m = np.array(Image.open(mask).convert("L"))

    if m.shape != tuple(expected_shape):
        raise ValueError(
            f"mask is {m.shape[1]}x{m.shape[0]} (width x height) but the blob is "
            f"{expected_shape[1]}x{expected_shape[0]}; they must be the same size "
            f"(nothing is resized automatically)"
        )

    values = np.unique(m)
    if m.dtype == bool:
        defect = m
    elif len(values) == 1:
        defect = np.full(m.shape, values[0] > 0)
    elif len(values) == 2:
        defect = m == values[1]
    else:
        defect = m > 0.5 * m.max()
    if not defect.any():
        raise ValueError("mask is empty: it contains no defect pixels")
    return defect


def compute_mask_features(gray, mask):
    """Return (parameters, diagnostics, warnings) for one blob."""
    from scipy import ndimage

    height, width = gray.shape
    n_regions = ndimage.label(mask)[1]
    if n_regions > 1:
        raise ValueError(
            f"mask has {n_regions} separate regions; this classifier expects one defect "
            f"per mask (use one mask per defect)"
        )

    ys, xs = np.where(mask)
    y1, y2, x1, x2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    box_h, box_w = y2 - y1, x2 - x1

    # An inverted mask has the whole image as its bounding box.
    if (y1, x1, y2, x2) == (0, 0, height, width) and mask.mean() > 0.5:
        raise ValueError(
            f"mask covers {mask.mean():.0%} of the image and touches all four edges; it looks "
            f"inverted (the defect must be the brighter value) or the blob has no surrounding material"
        )

    background = gray[~mask]
    if background.size < MIN_BACKGROUND_PIXELS:
        raise ValueError(
            f"only {background.size} pixel(s) lie outside the mask (need at least "
            f"{MIN_BACKGROUND_PIXELS}), so there is no usable background reference"
        )
    bg_mean, bg_std = float(background.mean()), float(background.std() + 1e-6)

    roi = gray[y1:y2, x1:x2]
    bright = roi > bg_mean + bg_std
    dark = roi < bg_mean - bg_std
    parameters = {
        "aspect_ratio": box_h / box_w,
        "PERC_BRIGHT": float(bright.mean()),
        "PERC_DARK": float(dark.mean()),
        "A_to_roi_ratio": float((bright | dark).mean()),
    }

    warnings = []
    if background.size < WARN_BACKGROUND_PIXELS:
        warnings.append(f"small background sample ({background.size} px): brightness parameters are less reliable")
    if y1 == 0 or x1 == 0 or y2 == height or x2 == width:
        warnings.append("mask touches the image border: the blob may be cut off or the crop too tight")
    if min(box_h, box_w) < WARN_MIN_BOX_SIDE_PX:
        warnings.append(f"very small blob ({box_w}x{box_h} px): the ratios are coarse and noisy")

    diagnostics = {
        "box_width_px": box_w, "box_height_px": box_h, "mask_area_px": int(mask.sum()),
        "background_pixels": int(background.size), "background_mean": bg_mean, "background_std": bg_std,
    }
    return parameters, diagnostics, warnings


# ---------------------------------------------------------------------------
# Production mode: checking parameters calculated elsewhere
# ---------------------------------------------------------------------------

def validate_parameters(parameters):
    """
    Return (parameters as floats, warnings), or raise ValueError.

    Refused: a missing parameter; a value that is not a finite number;
    aspect_ratio <= 0; PERC_BRIGHT or PERC_DARK outside 0-1.
    Warned: A_to_roi_ratio differing from PERC_BRIGHT + PERC_DARK, or outside 0-1.
    With the calculation above they are always equal (no pixel is both brighter
    and darker than the background), so a difference means the parameters were
    calculated another way and the ISRA ranges may not apply to them.
    """
    values, missing = {}, []
    for name in FEATURE_COLS:
        try:
            raw = parameters[name]
        except (KeyError, IndexError, TypeError):
            missing.append(name)
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number, got {raw!r}")
        if not np.isfinite(number):
            raise ValueError(f"{name} is not a finite number ({raw!r})")
        values[name] = number
    if missing:
        raise ValueError(f"missing parameter(s) {missing}; expected {FEATURE_COLS}")

    if values["aspect_ratio"] <= 0:
        raise ValueError(f"aspect_ratio must be positive, got {values['aspect_ratio']}")
    for name in ("PERC_BRIGHT", "PERC_DARK"):
        if not 0 <= values[name] <= 1:
            raise ValueError(f"{name} is a share of pixels and must be between 0 and 1, got {values[name]}")

    warnings = []
    area, shares = values["A_to_roi_ratio"], values["PERC_BRIGHT"] + values["PERC_DARK"]
    if not 0 <= area <= 1:
        warnings.append(f"A_to_roi_ratio={area:.3f} is outside 0-1, which the expected calculation cannot produce")
    elif abs(area - shares) > IDENTITY_TOLERANCE:
        warnings.append(
            f"A_to_roi_ratio={area:.3f} but PERC_BRIGHT + PERC_DARK={shares:.3f}; these are always "
            f"equal under the expected calculation, so the parameters may have been calculated differently"
        )
    return values, warnings


def blob_label(blob):
    """Short label for the result record. The blob itself is never read."""
    if blob is None:
        return None
    if isinstance(blob, np.ndarray):
        return f"<array {'x'.join(map(str, blob.shape))}>"
    return str(blob)[:120]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MaskBlobClassifier:
    def __init__(self, rules_csv=RULES_DIR / "class_rules.csv", subclass_rules_csv=RULES_DIR / "subclass_rules.csv",
                 confidence_threshold=0.35):
        self.rules = RuleBasedClassifier(rules_csv, subclass_rules_csv, confidence_threshold)

    def classify(self, blob, mask):
        """TEST MODE: blob and mask (paths or arrays) in, class out."""
        gray = load_gray(blob)
        parameters, diagnostics, warnings = compute_mask_features(gray, load_mask(mask, gray.shape))
        return self._result(parameters, diagnostics, warnings, "blob+mask", blob_label(blob))

    def classify_parameters(self, parameters, blob=None):
        """
        PRODUCTION MODE: the four parameters (dict or pandas row) in, class out.
        `blob` is only recorded as blob_id. Keep at least 3 decimals: on 56,933 ISRA
        records, rounding to 3 decimals changed 0.17% of the classes, to 2 decimals 1.6%.
        """
        values, warnings = validate_parameters(parameters)
        return self._result(values, {}, warnings, "parameters", blob_label(blob))

    def classify_parameter_table(self, table, id_col=None):
        """PRODUCTION MODE for a DataFrame. Returns (results, [(row, reason), ...] for refused rows)."""
        missing = [c for c in FEATURE_COLS if c not in table.columns]
        if missing:
            raise ValueError(f"table is missing column(s) {missing}; expected {FEATURE_COLS}")
        if id_col is not None and id_col not in table.columns:
            raise ValueError(f"id column {id_col!r} not found; columns are {list(table.columns)}")

        rows, problems = [], []
        for index, record in table.iterrows():
            ident = record[id_col] if id_col is not None else None
            try:
                rows.append({"row": index, **self.classify_parameters(record[FEATURE_COLS], blob=ident)})
            except ValueError as error:
                problems.append((f"row {index}" + (f" ({ident})" if ident is not None else ""), str(error)))
        return pd.DataFrame(rows), problems

    def classify_folder(self, blobs_dir, masks_dir):
        """TEST MODE for folders. Returns (results, [(blob, reason), ...] for blobs skipped)."""
        blobs_dir, masks_dir = Path(blobs_dir), Path(masks_dir)
        for folder in (blobs_dir, masks_dir):
            if not folder.is_dir():
                raise FileNotFoundError(f"folder not found: {folder}")
        same_folder = blobs_dir.resolve() == masks_dir.resolve()

        rows, problems = [], []
        for blob_path in sorted(blobs_dir.rglob("*")):
            if blob_path.suffix.lower() not in IMAGE_EXTENSIONS or not blob_path.is_file():
                continue
            if same_folder and blob_path.stem.endswith("_mask"):
                continue
            relative = blob_path.relative_to(blobs_dir)
            mask_path = self._find_mask(masks_dir / relative.parent, blob_path.stem)
            if mask_path is None:
                problems.append((str(relative), "no matching mask found"))
                continue
            try:
                rows.append({"folder": "" if str(relative.parent) == "." else str(relative.parent),
                             "blob": blob_path.name, "mask": mask_path.name,
                             **self.classify(blob_path, mask_path)})
            except ValueError as error:
                problems.append((str(relative), str(error)))
        return pd.DataFrame(rows), problems

    def explain(self, parameters):
        return self.rules.explain(parameters)

    @staticmethod
    def _find_mask(folder, stem):
        if not folder.is_dir():
            return None
        by_stem = {}
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                by_stem.setdefault(path.stem, path)
        return by_stem.get(stem) or by_stem.get(stem + "_mask")

    def _result(self, parameters, diagnostics, warnings, input_mode, blob_id):
        prediction = self.rules.predict(parameters)
        if not prediction["accept"]:
            warnings = warnings + [f"low confidence ({prediction['confidence']:.2f}): review manually"]
        return {
            "input_mode": input_mode,
            "blob_id": blob_id,
            "predicted_class": prediction["group"],
            "confidence": prediction["confidence"],
            "low_confidence": not prediction["accept"],
            "fits_all_ranges": prediction["in_range"],
            "total_penalty": round(prediction["total_penalty"], 4),
            "specific_match": prediction.get("subclass"),
            "specific_match_reliability": prediction.get("subclass_reliability"),
            "runner_up": prediction["top_3"][1][0],
            "warnings": "; ".join(warnings),
            **{name: parameters[name] for name in FEATURE_COLS},
            **diagnostics,
        }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def print_result(result, clf, show_explain=False):
    if result["input_mode"] == "parameters":
        print("  Input           : the four parameters only (no image analysed)"
              + (f"; blob {result['blob_id']}" if result["blob_id"] else ""))
    print(f"  Predicted class : {result['predicted_class']}   (confidence {result['confidence']:.2f})")
    if result["specific_match"]:
        print(f"  Closest specific class : {result['specific_match']} "
              f"(right ~{result['specific_match_reliability']:.0%} of the time when the family is correct)")
    print(f"  Runner-up       : {result['runner_up']}")
    print(f"  Fits every range of the predicted class: {'yes' if result['fits_all_ranges'] else 'no'}"
          f"   (total penalty {result['total_penalty']})")
    print("  Parameters      : " + ", ".join(f"{name}={result[name]:.3f}" for name in FEATURE_COLS))
    if "box_width_px" in result:
        print(f"  Blob ROI        : {result['box_width_px']}x{result['box_height_px']} px, background "
              f"{result['background_pixels']} px (mean {result['background_mean']:.1f}, std {result['background_std']:.1f})")
    if result["warnings"]:
        print(f"  Warnings        : {result['warnings']}")
    if show_explain:
        print("\n  " + clf.explain(result).replace("\n", "\n  "))


def report_batch(results, problems, total, out_path):
    results.to_csv(out_path, index=False)
    print(f"Classified {len(results)} of {total} -> {out_path}")
    if problems:
        print(f"\n{len(problems)} refused or skipped:")
        for item, reason in problems:
            print(f"  {item}: {reason}")
    if len(results):
        print("\n" + results["predicted_class"].value_counts().to_string())
        print(f"\nLow-confidence: {int(results['low_confidence'].sum())} of {len(results)}   "
              f"With warnings: {int((results['warnings'] != '').sum())} of {len(results)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rule-based defect classifier (blob + mask, or parameters)")
    parser.add_argument("--blob", help="blob image (test mode, with --mask)")
    parser.add_argument("--mask", help="mask for --blob")
    parser.add_argument("--blobs", help="folder of blob images (test mode, with --masks)")
    parser.add_argument("--masks", help="folder of masks")
    parser.add_argument("--params-csv", help="CSV with the four parameter columns (production mode)")
    parser.add_argument("--id-col", help="column of --params-csv that names each blob")
    parser.add_argument("--out", default="classification_results.csv", help="output CSV for batch modes")
    parser.add_argument("--explain", action="store_true", help="show the range check for the predicted class")
    args = parser.parse_args(argv)

    try:
        clf = MaskBlobClassifier()
        if args.params_csv:
            table = pd.read_csv(args.params_csv)
            results, problems = clf.classify_parameter_table(table, id_col=args.id_col)
            report_batch(results, problems, len(table), args.out)
        elif args.blobs and args.masks:
            results, problems = clf.classify_folder(args.blobs, args.masks)
            report_batch(results, problems, len(results) + len(problems), args.out)
        elif args.blob and args.mask:
            print(f"Blob: {args.blob}   Mask: {args.mask}")
            print_result(clf.classify(args.blob, args.mask), clf, args.explain)
        else:
            parser.error("give --blob with --mask, --blobs with --masks, or --params-csv")
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
