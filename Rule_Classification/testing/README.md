# Testing pipeline

Classifies a defect blob into one of 18 ISRA defect classes using the rules in `rules/`. It needs
nothing from the `training/` folder.

## Two modes

| Mode | Call | Input | What is calculated |
|---|---|---|---|
| Test | `classify(blob, mask)` | blob image + mask | the four parameters, from the image |
| Production | `classify_parameters(parameters, blob=...)` | the four parameters + the original blob | nothing; the blob is not opened, only recorded as `blob_id` |

Both modes end in the same rule, so a blob gets the same class either way. Production mode needs
only numpy and pandas.

## Quick start

```bash
pip install -r requirements.txt
python test_blob_classifier.py        # set BLOB_PATH / MASK_PATH at the top of the file
```

```python
from blob_mask_classifier import MaskBlobClassifier
clf = MaskBlobClassifier()

result = clf.classify("blob.png", "mask.png")                      # test mode

result = clf.classify_parameters(                                  # production mode
    {"aspect_ratio": 1.0, "PERC_BRIGHT": 0.036, "PERC_DARK": 0.791, "A_to_roi_ratio": 0.827},
    blob="blob_0042.png")

result["predicted_class"], result["confidence"], result["warnings"]
```

Command line (works from any folder):

```bash
python blob_mask_classifier.py --blob blob.png --mask mask.png --explain
python blob_mask_classifier.py --blobs BLOBS_DIR --masks MASKS_DIR --out results.csv
python blob_mask_classifier.py --params-csv example_parameters.csv --id-col blob --out results.csv
```

In folder mode the mask for `BLOBS_DIR/<sub>/name.ext` is `MASKS_DIR/<sub>/name.ext` or
`name_mask.ext`. `test_blob_classifier.py` runs test mode, prints the four parameters, then repeats
the classification from those parameters alone and confirms the result matches.

## The four parameters

| Parameter | Definition |
|---|---|
| `aspect_ratio` | `box_h / box_w` |
| `PERC_BRIGHT` | share of ROI pixels brighter than the background (mean + 1 std) |
| `PERC_DARK` | share of ROI pixels darker than the background (mean - 1 std) |
| `A_to_roi_ratio` | share of ROI pixels that are bright or dark |

The ROI is the mask's bounding box. The background mean and std come from every pixel outside the
mask, so a tightly cropped blob works as long as some non-defect pixels remain. Blob size and image
size do not matter: all four are ratios. Production mode expects parameters calculated this way, at
full precision (rounding to 3 decimals changed 0.17% of classes on 56,933 ISRA records, to 2
decimals 1.6%).

## How a class is chosen

For each class, a parameter inside the class's range costs nothing; outside it costs
`(distance outside / range width)^2`. The class with the smallest total wins, ties going to the
class whose range centre the defect is closest to. The result also gives a confidence score (a
fit-and-margin score, not a probability), the runner-up, the closest specific class within a
family, `fits_all_ranges`, `total_penalty`, and any warnings.

## Input checks

| Refused with an error | Warned |
|---|---|
| mask and blob differ in size; mask empty, inverted, or with several defect regions; fewer than 50 background pixels | mask touches the image border; blob under 8 px on a side; fewer than 200 background pixels; low confidence |
| production: missing, non-numeric or NaN parameter; `aspect_ratio` <= 0; `PERC_*` outside 0-1 | production: `A_to_roi_ratio` differs from `PERC_BRIGHT + PERC_DARK` (see Limitations) |

Mask format: same height and width as the blob; the defect is the brighter of two values (255/0 or
1/0); exactly one defect region. A mask with many grey levels is thresholded at half its maximum.

## Accuracy of the shipped rules

Measured by the training pipeline on ISRA's own parameter values: **39.39%** on held-out data
(22,425 of 56,933) and **39.47% +/- 0.54** points in 5-fold cross-validation. Dot, Streak/Scratch,
Hole Small and Dark Spot are the most reliable classes; Floating Debris, Floating Debris Small and
Lump are consistently poor; Edge Crack is unstable (45-80% across folds). Details are in
`training/README.md`. If you retrain, these figures change with the rules.

Test mode and production mode gave identical results on 400 of 400 varied synthetic blobs.

## Limitations

1. **Not validated on image-derived parameters.** The accuracy figures use ISRA's own parameter values. Parameters calculated from images are proxies; accuracy on them has not been measured. Check against blobs whose class you know before relying on it.
2. **`A_to_roi_ratio` is not independent here.** With the calculation above it always equals `PERC_BRIGHT + PERC_DARK`. ISRA's own value is calibrated defect area over box area and is a separate dimension (it can exceed 1). From an image the classifier effectively has three independent values.
3. **Defect-free regions score high.** A region with no defect gives `A_to_roi_ratio` of about 0.34 (Gaussian noise) to 0.40 (uniform noise), because pixels beyond +/-1 std count as bright or dark. The ISRA ranges for Wrinkle Long, Ripple and Tranv.Crack lie entirely below that, so they will usually be penalised. Dot has all three share parameters at exactly 0 (ISRA reports no area or brightness for sub-resolution dots), which an image blob will not reproduce.
4. **Orientation.** `aspect_ratio = box_h / box_w` assumes the vertical axis is the direction ISRA measures as length. Rotate blobs first if yours are turned 90 degrees.
5. **Always returns one of the 18 classes.** There is no "unknown" output; use `fits_all_ranges`, `total_penalty` and `confidence` to judge fit.
6. **No size tiers.** Small / Medium / Large cannot be recovered without physical calibration, so families are reported with an advisory closest specific class.
7. **The output classes are the 18 ISRA classes.** Mapping them to any other class scheme has to be validated separately.

## The 18 classes

| Class | Specific ISRA classes it covers |
|---|---|
| Bright Spot (family) | Bright Spot Small / Medium / Large / Extra Large, Periodic Bright Spot |
| Dark Spot (family) | Dark Spot Small / Medium / Large / Extra Large, Periodic Dark Spot |
| Hole (family) | Hole Medium / Large / Extra Large, Periodic Hole |
| Streak/Scratch (family) | Scratch, Streak Bright, Streak Dark |
| Edge Crack (family) | Edge Crack, Deep Edge Crack |
| Minor Surface Anomaly | Water Splash, Coating Skip |
| Single classes | Hole Small, Wrinkle Long, Splice, Dirt or Debris, Punching, Lump, Dot, Slime Spot, Tranv.Crack, Ripple, Floating Debris Small, Floating Debris |

## Files

| File | Purpose |
|---|---|
| `blob_mask_classifier.py` | the classifier: test mode, production mode, command line |
| `classify_rules.py` | the rule engine (identical scoring copy in `training/rule_scoring.py`) |
| `test_blob_classifier.py` | test one blob + mask, then check the production path |
| `rules/class_rules.csv`, `rules/subclass_rules.csv` | the rule tables this folder reads; replace them with new ones from `training/output/` after retraining |
| `example_parameters.csv` | production-mode CSV example (row 4 is invalid on purpose; row 5 triggers the consistency warning) |
| `sample_blob.png`, `sample_mask.png` | synthetic sample for the test file |
| `requirements.txt` | numpy, pandas, scipy, Pillow |

Tested with Python 3.12, numpy 2.5, pandas 3.0, scipy 1.18, Pillow 12.3.
