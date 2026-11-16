# Testing pipeline

Classifies a defect blob and returns JSON with the class it falls in and the runner-up class.
It needs nothing from the `training/` folder.

```bash
pip install -r requirements.txt
python test_blob_classifier.py     # set BLOB_PATH and MASK_PATH in the file
```

```python
from blob_mask_classifier import MaskBlobClassifier

MaskBlobClassifier().classify("blob.png", "mask.png")
# '{"class": "Dark Spot (family)", "runner_up": "Lump"}'
```

`blob` and `mask` are image file paths or 2-D arrays of the same size. The defect in the mask is
the brighter value (255/0 or 1/0) and forms exactly one region.

If the four parameters are already calculated, skip the image step:

```python
MaskBlobClassifier().classify_parameters(
    {"aspect_ratio": 1.0, "PERC_BRIGHT": 0.04, "PERC_DARK": 0.79, "A_to_roi_ratio": 0.83})
```

## The four parameters

The ROI is the mask's bounding box; the background (mean, std) is every pixel outside the mask.

| Parameter | Definition |
|---|---|
| `aspect_ratio` | `box_h / box_w` |
| `PERC_BRIGHT` | share of ROI pixels brighter than the background (mean + 1 std) |
| `PERC_DARK` | share of ROI pixels darker than the background (mean - 1 std) |
| `A_to_roi_ratio` | share of ROI pixels that are bright or dark |

All four are ratios, so blob size and image size do not matter. A class is chosen by how well the
four values fit each class's [lower, upper] ranges in `rules/class_rules.csv`: outside a range
costs `(distance outside / range width)^2`, and the lowest total wins.

## Refused inputs

A `ValueError` is raised for: blob and mask of different size; a mask with no defect or with more
than one defect region; a mask that is inverted or leaves fewer than 50 background pixels; and
parameters that are missing, not finite, `aspect_ratio` <= 0, or `PERC_*` outside 0-1.

## Accuracy and limitations

The shipped rules were measured by the training pipeline on ISRA's own parameter values: **39.39%**
on held-out data (22,425 of 56,933) and 39.47% +/- 0.54 points in 5-fold cross-validation.

1. **Not validated on image-derived parameters.** Parameters calculated from images are proxies for ISRA's; accuracy on them has not been measured. Check against blobs whose class you know.
2. **Always returns a class.** There is no "unknown" and no confidence in the output; a defect unlike any ISRA class still gets the nearest one.
3. **`A_to_roi_ratio` is not independent here.** From an image it always equals `PERC_BRIGHT + PERC_DARK`. A region with no defect scores about 0.34-0.40 on it, above the ISRA ranges for Wrinkle Long, Ripple and Tranv.Crack.
4. **Orientation.** `aspect_ratio = box_h / box_w` assumes the vertical axis is the direction ISRA measures as length; rotate blobs first if yours are turned 90 degrees.

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
| `blob_mask_classifier.py` | the classifier (one class) |
| `test_blob_classifier.py` | classifies one blob + mask and prints the JSON |
| `rules/class_rules.csv` | the rule table; replace it with a new one from `training/output/` after retraining |
| `sample_blob.png`, `sample_mask.png` | synthetic sample for the test file |
| `requirements.txt` | numpy, pandas, scipy, Pillow |

Tested with Python 3.12, numpy 2.5, pandas 3.0, scipy 1.18, Pillow 12.3.
