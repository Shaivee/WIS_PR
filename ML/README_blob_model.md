# ISRA Blob-Ready Defect Classifier — v1

Trained on ISRA CSV data using **only** the 4 features computable from a plain,
uncalibrated blob image: `aspect_ratio`, `PERC_BRIGHT`, `PERC_DARK`, `A_to_roi_ratio`.

Random Forest, 100 trees, max depth 14 (63.2 MB) — reduced from an initial
300-tree build (189 MB) with no measurable accuracy loss (70.76% → 70.74%).

## Files

- `blob_model_v1.pkl` — trained Random Forest (300 trees) + metadata bundle
- `predict_defect.py` — inference script; load the model and predict from raw blob measurements
- `final_model_test_report.csv` — full per-class precision/recall/F1 on held-out test data

## Performance (held-out test set, 56,933 examples, never seen during training)

| Metric | Value |
|---|---|
| Overall accuracy (no threshold) | 70.8% |
| Macro-F1 | 55.1% |
| Accuracy at 0.5 confidence threshold (72.5% coverage) | 82.5% |
| Accuracy at 0.6 confidence threshold (58.1% coverage) | 88.5% |

**Recommended default threshold: 0.5.** Predictions below this confidence should route to manual review rather than being trusted automatically.

## Strong classes (F1 > 0.75) — trust these

Dot, Splice, Hole Small, Hole (family), Slime Spot, Bright Spot (family)

## Weak classes (F1 < 0.4) — expect frequent manual review

Punching, Tranv.Crack, Edge Crack (family), Floating Debris Small, Ripple, Minor Surface Anomaly, Floating Debris

These were already the weakest groups in the full-feature Stage 1 model — removing calibration and USER_3 made them weaker still, not worse in kind. This is a known, documented limitation, not a bug.

## Quick start

```python
from predict_defect import BlobDefectClassifier

clf = BlobDefectClassifier("blob_model_v1.pkl")

result = clf.predict_from_blob(
    blob_width_px=42,
    blob_height_px=118,
    blob_pixel_count=3210,
    blob_mean_intensity=180,
    background_mean_intensity=110,
)
print(result)
```

For higher-fidelity brightness/darkness features (recommended once your
segmentation pipeline is stable), pass actual pixel arrays to
`compute_features_from_pixels()` instead of just mean intensities — see the
docstring in `predict_defect.py`.

## What this model does NOT do

- **No severity tier** (Small/Medium/Large/XL) — that requires physical size calibration, which this version deliberately excludes. It predicts family/type only.
- **Not trained on your blob images** — trained entirely on ISRA CSV data. Validate against a labeled sample of your own blobs before trusting it in production; if accuracy is notably lower than the numbers above, your segmentation or intensity-normalization likely differs from the assumptions baked into feature extraction.

## Upgrade path

If calibration becomes feasible later (a reference object or ruler in frame
to establish mm-per-pixel), retrain using the 8-feature "with calibration"
set — measured at 80.3% accuracy, an 11-point improvement, in the companion
`ISRA_Blob_Ready_Model.html` report.
