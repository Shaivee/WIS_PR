# ISRA Blob Defect Classifier — pure statistics, range-based, no ML model

Still no scikit-learn model, no fitting, no optimization. This version
captures each class as a **range** (not a single point) per feature — the
25th-75th percentile (IQR) of real training examples — so it reflects the
actual spread of each class, not just its typical center.

## The method, in full

1. **For each of the 18 reference classes**, compute the **25th and 75th
   percentile** (the IQR) of each of the 4 features
   (`aspect_ratio`, `PERC_BRIGHT`, `PERC_DARK`, `A_to_roi_ratio`) from the
   training data. That's the entire rule table: `class_rules.csv`, 18 rows
   × 4 features × 2 numbers (lower bound, upper bound) each.

2. **To classify a new defect**, for every class check whether each
   feature value falls inside that class's range:

   ```
   penalty_i = 0                                    if value is inside [lo, hi]
   penalty_i = ((lo - value) / range_width)²         if value is below the range
   penalty_i = ((value - hi) / range_width)²         if value is above the range

   total_penalty(class) = sum of penalty_i over all 4 features
   ```

   Classify as whichever class has the smallest total penalty. Ties (the
   sample fits perfectly inside more than one class's range) are broken by
   preferring the class whose range **center** the value sits closest to —
   among equally plausible classes, prefer the more central match.

## Why not the literal full min-max range (what was tried first)

Tested directly, and it performs badly — **9.6% accuracy**, barely above
the ~5.6% random-guess baseline for 18 classes. The reason: a single
extreme outlier row can blow one class's range out enormously (Hole
Small's `A_to_roi_ratio` literal range spans over 9,000 units because of
one bad row). Once a class's range is that wide, it "contains" almost
every sample for that feature, and the classifier loses all power to
discriminate between classes.

**The fix: IQR (25th-75th percentile) instead of literal min/max.** This
still describes the real spread of the class — the middle 50% of actual
examples — but up to 25% of extreme values on each end are excluded by
definition, so a handful of outliers can no longer distort it. This is the
standard statistical technique for describing a distribution's range
robustly.

## Results, tested honestly across every option

| Range definition | Accuracy | Verdict |
|---|---|---|
| Full min-max (literal) | 9.6% | Broken by outliers — not used |
| 5th-95th percentile | 33.4% | Better, still outlier-sensitive |
| 10th-90th percentile | 36.4% | Better still |
| **IQR, 25th-75th percentile (delivered)** | **39.4%** | Best of the range-based options |
| (for comparison: earlier median+MAD nearest-centroid) | 39.1% | Range-based edges this out |
| (for comparison: fitted Decision Tree, not used anymore) | 57.6% | The real cost of zero fitting |

The IQR-range method is a genuine, measured improvement over the earlier
single-point (median) approach — not just conceptually closer to "the
whole class," but very slightly more accurate too.

## What's actually inside each family

The 18 classes in `class_rules.csv` aren't all single defect types — 6 of
them are groups covering multiple original ISRA classes. Here's exactly
what each one contains:

| Group | Original ISRA classes inside it |
|---|---|
| **Bright Spot (family)** | Bright Spot Small, Bright Spot Medium, Bright Spot Large, Bright Spot Extra Large, Periodic Bright Spot |
| **Dark Spot (family)** | Dark Spot Small, Dark Spot Medium, Dark Spot Large, Dark Spot Extra Large, Periodic Dark Spot |
| **Hole (family)** | Hole Medium, Hole Large, Hole Extra Large, Periodic Hole *(Hole Small is deliberately NOT included — see "why some classes stayed separate" below)* |
| **Streak/Scratch (family)** | Scratch, Streak Bright, Streak Dark |
| **Edge Crack (family)** | Edge Crack, Deep Edge Crack |
| **Minor Surface Anomaly** | Water Splash, Coating Skip |

The remaining 12 classes in the rule table are single, ungrouped ISRA
classes: Hole Small, Wrinkle Long, Splice, Dirt or Debris, Punching, Lump,
Dot, Slime Spot, Tranv.Crack, Ripple, Floating Debris Small, Floating
Debris.

3 classes are excluded entirely — Streak Short (data anomaly, impossible
size values), Formation Irregularity (a system/report event, not a real
defect), and Unknown (a catch-all label, not a real class).

## How each grouping decision was actually made

Every group above passed two independent checks — nothing was grouped on
looks alone.

**Check 1 — the manufacturer's own governed taxonomy.** The ISRA reference
data (`Defect classes.csv`) has a `CustomClassId` column — an official
rollup grouping that existed in the source system but was never surfaced
in the original CSV handoff documentation. Edge Crack + Deep Edge Crack
already share one official group (id 4); Scratch + Streak Bright + Streak
Dark already share another (id 6). Both matched what the structural data
independently suggested.

**Check 2 — isolated separability testing.** Before merging any two
classes, they were tested head-to-head in isolation (not competing against
all 17 other classes at once), because early testing showed that a crowded
multi-class comparison can make two genuinely distinct classes *look*
confused with each other simply from lack of "room" to find the real
boundary — that false lead happened once already (Dark Spot Small vs.
Lump looked confused in a full 18-class test, but separated at 95%+
accuracy once tested one-on-one, so they were kept separate). The same
check was applied to the actual merge candidate: Water Splash and Coating
Skip, tested in isolation, genuinely could not be told apart (near-random
accuracy) — that's real overlap, not a testing artifact, so those two were
merged into "Minor Surface Anomaly."

**Why the size tiers (Small/Medium/Large/Extra Large) got collapsed.**
Within Bright Spot, Dark Spot, and Hole, the size-tier labels turned out to
sit on one continuous size spectrum rather than being structurally
distinct shapes — a "Small" and a "Medium" of the same family differ
mainly by where a size cutoff falls, not by a different silhouette. Rather
than ask the classifier to guess an essentially arbitrary threshold, the
tiers were grouped into one family label; recovering severity (if needed)
is a separate, deterministic size-threshold step, not a classification
problem, and isn't part of this rule table since it needs physical
calibration this model deliberately doesn't use.

**Why some classes stayed separate despite looking similar.** Hole Small
is officially filed under the "Bright Spots" custom group in the source
taxonomy (not "Holes," where every other Hole size sits) — a genuine
inconsistency in ISRA's own reference data. Rather than force it into
either family based on a guess, it was kept as its own standalone class
here, flagged as a taxonomy quirk worth raising with whoever owns that
system.

## Which specific class matches best within a family

`class_rules.csv` tells you the family (e.g. "Dark Spot (family)"). A
second file, `subclass_rules.csv`, goes one level deeper: given the family
is right, which specific original ISRA class inside it does the sample
match best? Same range-check formula, just scoped to only that family's
members.

```python
result = clf.predict(features)
print(result["group"])                  # "Dark Spot (family)"
print(result["subclass"])                # "Dark Spot Medium" -- the specific best match
print(result["subclass_reliability"])    # 0.837 -- how often this sub-step is right, GIVEN the family is correct
print(result["subclass_options"])        # all members of the family, ranked best to worst match
```

**This was tested, not assumed to work.** Sub-class accuracy varies a lot
by family, because some families' members mainly differ by absolute size
(which this model doesn't have) while others differ by real shape/
brightness signature (which it does):

| Family | Sub-class accuracy (given correct family) |
|---|---|
| Streak/Scratch (family) | 94.7% — Scratch, Streak Bright, Streak Dark have genuinely different brightness signatures |
| Edge Crack (family) | 85.0% |
| Dark Spot (family) | 83.7% |
| Minor Surface Anomaly | 66.3% |
| Hole (family) | 59.3% |
| Bright Spot (family) | 45.8% — its 5 members differ mostly by size, which isn't available here |

Every `classify_blob_folders.py` summary now reports the reliability
number alongside the sub-class guess, so a 46% number isn't presented with
the same confidence as a 95% one.

## What each family is actually made of (the real population baseline)

`family_composition.csv` has the exact composition of every family,
computed directly from the full ISRA dataset (not a sample) — how common
each specific class really is within its family:

| Family | Composition |
|---|---|
| Bright Spot (family) | Small 90.1%, Medium 5.1%, Periodic 3.4%, Large 1.1%, Extra Large 0.3% |
| Dark Spot (family) | Small 95.1%, Medium 3.9%, Periodic 0.9%, Large 0.2%, Extra Large 0.0% |
| Hole (family) | Extra Large 39.7%, Medium 31.0%, Large 27.7%, Periodic 1.6% |
| Streak/Scratch (family) | Scratch 73.2%, Streak Dark 13.7%, Streak Bright 13.1% |
| Edge Crack (family) | Deep Edge Crack 94.4%, Edge Crack 5.6% |
| Minor Surface Anomaly | Coating Skip 52.7%, Water Splash 47.3% |

`classify_blob_folders.py` shows this baseline next to your own images'
breakdown automatically — e.g. if your batch matches 100% to a sub-class
that's normally only 14% typical for that family, that's worth a second
look; it may mean your batch is genuinely unusual, or that the sub-class
matching (which is far less reliable than the family-level call — see the
accuracy table above) is struggling on this particular batch.

## Visual QA: masks and overlays

Both `build_blob_dataset.py` and `classify_blob_folders.py` can save two
images per input, mirroring your folder structure under a `masks/`
directory:

- `{name}_mask.png` — binary mask, white = what segmentation identified as
  the blob, black = background
- `{name}_overlay.png` — the original image with the detected blob
  tinted red, for a fast visual check that segmentation found the actual
  defect and not something else

Controlled by `SAVE_MASKS_TO` at the top of each script (set to `None` to
skip). This is the single best way to catch a bad segmentation before it
silently produces wrong features — if the red tint in an overlay doesn't
land on the real defect, the resulting `aspect_ratio`/`PERC_BRIGHT`/etc.
for that image are unreliable, no matter what the rule table says.

## Using your own masks instead of auto-segmentation

If you have more reliable masks of your own, `build_blob_dataset_from_masks.py`
uses them directly instead of Otsu auto-segmentation.

**Mask format required:**

| Property | Requirement |
|---|---|
| File type | PNG (lossless — avoid JPEG, its compression blurs binary edges) |
| Color mode | Single-channel grayscale |
| Pixel values | Strictly binary: 255 (white) = defect, 0 (black) = background |
| Dimensions | Must exactly match the corresponding original image, pixel-for-pixel |
| Background | Original image must still show real background around the blob — the mask says *where* the blob is, but brightness comparison still needs real background pixels to compare against |

**Folder structure required** — a mask folder mirroring your image folder,
same filenames:

```
blob_images/bright_spot/img001.png
blob_masks/bright_spot/img001.png   <- its mask
```

Edit `IMAGES_ROOT`, `MASKS_ROOT`, and `FOLDER_TO_GROUP` at the top of
`build_blob_dataset_from_masks.py` and run it — same output format as
`build_blob_dataset.py`, so `evaluate_on_blobs.py` works on either one
unchanged.

Three checks happen automatically and fail loudly rather than silently:
a mask with different dimensions than its image, an all-white mask (no
background to compare against), an all-black mask (no blob at all), and
**a mask containing more than one disconnected blob region** — each raises
a clear error naming the exact file, instead of producing a quietly wrong
feature row. That last check matters if any of your 127 images might
contain more than one defect: give each defect its own mask (and, if
needed, a duplicated copy of the source image under a distinct filename)
rather than one mask covering multiple separate blobs.

**Both entry points support provided masks:**
- `build_blob_dataset_from_masks.py` — for when you have ground-truth
  labels and want a real accuracy score (mirrors `build_blob_dataset.py` +
  `evaluate_on_blobs.py`)
- `classify_blob_folders_from_masks.py` — for unsupervised exploration, no
  labels needed (mirrors `classify_blob_folders.py`, same rich sub-class +
  family-baseline output)

## Working from YOLO-labeled images (no auto-segmentation at all)

If you have raw images plus YOLO-format labels (`class_id x_center y_center
width height`, normalized 0-1) instead of pre-cropped blobs or masks, use
`extract_from_yolo_labels.py` + `run_yolo_pipeline.py` — this replaces
Otsu-based segmentation entirely, since your annotations already tell us
exactly where each defect is.

**How the 4 features are computed without any segmentation guessing:**

| Feature | How |
|---|---|
| `aspect_ratio` | Directly from the YOLO box dimensions (height ÷ width) — exact, ground-truth, no guessing |
| `PERC_BRIGHT` / `PERC_DARK` | Compared against background statistics computed from **the rest of the source image** — every pixel outside any labeled box in that image. Real background, not a padding guess. |
| `A_to_roi_ratio` | The proportion of pixels *inside* the box that are meaningfully different (brighter or darker) from that real background — the closest honest equivalent to solidity without a true pixel mask |

This is more reliable than Otsu precisely because nothing is guessed blind:
the one feature that's exact uses your real annotation, and the background
reference is measured from real pixels in the same image, not assumed.

**Usage:**

1. Edit `IMAGES_DIR`, `LABELS_DIR`, `OUTPUT_ROOT`, and `CLASSES` at the top
   of `extract_from_yolo_labels.py` to match your setup (your `CLASSES`
   dict — e.g. `{0:'grey-spot', 1:'hole', ...}` — doesn't need to match the
   18 reference classes; the pipeline maps between them automatically).
2. Run `python run_yolo_pipeline.py` — this extracts every defect crop
   (saved into `OUTPUT_ROOT/<your_class_name>/`), computes features, and
   classifies each one against the 18 reference classes in one pass,
   printing the same rich per-class + sub-class + family-baseline
   comparison report as `classify_blob_folders.py`.

**Tested end-to-end** against synthetic YOLO-labeled images (including a
missing-label-file case, correctly warned and skipped) — a dark round
blob, a bright round blob, and an elongated dark region all extracted with
exactly correct aspect ratios and brightness/darkness classification.

**Visual QA for this pipeline is different from the mask-based ones.**
There's no separate blob/background pixel split here — every pixel in a
saved crop is inside your annotated box by definition. What `SAVE_MASKS_TO`
in `extract_from_yolo_labels.py` visualizes instead is the brightness
classification `A_to_roi_ratio` is actually built from: pixels tinted
**yellow** counted as "bright" (relative to real background from
elsewhere in the source image), pixels tinted **blue** counted as "dark,"
and untinted pixels counted as neither. If a crop comes back entirely one
color, that's expected for a solid, uniform defect; if it looks noisy or
inverted relative to what you'd expect by eye, that's the signal to check
your background-stats assumption is right for that image.

## Reversing direction: build rules from YOUR data, test on real ISRA data

Everything above builds rules from ISRA data and applies them to your
images. `compute_rules_generic.py` + `test_rules_on_isra.py` do the
reverse — build rules from your own extracted features (any CSV with the
4 feature columns + a class column), then test those rules against
`rules_test_set.csv`, ISRA's real held-out test data.

**Two modes:**
- **With a class mapping** (`{"hole": "Hole (family)", ...}`) — scores real
  accuracy: how often your rules correctly predict the mapped ISRA class
  on real ISRA examples.
- **Without one** — fully unsupervised: shows which of *your* classes each
  real ISRA reference class's data resembles, no correspondence assumed.

```python
from compute_rules_generic import compute_rules_from_dataframe
from test_rules_on_isra import evaluate_custom_rules_on_isra
import pandas as pd

my_features = pd.read_csv("yolo_extracted_features.csv")
compute_rules_from_dataframe(my_features, class_col="class_name", output_csv="my_rules.csv")

evaluate_custom_rules_on_isra("my_rules.csv", class_mapping={
    "hole": "Hole (family)",
    "wrinkle": "Wrinkle Long",
})
```

**Sample size matters a lot here, and I tested this directly rather than
just warning about it abstractly.** Running this with only 1 example per
class (the synthetic YOLO test data) produced completely uninformative
results — e.g. real ISRA "Bright Spot" examples matched "hole" 95% of the
time, "Wrinkle Long" matched "hole" 64% of the time — because with a
single example, a class's "range" collapses to one exact point with zero
width, so everything gets force-matched to whichever tiny point happens to
be least-far-away, regardless of whether it's actually close. This isn't
a bug — it's exactly why `compute_rules_from_dataframe` prints a warning
for any class under 5 samples. **For this to produce a meaningful result
on your real 127 images, each of your 6 classes needs enough real examples
(ideally dozens+) for a genuine 25th-75th percentile spread to exist.**

## How stable is this, really? (cross-validation)

A single train/test split — what `compute_rules.py` + `evaluate_rules.py`
use day-to-day — gives one accuracy number per class. For small classes,
that one number can be misleading purely by chance, depending on which
examples happened to land in the test set. `cross_validate_rules.py`
re-splits the data 5 times, rebuilds the rules fresh each time, and
reports the full range — so you know which numbers to trust and which to
treat as noisy.

**Overall accuracy is very stable: 39.47% ± 0.54%** across 5 different
splits (range 38.5%–40.0%) — the headline number isn't a lucky fluke.

**But that stability breaks down completely for the smallest class:**

| Class | Mean accuracy | Std dev | Range across folds | Verdict |
|---|---|---|---|---|
| Edge Crack (family) | 67.5% | **±13.2%** | 45%–80% | **Don't trust any single number** — could be anywhere in this range depending purely on which examples land in the test set (only 108 total examples) |
| Floating Debris | 1.6% | ±2.0% | 0%–4% | Stable, but consistently poor — a trustworthy negative result |
| Floating Debris Small | 0.3% | ±0.4% | 0%–1% | Same — consistently near-zero, not bad luck |
| Everything else (14 classes) | — | under ±6% | — | Stable — trust the reported number |

Full results in `cross_validation_stability.csv`. Re-run this any time the
underlying training data changes — a class moving from "stable" to
"unstable" (or vice versa) is a real signal about whether it has enough
examples yet.

## Reading a class's rules directly

```
Streak/Scratch (family)   aspect_ratio: [6.829, 17.389]   PERC_BRIGHT: [0.100, 0.389]
                           PERC_DARK: [0.346, 0.794]        A_to_roi_ratio: [0.152, 0.375]
```

Read this as: "the middle 50% of real Streak/Scratch examples have an
aspect ratio between 6.8 and 17.4, are 10-39% bright, 35-79% dark, and fill
15-38% of their own bounding box." Every one of the 18 classes has a row
like this in `class_rules.csv` — the complete, human-readable rule book.

## Per-class results (held-out test data)

Strong (F1 > 0.5): Dot (0.91), Streak/Scratch (0.63), Bright Spot (0.54), Dark Spot (0.52)

Weak (F1 < 0.1): Floating Debris (0.00), Floating Debris Small (0.01), Ripple (0.02), Minor Surface Anomaly (0.05), Tranv.Crack (0.07), Edge Crack (0.08)

Full numbers in `rules_test_report.csv`.

## Files

| File | Purpose |
|---|---|
| `compute_rules.py` | Computes `class_rules.csv` from `training_data.csv` — descriptive percentiles only, no fitting |
| `class_rules.csv` | The complete family-level rule table — 18 classes × 4 features × [lower, upper] bounds |
| `subclass_rules.csv` | The specific-class rule table within each family — same format, scoped to family members |
| `classify_rules.py` | The range-check formula + `RuleBasedClassifier`, plus `.explain()` for full transparency on any prediction, plus sub-class matching |
| `evaluate_rules.py` | Scores the rules against held-out test data |
| `blob_features.py`, `build_blob_dataset.py`, `classify_blob_folders.py`, `evaluate_on_blobs.py`, `summarize_dataset.py` | Unchanged in purpose — call `RuleBasedClassifier`, same interface regardless of which statistical method is behind it. First two also save mask/overlay QA images. |
| `build_blob_dataset_from_masks.py` | Same as `build_blob_dataset.py`, but uses your own provided masks instead of auto-segmentation — see mask format spec above |
| `classify_blob_folders_from_masks.py` | Same as `classify_blob_folders.py`, but uses your own provided masks — no labels needed |
| `extract_from_yolo_labels.py` | Extracts defect crops + computes features directly from raw images + YOLO labels — no segmentation of any kind |
| `run_yolo_pipeline.py` | Chains extraction + classification into one command, with the same rich reporting as `classify_blob_folders.py` |
| `compute_rules_generic.py` | Builds IQR rules from ANY labeled feature dataset (not just ISRA) — the reverse-direction tool |
| `test_rules_on_isra.py` | Tests rules built from any source against real, held-out ISRA data |
| `cross_validate_rules.py` | 5-fold cross-validation — tells you which per-class accuracy numbers are stable vs. noisy |

## Quick start

```python
from classify_rules import RuleBasedClassifier

clf = RuleBasedClassifier("class_rules.csv")

result = clf.predict({
    "aspect_ratio": 12.5,
    "PERC_BRIGHT": 0.1,
    "PERC_DARK": 0.7,
    "A_to_roi_ratio": 0.25,
})
print(result)

# See exactly which class ranges this value falls inside/outside of:
print(clf.explain(result and {"aspect_ratio": 12.5, "PERC_BRIGHT": 0.1,
                               "PERC_DARK": 0.7, "A_to_roi_ratio": 0.25}))
```

## Changing the range width

`RANGE_LOWER_PERCENTILE` / `RANGE_UPPER_PERCENTILE` at the top of
`compute_rules.py` control this. 25/75 (IQR) tested best; wider bounds
(10/90, 5/95) are also available and were measured above if a different
tradeoff is wanted — wider ranges are more forgiving of edge cases but
less discriminating between classes.

## What this does NOT do

- No severity tier (needs calibration, excluded by design, same as before).
- No shape boundary more complex than "which class's typical range do you
  fit inside" — a fitted tree or forest will always beat this on raw
  accuracy; that is the tradeoff of using zero fitting.
- Not trained on your blob images — validate against a labeled sample of
  your own before trusting it.
