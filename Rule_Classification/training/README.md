# Training pipeline

Builds the class rules from the ISRA data and measures how accurate they are. It needs nothing
from the `testing/` folder.

```bash
pip install -r requirements.txt
python build_rules.py              # writes output/class_rules.csv and output/subclass_rules.csv
python cross_validate_rules.py     # writes output/cross_validation_results.csv
```

## What it produces

| File in `output/` | Content |
|---|---|
| `class_rules.csv` | for each of the 18 classes, a lower and upper bound (25th and 75th percentile) of each of the four parameters |
| `subclass_rules.csv` | the same for each specific ISRA class inside a family, plus the measured accuracy of that within-family match |
| `cross_validation_results.csv` | accuracy of every class in each of 5 folds, with mean and standard deviation |

There is no trained model. A rule is just a percentile range per class and parameter, computed
from the training rows.

## The data

`training_data.csv` has one row per physical defect from ISRA files 1238-1800: merged duplicates
removed, system classes (0, 2, 3, 90, 100) and size outliers over 1,000 mm excluded, at most
30,000 randomly sampled rows per class. `Stage1aGroup` is the class the rules are built for.
The four parameters are ISRA's own values: `aspect_ratio = SizeMD / SizeCD`, `PERC_BRIGHT`,
`PERC_DARK`, `A_to_roi_ratio = A / (roi_w * roi_h)`. The extraction from the raw ISRA export is
not included, because the raw export is not part of this package.

The 18 classes are 12 single ISRA classes (Hole Small, Wrinkle Long, Splice, Dirt or Debris,
Punching, Lump, Dot, Slime Spot, Tranv.Crack, Ripple, Floating Debris Small, Floating Debris)
and 6 groups defined by `FAMILIES` in `build_rules.py`:

| Group | ISRA classes it covers |
|---|---|
| Bright Spot (family) | Bright Spot Small / Medium / Large / Extra Large, Periodic Bright Spot |
| Dark Spot (family) | Dark Spot Small / Medium / Large / Extra Large, Periodic Dark Spot |
| Hole (family) | Hole Medium / Large / Extra Large, Periodic Hole |
| Streak/Scratch (family) | Scratch, Streak Bright, Streak Dark |
| Edge Crack (family) | Edge Crack, Deep Edge Crack |
| Minor Surface Anomaly | Water Splash, Coating Skip |

## How accuracy is measured

`training_data.csv` is split 60 / 20 / 20, stratified by class. The rules use the 60% only. The
held-out 20% (never used to build them) gives the accuracy. The middle 20% is unused and exists
only so the held-out rows are the same ones the reported figures were measured on.

A defect is scored against every class: a parameter inside a class's range costs nothing, outside
it costs `(distance outside / range width)^2`; the lowest total wins, ties going to the class whose
range centre it is closest to (`rule_scoring.py`).

## Results

- Held-out accuracy: **39.39%** (22,425 of 56,933 defects).
- 5-fold cross-validation: **39.47% +/- 0.54** percentage points, so the figure does not depend on the split.
- By class: Dot 90%, Streak/Scratch 72%, Hole Small 63% and Dark Spot 52% are stable across folds. Floating Debris (2%), Floating Debris Small (0.3%) and Lump (14%) are consistently poor. Edge Crack ranges from 45% to 80% across folds (about 100 examples), so no single figure for it is reliable.
- Closest specific class within a family, when the family is right: Streak/Scratch 94.7%, Edge Crack 85.0%, Dark Spot 84.1%, Minor Surface Anomaly 66.3%, Hole 59.2%, Bright Spot 45.8%.

All of these are measured on ISRA's own parameter values.

## Using the new rules

Copy `output/class_rules.csv` and `output/subclass_rules.csv` into `testing/rules/`.

## Files

| File | Purpose |
|---|---|
| `build_rules.py` | builds the rule tables and prints held-out accuracy |
| `cross_validate_rules.py` | 5-fold cross-validation |
| `rule_scoring.py` | the range-scoring formula (identical copy in `testing/classify_rules.py`) |
| `training_data.csv` | the ISRA data the rules are built from |
| `output/` | the generated rule tables and cross-validation results |
| `requirements.txt` | numpy, pandas, scikit-learn |

Tested with Python 3.12, numpy 2.5, pandas 3.0, scikit-learn 1.9.
