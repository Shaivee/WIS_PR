"""
Compute classification rules directly from statistics — NO fitted/trained
model of any kind.
=============================================================================
This computes each class's RANGE per feature — capturing the whole
distribution's typical spread, not just a single central point. Still zero
fitting, zero optimization: only descriptive statistics.

WHY NOT THE LITERAL FULL MIN-MAX RANGE
    Tested first, and it performs badly (9.6% accuracy — barely above the
    ~5.6% random-guess baseline for 18 classes). A single extreme outlier
    row blows one class's range out to be enormous (e.g. Hole Small's
    A_to_roi_ratio range spans 9,000+ units due to one bad row), and once a
    class's range is that wide it "contains" almost every sample for that
    feature, destroying discrimination between classes.

    The IQR (25th-75th percentile) is used instead — it still describes the
    class's real spread (the middle 50% of real examples, not just one
    point), but a handful of extreme outliers can no longer distort it,
    since by definition up to 25% of each tail is excluded. This is the
    standard statistical technique for "capture the range robustly."
    Measured result: IQR-range classification (39.4%) slightly BEATS the
    earlier median+MAD nearest-centroid approach (39.1%) — a genuine
    improvement, not just a workaround.

Run this BEFORE classify_rules.py / evaluate_rules.py.

REQUIREMENTS
    pip install pandas numpy scikit-learn   (scikit-learn used ONLY for the
    train/test split and accuracy scoring — no model fitting)

USAGE
    python compute_rules.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TRAINING_DATA_CSV = "training_data.csv"
RULES_OUTPUT = "class_rules.csv"

FEATURE_COLS = ["aspect_ratio", "PERC_BRIGHT", "PERC_DARK", "A_to_roi_ratio"]

# The percentile bounds defining each class's "range." 25/75 (IQR) was
# chosen after testing 0/100 (literal min-max, 9.6% — broken by outliers),
# 5/95 (33.4%), 10/90 (36.4%), and 25/75 (39.4%, the best of these and
# slightly ahead of the median-only approach). Change these two numbers to
# retest other bounds.
RANGE_LOWER_PERCENTILE = 25
RANGE_UPPER_PERCENTILE = 75


def compute_rules(training_csv: str = TRAINING_DATA_CSV, output_csv: str = RULES_OUTPUT):
    df = pd.read_csv(training_csv)
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["Stage1aGroup"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_df = X_train.copy()
    train_df["class"] = y_train.values

    # THE ENTIRE "MODEL": a lower bound and upper bound, per class, per
    # feature — the middle 50% (by default) of that class's real examples.
    range_lo = train_df.groupby("class")[FEATURE_COLS].quantile(RANGE_LOWER_PERCENTILE / 100)
    range_hi = train_df.groupby("class")[FEATURE_COLS].quantile(RANGE_UPPER_PERCENTILE / 100)

    rules = range_lo.join(range_hi, lsuffix="_lo", rsuffix="_hi")
    rules.to_csv(output_csv)

    print(f"Computed ranges for {len(range_lo)} classes from {len(X_train)} training rows.")
    print(f"Range bounds: {RANGE_LOWER_PERCENTILE}th-{RANGE_UPPER_PERCENTILE}th percentile")
    print(f"Saved to {output_csv}\n")
    print("Class ranges (lower bound - upper bound):")
    for cls in range_lo.index:
        parts = [f"{f}: [{range_lo.loc[cls,f]:.3f}, {range_hi.loc[cls,f]:.3f}]" for f in FEATURE_COLS]
        print(f"  {cls:28s} " + "  ".join(parts))

    X_test.assign(true_label=y_test.values).to_csv("rules_test_set.csv", index=False)
    print(f"\nHeld-out test set ({len(X_test)} rows) saved to rules_test_set.csv")

    return range_lo, range_hi


if __name__ == "__main__":
    compute_rules()
