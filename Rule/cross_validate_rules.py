"""
K-fold cross-validation for the IQR-range rule-based classifier.
=============================================================================
A single train/test split (what compute_rules.py + evaluate_rules.py use
day-to-day) gives one accuracy number — but for small classes, that one
number can be misleading purely by chance, depending on which examples
happened to land in the test set. This script re-splits the data K times
(default 5), rebuilds the rules fresh from each fold's training portion,
and evaluates on that fold's held-out portion — so you get a genuine
accuracy RANGE per class, not a single possibly-lucky-or-unlucky number.

No model is trained here in any fitted/optimized sense — every fold just
recomputes the same IQR percentiles from that fold's training rows. This
is purely about checking how STABLE the resulting rules and their accuracy
are across different data splits, not a different classification method.

HOW TO READ THE RESULTS
    - A class with a small std-dev across folds: the accuracy number for
      it is trustworthy — different data splits agree closely.
    - A class with a LARGE std-dev across folds: don't trust any single
      reported number for it — the true accuracy could be anywhere in
      the range the folds show. This usually means the class doesn't
      have enough examples for a stable percentile-range estimate.
    - A class that's consistently low (small std-dev, low mean): that's
      a stable, trustworthy NEGATIVE finding — the rules genuinely don't
      work well for it, reliably, not due to bad luck in one split.

REQUIREMENTS
    pip install pandas numpy scikit-learn

USAGE
    python cross_validate_rules.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

TRAINING_DATA_CSV = "training_data.csv"
CLASS_COL = "Stage1aGroup"
FEATURE_COLS = ["aspect_ratio", "PERC_BRIGHT", "PERC_DARK", "A_to_roi_ratio"]
N_FOLDS = 5
LOWER_PCT = 25
UPPER_PCT = 75
RANDOM_STATE = 42


def build_ranges(train_df: pd.DataFrame, class_col: str):
    """Same method as compute_rules.py: per-class IQR from this fold's training rows only."""
    lo = train_df.groupby(class_col)[FEATURE_COLS].quantile(LOWER_PCT / 100)
    hi = train_df.groupby(class_col)[FEATURE_COLS].quantile(UPPER_PCT / 100)
    return lo, hi


def classify_vectorized(X_test: pd.DataFrame, lo: pd.DataFrame, hi: pd.DataFrame) -> np.ndarray:
    """Same range-check + centering-tiebreak formula as classify_rules.py, vectorized for speed."""
    classes = lo.index.tolist()
    LO, HI = lo.values, hi.values
    WIDTH = np.where(HI - LO == 0, 0.01, HI - LO)
    CENTER = (HI + LO) / 2

    v = X_test.values[:, None, :]
    below = np.maximum(LO[None, :, :] - v, 0)
    above = np.maximum(v - HI[None, :, :], 0)
    penalty = (((below + above) / WIDTH[None, :, :]) ** 2).sum(axis=2)
    centering = (((v - CENTER[None, :, :]) / WIDTH[None, :, :]) ** 2).sum(axis=2)
    score = penalty * 1e6 + centering
    best_idx = np.argmin(score, axis=1)
    return np.array(classes)[best_idx]


def cross_validate(training_csv: str = TRAINING_DATA_CSV, class_col: str = CLASS_COL,
                    n_folds: int = N_FOLDS):
    df = pd.read_csv(training_csv)
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[class_col]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_accuracies, fold_f1s = [], []
    per_class_fold_acc = {cls: [] for cls in y.unique()}

    print(f"Running {n_folds}-fold cross-validation on {len(df)} rows, {y.nunique()} classes...\n")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_df = X_train.copy()
        train_df[class_col] = y_train.values
        lo, hi = build_ranges(train_df, class_col)

        preds = classify_vectorized(X_test, lo, hi)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        print(f"Fold {fold+1}/{n_folds}: accuracy={acc:.4f}  macro-F1={f1:.4f}")

        results = pd.DataFrame({"true": y_test.values, "pred": preds})
        for cls in per_class_fold_acc:
            sub = results[results["true"] == cls]
            if len(sub):
                per_class_fold_acc[cls].append(accuracy_score(sub["true"], sub["pred"]))

    print(f"\n{'='*70}")
    print(f"OVERALL: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f} accuracy "
          f"across {n_folds} folds  (macro-F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f})")
    print(f"{'='*70}\n")

    # Per-class stability report, sorted by how UNSTABLE each class is
    # (highest std-dev first) — the ones most likely to be misleading
    # from a single train/test split show up at the top.
    rows = []
    for cls, accs in per_class_fold_acc.items():
        if not accs:
            continue
        rows.append({
            "class": cls,
            "mean_accuracy": np.mean(accs),
            "std_dev": np.std(accs),
            "min": np.min(accs),
            "max": np.max(accs),
            "n_folds_seen": len(accs),
        })
    stability_df = pd.DataFrame(rows).sort_values("std_dev", ascending=False)

    print(f"{'Class':28s} {'Mean acc':>10s} {'Std dev':>9s} {'Range':>16s}  Reliability")
    for _, r in stability_df.iterrows():
        range_str = f"{r['min']:.0%}-{r['max']:.0%}"
        if r["std_dev"] > 0.10:
            flag = "UNSTABLE — don't trust a single number"
        elif r["mean_accuracy"] < 0.15 and r["std_dev"] < 0.05:
            flag = "stable, but consistently poor"
        else:
            flag = "stable"
        print(f"{r['class']:28s} {r['mean_accuracy']:>10.1%} {r['std_dev']:>9.1%} {range_str:>16s}  {flag}")

    stability_df.to_csv("cross_validation_stability.csv", index=False)
    print(f"\nFull results saved to cross_validation_stability.csv")

    return stability_df


if __name__ == "__main__":
    cross_validate()
