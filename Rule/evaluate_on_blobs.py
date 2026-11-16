"""
Evaluate the trained blob model against a real labeled blob dataset.
========================================================================
Takes the CSV produced by build_blob_dataset.py and scores it against the
pure-statistics rules in class_rules.csv — accuracy, per-class precision/recall/F1, confusion
matrix, and the confidence-threshold coverage/accuracy tradeoff, computed
on YOUR images rather than the original ISRA held-out test set.

This is the real validation step: the numbers in README_blob_model.md were
measured on ISRA CSV data. Your imaging setup, lighting, and segmentation
are different, so accuracy here may differ — that's expected and worth
watching, not a failure.

REQUIREMENTS
    pip install pandas scikit-learn

USAGE
    python evaluate_on_blobs.py blob_test_dataset.csv
"""

import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)
from classify_rules import RuleBasedClassifier


def evaluate(dataset_csv: str, rules_path: str = "class_rules.csv",
             confidence_threshold: float = 0.35):
    clf = RuleBasedClassifier(rules_path, confidence_threshold=confidence_threshold)
    df = pd.read_csv(dataset_csv)

    if "true_label" not in df.columns:
        raise ValueError("Dataset must have a 'true_label' column — "
                          "use build_blob_dataset.py to create it correctly.")

    missing = [c for c in clf.feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {missing}")

    # Run every row through the model
    preds, confs, accepts = [], [], []
    for _, row in df.iterrows():
        feats = {c: row[c] for c in clf.feature_names}
        result = clf.predict(feats)
        preds.append(result["group"])
        confs.append(result["confidence"])
        accepts.append(result["accept"])

    df["predicted_label"] = preds
    df["confidence"] = confs
    df["accepted"] = accepts

    # --- Overall metrics (every prediction, no threshold) ---
    acc = accuracy_score(df["true_label"], df["predicted_label"])
    macro_f1 = f1_score(df["true_label"], df["predicted_label"], average="macro", zero_division=0)

    print("=" * 70)
    print(f"Dataset:            {dataset_csv}  ({len(df)} images)")
    print(f"Overall accuracy:    {acc:.1%}  (no confidence threshold)")
    print(f"Macro-F1:            {macro_f1:.1%}")
    print("=" * 70)

    # --- Per-class report ---
    print("\nPer-class precision / recall / F1:\n")
    report = classification_report(df["true_label"], df["predicted_label"], zero_division=0)
    print(report)

    # --- Confusion matrix ---
    labels = sorted(set(df["true_label"]) | set(df["predicted_label"]))
    cm = confusion_matrix(df["true_label"], df["predicted_label"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion matrix (rows = true label, columns = predicted):\n")
    print(cm_df.to_string())

    # --- Confidence threshold tradeoff, measured on YOUR data ---
    print(f"\n{'threshold':>10} {'coverage':>10} {'accuracy':>10}")
    for t in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:
        keep = df["confidence"] >= t
        cov = keep.mean()
        a = accuracy_score(df.loc[keep, "true_label"], df.loc[keep, "predicted_label"]) if keep.sum() else float("nan")
        print(f"{t:>10.1f} {cov:>10.1%} {a:>10.1%}")

    # --- Save annotated results ---
    out_path = dataset_csv.replace(".csv", "_evaluated.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull results (with predictions) saved to {out_path}")

    # --- Flag the biggest confusions, for a quick look at what's going wrong ---
    wrong = df[df["true_label"] != df["predicted_label"]]
    if len(wrong):
        print("\nMost common misclassifications (true -> predicted):")
        confusion_pairs = wrong.groupby(["true_label", "predicted_label"]).size().sort_values(ascending=False)
        print(confusion_pairs.head(10).to_string())

    return df


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "blob_test_dataset.csv"
    evaluate(csv_path)
