"""
Evaluate the pure-statistics range-based classification rules on held-out
test data.
=============================================================================
No model is loaded or fit here — just the range lookup table
(class_rules.csv) and the fixed distance formula in classify_rules.py,
scored against data that was never used to compute the ranges.

USAGE
    python evaluate_rules.py
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from classify_rules import RuleBasedClassifier, FEATURE_COLS


def evaluate(test_csv: str = "rules_test_set.csv", rules_csv: str = "class_rules.csv"):
    clf = RuleBasedClassifier(rules_csv)
    df = pd.read_csv(test_csv)

    preds = []
    for _, row in df.iterrows():
        features = {c: row[c] for c in FEATURE_COLS}
        result = clf.predict(features)
        preds.append(result["group"])

    df["predicted"] = preds
    acc = accuracy_score(df["true_label"], df["predicted"])
    f1 = f1_score(df["true_label"], df["predicted"], average="macro")

    print(f"Rule-based (range) accuracy: {acc:.4f}")
    print(f"Macro-F1:                    {f1:.4f}\n")

    report = pd.DataFrame(
        classification_report(df["true_label"], df["predicted"], output_dict=True, zero_division=0)
    ).T
    report = report[report.index.isin(df["true_label"].unique())].sort_values("f1-score", ascending=False)
    print(report[["precision", "recall", "f1-score", "support"]].round(3).to_string())
    report.to_csv("rules_test_report.csv")

    return df


if __name__ == "__main__":
    evaluate()
