"""
Step 2 of 2 — Train the blob-ready defect classifier.
=============================================================================
Trains the Random Forest on the 4 calibration-free features (aspect_ratio,
PERC_BRIGHT, PERC_DARK, A_to_roi_ratio), evaluates on a held-out test split,
and saves the model bundle used by predict_defect.py.

Run this AFTER extract_training_data.py.

REQUIREMENTS
    pip install pandas numpy scikit-learn

USAGE
    python train_blob_model.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
TRAINING_DATA_CSV = "training_data.csv"
MODEL_OUTPUT_PATH = "blob_model_v1.pkl"
TEST_REPORT_OUTPUT = "final_model_test_report.csv"

# The 4 features computable from a plain, uncalibrated blob image — see
# ISRA_Blob_Ready_Model.html for why these 4 specifically (everything else
# needs either physical-size calibration or camera metadata a standalone
# blob image doesn't carry).
FEATURE_COLS = ["aspect_ratio", "PERC_BRIGHT", "PERC_DARK", "A_to_roi_ratio"]

RECOMMENDED_CONFIDENCE_THRESHOLD = 0.5   # see the coverage/accuracy tradeoff table this script prints

RANDOM_STATE = 42
N_ESTIMATORS = 100     # reduced from 300 during tuning: identical accuracy, 1/3 the file size
MAX_DEPTH = 14
MIN_SAMPLES_LEAF = 15


def train_model(training_csv: str = TRAINING_DATA_CSV, model_output: str = MODEL_OUTPUT_PATH):
    df = pd.read_csv(training_csv)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}. "
                          f"Did you run extract_training_data.py first?")

    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["Stage1aGroup"]

    # 60/20/20 train/validation/test split, stratified so every class is
    # represented proportionally in all three splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Train: {X_train.shape}   Val: {X_val.shape}   Test: {X_test.shape}\n")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",   # rare classes (Punching, Ripple, etc.) don't get drowned out
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Validation check (used during development to sanity-check hyperparameters
    # before ever touching the test set)
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"Validation accuracy: {val_acc:.4f}")

    # Final held-out test evaluation — untouched until this point
    pred_test = clf.predict(X_test)
    proba_test = clf.predict_proba(X_test)
    test_acc = accuracy_score(y_test, pred_test)
    test_f1 = f1_score(y_test, pred_test, average="macro")
    print(f"Test accuracy:       {test_acc:.4f}")
    print(f"Test macro-F1:       {test_f1:.4f}\n")

    report = pd.DataFrame(
        classification_report(y_test, pred_test, output_dict=True, zero_division=0)
    ).T
    report = report[report.index.isin(y.unique())].sort_values("f1-score", ascending=False)
    print(report[["precision", "recall", "f1-score", "support"]].round(3).to_string())
    report.to_csv(TEST_REPORT_OUTPUT)

    # Confidence threshold / coverage tradeoff — measured, not assumed
    conf = proba_test.max(axis=1)
    pred_labels = clf.classes_[np.argmax(proba_test, axis=1)]
    threshold_table = []
    print(f"\n{'threshold':>10} {'coverage':>10} {'accuracy':>10}")
    for t in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:
        keep = conf >= t
        cov = keep.mean()
        acc = accuracy_score(y_test[keep], pred_labels[keep]) if keep.sum() > 0 else float("nan")
        threshold_table.append((t, cov, acc))
        print(f"{t:>10.1f} {cov:>10.1%} {acc:>10.1%}")

    # Save the full bundle predict_defect.py expects
    with open(model_output, "wb") as f:
        pickle.dump({
            "model": clf,
            "feature_names": FEATURE_COLS,
            "feature_order": FEATURE_COLS,
            "classes": list(clf.classes_),
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "recommended_confidence_threshold": RECOMMENDED_CONFIDENCE_THRESHOLD,
            "threshold_table": threshold_table,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    import os
    size_mb = os.path.getsize(model_output) / 1e6
    print(f"\nModel saved to {model_output} ({size_mb:.1f} MB)")

    return clf


if __name__ == "__main__":
    train_model()
