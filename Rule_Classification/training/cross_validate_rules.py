"""
TRAINING PIPELINE: 5-fold stratified cross-validation of the range rules.

For each fold the rules are rebuilt from the other four folds and used to
classify the held-out fold. This shows whether an accuracy figure depends on
which rows happened to be held out: a class whose accuracy swings between
folds has too few examples for its figure to be trusted.

Run:  python cross_validate_rules.py     (writes output/cross_validation_results.csv)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from build_rules import (OUTPUT_DIR, TRAINING_DATA, GROUP_COL, RANDOM_STATE,
                         percentile_rules, bounds)
from rule_scoring import FEATURE_COLS, range_scores, best_class_index

N_FOLDS = 5


def main():
    df = pd.read_csv(TRAINING_DATA)
    X, y = df[FEATURE_COLS].to_numpy(), df[GROUP_COL].to_numpy()
    classes = sorted(set(y))

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    overall, per_class = [], {}
    for k, (train_idx, test_idx) in enumerate(folds.split(X, y), start=1):
        rules = percentile_rules(df.iloc[train_idx], GROUP_COL)
        penalty, centering = range_scores(X[test_idx], *bounds(rules))
        predicted = rules.index.to_numpy()[best_class_index(penalty, centering)]
        truth = y[test_idx]
        overall.append((float((predicted == truth).mean()), f1_score(truth, predicted, average="macro")))
        per_class[f"fold_{k}"] = {c: float((predicted[truth == c] == c).mean()) for c in classes}
        print(f"fold {k}: accuracy {overall[-1][0]:.4f}   macro-F1 {overall[-1][1]:.4f}")

    accuracy = np.array([a for a, _ in overall])
    print(f"\noverall accuracy {accuracy.mean():.4f} +/- {accuracy.std():.4f} across {N_FOLDS} folds\n")

    table = pd.DataFrame(per_class)
    table["mean"] = table.mean(axis=1)
    table["std"] = table[list(per_class)].std(axis=1, ddof=0)
    table = table.sort_values("std", ascending=False)
    table.index.name = "class"
    OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_DIR / "cross_validation_results.csv")
    print((table * 100).round(1).to_string())


if __name__ == "__main__":
    main()
