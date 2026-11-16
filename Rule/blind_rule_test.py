"""
Blind test: given ONLY class_rules.csv — no knowledge of the training
pipeline, the ISRA data, or how these bounds were computed — how well do
plain greater-than/less-than comparisons classify real defects?
=============================================================================
This script is deliberately self-contained. It does not import
classify_rules.py or reuse any of the earlier distance-formula code — it
treats class_rules.csv as if it just arrived as a rule book with no other
context, and implements classification with literal boundary comparisons,
exactly like a human would read the rule file and check it by hand.

THE RULE FOR EACH CLASS (18 of these, one row per class in class_rules.csv):

    a defect belongs to class C if, for every one of its 4 features:
        class_rules[C]['<feature>_lo']  <=  defect['<feature>']  <=  class_rules[C]['<feature>_hi']

If a defect's values fall inside every one of a class's 4 ranges, it's a
clean match for that class. If it falls inside more than one class's
ranges (or none), a scoring rule breaks the tie (see score_class below) —
but the core check on every single feature is a plain >= and <=.

USAGE
    python blind_rule_test.py
    (expects class_rules.csv and a test CSV with the 4 feature columns +
     a true_label column in the same folder)
"""

import csv
import math


RULES_FILE = "class_rules.csv"
TEST_FILE = "rules_test_set.csv"
FEATURES = ["aspect_ratio", "PERC_BRIGHT", "PERC_DARK", "A_to_roi_ratio"]


def load_rules(path):
    """Read class_rules.csv into a plain dict: {class_name: {feature: (lo, hi)}}."""
    rules = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row["class"]
            rules[class_name] = {}
            for feat in FEATURES:
                lo = float(row[f"{feat}_lo"])
                hi = float(row[f"{feat}_hi"])
                rules[class_name][feat] = (lo, hi)
    return rules


def load_test_rows(path):
    """Read the test CSV into a list of dicts: features + true_label."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {feat: float(row[feat]) for feat in FEATURES}
            parsed["true_label"] = row["true_label"]
            rows.append(parsed)
    return rows


def fits_inside(value, lo, hi):
    """The literal rule: is this value between the class's lower and upper bound?"""
    return lo <= value <= hi


def score_class(defect, class_ranges):
    """
    For one class, check every feature against its (lo, hi) range with
    plain comparisons. Returns (n_features_fit_inside, total_distance).

    n_features_fit_inside : how many of the 4 features passed the
        greater-than/less-than check outright (4 = perfect match on
        every feature).
    total_distance : for any feature that DIDN'T fit inside its range,
        how far outside it fell, scaled by the range's own width — used
        only to break ties between classes that fit equally well.
    """
    n_fit = 0
    total_distance = 0.0
    for feat in FEATURES:
        lo, hi = class_ranges[feat]
        value = defect[feat]
        width = hi - lo if hi > lo else 0.01

        if fits_inside(value, lo, hi):
            n_fit += 1
        elif value < lo:
            total_distance += ((lo - value) / width) ** 2
        else:  # value > hi
            total_distance += ((value - hi) / width) ** 2

    return n_fit, total_distance


def classify(defect, rules):
    """
    Check the defect against every one of the 18 classes' rules. The best
    match is whichever class the defect fits inside the MOST features for
    (n_fit, highest first); ties broken by the smallest total_distance
    (closest overall, even on the features that didn't fit).
    """
    best_class = None
    best_score = (-1, math.inf)   # (n_fit, total_distance) — want n_fit high, distance low

    for class_name, class_ranges in rules.items():
        n_fit, total_distance = score_class(defect, class_ranges)
        score = (n_fit, -total_distance)   # negate distance so "higher is better" sorts correctly
        if score > best_score:
            best_score = score
            best_class = class_name

    return best_class


def run_blind_test(rules_file=RULES_FILE, test_file=TEST_FILE):
    rules = load_rules(rules_file)
    test_rows = load_test_rows(test_file)

    print(f"Loaded {len(rules)} class rules from {rules_file} (18 classes expected).")
    print(f"Loaded {len(test_rows)} real ISRA test defects from {test_file}.\n")

    correct = 0
    per_class_total = {}
    per_class_correct = {}
    confusion_examples = {}   # (true, predicted) -> count, for the biggest mistakes

    for defect in test_rows:
        true_label = defect["true_label"]
        predicted = classify(defect, rules)

        per_class_total[true_label] = per_class_total.get(true_label, 0) + 1
        if predicted == true_label:
            correct += 1
            per_class_correct[true_label] = per_class_correct.get(true_label, 0) + 1
        else:
            key = (true_label, predicted)
            confusion_examples[key] = confusion_examples.get(key, 0) + 1

    accuracy = correct / len(test_rows)
    print(f"{'='*70}")
    print(f"BLIND RULE-BOOK ACCURACY: {accuracy:.4f}  ({correct}/{len(test_rows)} correct)")
    print(f"{'='*70}\n")

    print(f"{'Class':30s} {'Accuracy':>10s} {'n':>8s}")
    for cls in sorted(per_class_total, key=lambda c: per_class_correct.get(c, 0) / per_class_total[c], reverse=True):
        n = per_class_total[cls]
        n_correct = per_class_correct.get(cls, 0)
        print(f"{cls:30s} {n_correct/n:>10.1%} {n:>8d}")

    print(f"\nTop 10 most common mistakes (true class -> what it got classified as):")
    top_mistakes = sorted(confusion_examples.items(), key=lambda kv: -kv[1])[:10]
    for (true_cls, pred_cls), count in top_mistakes:
        print(f"  {true_cls:26s} -> {pred_cls:26s}  ({count} times)")

    return accuracy


if __name__ == "__main__":
    run_blind_test()
