"""
TRAINING PIPELINE: build the rule tables from training_data.csv.

    output/class_rules.csv     per class: the 25th and 75th percentile of each parameter
    output/subclass_rules.csv  the same for each specific ISRA class inside a family,
                               plus the measured accuracy of that within-family match

The rules use a 60% training split only. A 20% test split, never used to build
the rules, gives the accuracy printed at the end. (training_data.csv is split
60 / 20 / 20; the middle 20% is unused and exists only so the test rows are
the same ones the reported accuracy was measured on.)

Run:  python build_rules.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from rule_scoring import FEATURE_COLS, range_scores, best_class_index

HERE = Path(__file__).resolve().parent
TRAINING_DATA = HERE / "training_data.csv"
OUTPUT_DIR = HERE / "output"
GROUP_COL = "Stage1aGroup"   # the 18 classes the rules are built for
CLASS_COL = "Class"          # original ISRA class id
LOWER_Q, UPPER_Q = 0.25, 0.75
RANDOM_STATE = 42

# Specific ISRA classes (by ISRA class id) inside each family.
FAMILIES = {
    "Bright Spot (family)": {30: "Bright Spot Small", 31: "Bright Spot Medium", 32: "Bright Spot Large",
                             33: "Bright Spot Extra Large", 39: "Periodic Bright Spot"},
    "Dark Spot (family)": {50: "Dark Spot Small", 51: "Dark Spot Medium", 52: "Dark Spot Large",
                           53: "Dark Spot Extra Large", 59: "Periodic Dark Spot"},
    "Hole (family)": {21: "Hole Medium", 22: "Hole Large", 23: "Hole Extra Large", 29: "Periodic Hole"},
    "Streak/Scratch (family)": {80: "Scratch", 81: "Streak Bright", 82: "Streak Dark"},
    "Edge Crack (family)": {60: "Edge Crack", 62: "Deep Edge Crack"},
    "Minor Surface Anomaly": {200: "Water Splash", 180: "Coating Skip"},
}


def split_train_test(df):
    y = df[GROUP_COL]
    train_idx, rest_idx = train_test_split(df.index, test_size=0.4, random_state=RANDOM_STATE, stratify=y)
    _, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=RANDOM_STATE, stratify=y.loc[rest_idx])
    return df.loc[train_idx], df.loc[test_idx]


def percentile_rules(df, by):
    """Lower and upper bound of each parameter for every value of the column `by`."""
    grouped = df.groupby(by)[FEATURE_COLS]
    return grouped.quantile(LOWER_Q).join(grouped.quantile(UPPER_Q), lsuffix="_lo", rsuffix="_hi")


def bounds(rules):
    return (rules[[f"{c}_lo" for c in FEATURE_COLS]].to_numpy(),
            rules[[f"{c}_hi" for c in FEATURE_COLS]].to_numpy())


def build_subclass_rules(train, test):
    name_of = {cid: name for members in FAMILIES.values() for cid, name in members.items()}
    family_of = {name: family for family, members in FAMILIES.items() for name in members.values()}

    def with_names(df):
        df = df[df[CLASS_COL].isin(name_of)]
        return df.assign(ClassName=df[CLASS_COL].map(name_of))

    train, test = with_names(train), with_names(test)
    rules = percentile_rules(train, "ClassName")
    rules["Family"] = rules.index.map(family_of)

    # Accuracy of the within-family match, given the family is already right.
    reliability = {}
    for family, members in rules.groupby("Family"):
        rows = test[test["ClassName"].map(family_of) == family]
        penalty, centering = range_scores(rows[FEATURE_COLS].to_numpy(), *bounds(members))
        predicted = members.index.to_numpy()[best_class_index(penalty, centering)]
        reliability[family] = float((predicted == rows["ClassName"].to_numpy()).mean())
    rules["Reliability"] = rules["Family"].map(reliability)
    rules.index.name = "ClassName"
    return rules, reliability


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(TRAINING_DATA)
    train, test = split_train_test(df)
    print(f"{len(df)} rows: {len(train)} to build the rules, {len(test)} held out for testing")

    class_rules = percentile_rules(train, GROUP_COL)
    class_rules.index.name = "class"
    class_rules.to_csv(OUTPUT_DIR / "class_rules.csv")

    penalty, centering = range_scores(test[FEATURE_COLS].to_numpy(), *bounds(class_rules))
    predicted = class_rules.index.to_numpy()[best_class_index(penalty, centering)]
    correct = int((predicted == test[GROUP_COL].to_numpy()).sum())
    print(f"output/class_rules.csv written: {len(class_rules)} classes")
    print(f"Held-out accuracy: {correct / len(test):.2%} ({correct} of {len(test)})")

    subclass_rules, reliability = build_subclass_rules(train, test)
    subclass_rules.to_csv(OUTPUT_DIR / "subclass_rules.csv")
    print(f"output/subclass_rules.csv written: {len(subclass_rules)} specific classes")
    for family, value in sorted(reliability.items(), key=lambda kv: -kv[1]):
        print(f"  within-family accuracy, {family}: {value:.1%}")


if __name__ == "__main__":
    main()
