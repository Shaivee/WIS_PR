"""
End-to-end: YOLO-labeled images -> defect crops -> features -> classification
against the 18 reference classes. No auto-segmentation anywhere.
=============================================================================
This is extract_from_yolo_labels.py + classify_rules.py chained together,
with the same rich reporting classify_blob_folders.py gives you (per-class
breakdown, sub-class matching within families, comparison against the real
family composition baseline) — but starting from your actual raw images
and YOLO labels instead of pre-cropped blob folders.

REQUIREMENTS
    pip install pillow numpy pandas

USAGE
    python run_yolo_pipeline.py
    (edit IMAGES_DIR, LABELS_DIR, OUTPUT_ROOT, CLASSES in
     extract_from_yolo_labels.py first)
"""

import pandas as pd
from extract_from_yolo_labels import run_extraction, CLASSES
from classify_rules import RuleBasedClassifier

RULES_PATH = "class_rules.csv"
CONFIDENCE_THRESHOLD = 0.35
FAMILY_COMPOSITION_PATH = "family_composition.csv"
OUTPUT_CSV = "yolo_classified.csv"


def run_pipeline():
    # Step 1: extract crops + compute features directly from YOLO labels
    print("=" * 80)
    print("STEP 1: Extracting defect crops from YOLO labels")
    print("=" * 80)
    features_df = run_extraction()

    if len(features_df) == 0:
        print("\nNo defects extracted — check IMAGES_DIR/LABELS_DIR and try again.")
        return features_df

    # Step 2: classify every extracted defect against the 18 reference classes
    print("\n" + "=" * 80)
    print("STEP 2: Classifying against the 18 reference classes")
    print("=" * 80)

    clf = RuleBasedClassifier(RULES_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)

    family_baseline = None
    try:
        fc = pd.read_csv(FAMILY_COMPOSITION_PATH)
        family_baseline = {fam: dict(zip(g["ClassName"], g["Percent"])) for fam, g in fc.groupby("Family")}
    except FileNotFoundError:
        pass

    predicted, confidences, accepted, subclasses, sub_reliability = [], [], [], [], []
    for _, row in features_df.iterrows():
        feats = {c: row[c] for c in clf.feature_names}
        result = clf.predict(feats)
        predicted.append(result["group"])
        confidences.append(result["confidence"])
        accepted.append(result["accept"])
        subclasses.append(result.get("subclass"))
        sub_reliability.append(result.get("subclass_reliability"))

    features_df["predicted_reference_class"] = predicted
    features_df["confidence"] = confidences
    features_df["high_confidence"] = accepted
    features_df["best_specific_match"] = subclasses
    features_df["specific_match_reliability"] = sub_reliability
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Full results saved to {OUTPUT_CSV}\n")

    # Step 3: report — for each of YOUR real classes, what reference class does it match?
    print("=" * 80)
    print("YOUR CLASSES vs. THE 18 REFERENCE CLASSES")
    print("=" * 80)

    for class_name, group in features_df.groupby("class_name"):
        n = len(group)
        counts = group["predicted_reference_class"].value_counts()
        top_class = counts.index[0]
        top_pct = counts.iloc[0] / n
        avg_conf = group.loc[group["predicted_reference_class"] == top_class, "confidence"].mean()

        print(f"\n'{class_name}'  ({n} defects, class_id {[k for k,v in CLASSES.items() if v==class_name][0]})")
        print(f"  Best structural match: {top_class}  ({top_pct:.0%} of defects, avg confidence {avg_conf:.2f})")

        top_group = group[group["predicted_reference_class"] == top_class]
        if top_group["best_specific_match"].notna().any():
            sub_counts = top_group["best_specific_match"].value_counts()
            reliability = top_group["specific_match_reliability"].dropna().iloc[0] if top_group["specific_match_reliability"].notna().any() else None
            rel_str = f" (right ~{reliability:.0%} of the time when the family is correct)" if reliability else ""
            print(f"  Specific-class breakdown within '{top_class}'{rel_str}:")
            n_sub = len(top_group)
            baseline = family_baseline.get(top_class) if family_baseline else None
            for sub_cls, cnt in sub_counts.items():
                your_pct = cnt / n_sub
                if baseline and sub_cls in baseline:
                    print(f"    {sub_cls:26s} {cnt:5d}  ({your_pct:5.0%} of yours  |  {baseline[sub_cls]:5.1f}% typical)")
                else:
                    print(f"    {sub_cls:26s} {cnt:5d}  ({your_pct:5.0%} of yours)")

        print(f"  Full breakdown across all reference classes:")
        for cls, cnt in counts.items():
            print(f"    {cls:30s} {cnt:5d}  ({cnt/n:.0%})")

    crosstab = pd.crosstab(features_df["class_name"], features_df["predicted_reference_class"])
    crosstab.to_csv("yolo_classes_vs_reference_crosstab.csv")
    print(f"\n\nFull cross-tab saved to yolo_classes_vs_reference_crosstab.csv")
    print(crosstab.to_string())

    return features_df


if __name__ == "__main__":
    run_pipeline()
