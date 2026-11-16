"""
Classify every blob image against all 18 reference classes — no manual
label mapping, no fitted/trained model.
=============================================================================
Instead of forcing you to map your folder names onto the 18 reference
groups up front, this script does the opposite: it runs every image through
the pure-statistics classification rules (see classify_rules.py) and lets
the rules decide which of the 18 known classes each defect's measurements
are closest to.

No scikit-learn model is used here — classification is a fixed distance
formula against a lookup table of per-class medians (class_rules.csv),
computed directly from the ISRA data by compute_rules.py.

Your folder names are kept only as a reference label ("what I originally
called this batch of images") — never used to constrain or check the
prediction. The output shows, for each of your folders, the full breakdown
of which reference classes its images actually got matched to.

This answers: "if I didn't know anything about ISRA's taxonomy, which of
its 18 classes does my data naturally fall into?"

REQUIREMENTS
    pip install pandas numpy pillow scikit-image

USAGE
    python classify_blob_folders.py
    (edit DATASET_ROOT below — that's the only required setting)
"""

from pathlib import Path
import pandas as pd

from classify_rules import RuleBasedClassifier
from blob_features import compute_features_for_image


# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------

DATASET_ROOT = "/home/shaivee/PycharmProjects/WIS/DATA/defects_shubham/data/defect_img"
RULES_PATH = "class_rules.csv"
CONFIDENCE_THRESHOLD = 0.35   # predictions below this get flagged as low-confidence
FAMILY_COMPOSITION_PATH = "family_composition.csv"   # optional reference baseline, set to None to skip
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
OUTPUT_CSV = "blob_structural_classification.csv"

# Set to a folder path to save a binary mask + red-tinted overlay for every
# processed image (for visual QA of segmentation) — set to None to skip.
SAVE_MASKS_TO = "masks"


# Segmentation + feature extraction now live in blob_features.py, shared
# with build_blob_dataset.py — see that file for the implementation and the
# segmentation-robustness notes.


# ---------------------------------------------------------------------------
# Main: classify every image, no manual mapping needed
# ---------------------------------------------------------------------------

def classify_folders(dataset_root: str, rules_path: str = RULES_PATH) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"DATASET_ROOT does not exist: {dataset_root}")

    clf = RuleBasedClassifier(rules_path, confidence_threshold=CONFIDENCE_THRESHOLD)

    # Optional reference baseline: the real, full-population composition of
    # each family (computed directly from the ISRA training data) — lets you
    # compare "what my images matched to" against "what's typical."
    family_baseline = None
    if FAMILY_COMPOSITION_PATH:
        try:
            fc = pd.read_csv(FAMILY_COMPOSITION_PATH)
            family_baseline = {
                fam: dict(zip(g["ClassName"], g["Percent"]))
                for fam, g in fc.groupby("Family")
            }
        except FileNotFoundError:
            pass

    rows = []
    skipped = []
    folders = sorted([f for f in dataset_root.iterdir() if f.is_dir()])
    print(f"Found {len(folders)} folders: {[f.name for f in folders]}\n")

    for folder in folders:
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
        print(f"  {folder.name:20s} ({len(images)} images)")

        for img_path in images:
            try:
                mask_dir = f"{SAVE_MASKS_TO}/{folder.name}" if SAVE_MASKS_TO else None
                feats = compute_features_for_image(img_path, save_mask_dir=mask_dir)
                result = clf.predict(feats)
                rows.append({
                    "source_folder": folder.name,       # your original label, reference only
                    "filename": img_path.name,
                    "predicted_class": result["group"],  # the model's own decision
                    "confidence": result["confidence"],
                    "high_confidence": result["accept"],
                    "best_specific_match": result.get("subclass"),         # e.g. "Dark Spot Medium" within "Dark Spot (family)"
                    "specific_match_reliability": result.get("subclass_reliability"),  # measured accuracy of this sub-step, given the family is correct
                    "runner_up_1": result["top_3"][1][0] if len(result["top_3"]) > 1 else None,
                    "runner_up_1_conf": result["top_3"][1][1] if len(result["top_3"]) > 1 else None,
                    **feats,
                })
            except Exception as e:
                skipped.append((str(img_path), str(e)))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nClassified {len(df)} images -> {OUTPUT_CSV}")
    if skipped:
        print(f"Skipped {len(skipped)} images due to errors:")
        for path, err in skipped[:10]:
            print(f"  {path}: {err}")

    # ------------------------------------------------------------------
    # The report: for each of YOUR folders, what did the model call it?
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STRUCTURAL MATCH SUMMARY — what trained class does each folder resemble?")
    print("=" * 80)

    for folder_name, group in df.groupby("source_folder"):
        n = len(group)
        counts = group["predicted_class"].value_counts()
        top_class = counts.index[0]
        top_pct = counts.iloc[0] / n
        avg_conf = group.loc[group["predicted_class"] == top_class, "confidence"].mean()

        print(f"\n'{folder_name}'  ({n} images)")
        print(f"  Best structural match: {top_class}  ({top_pct:.0%} of images, avg confidence {avg_conf:.2f})")

        # If the dominant match is a family, show EVERY specific sub-class
        # matched within it (not just the top one) — the full breakdown of
        # "which class matches most" within that family.
        top_group = group[group["predicted_class"] == top_class]
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
                    print(f"    {sub_cls:26s} {cnt:5d}  ({your_pct:5.0%} of your images  |  {baseline[sub_cls]:5.1f}% typical for this family)")
                else:
                    print(f"    {sub_cls:26s} {cnt:5d}  ({your_pct:5.0%} of your images)")

        print(f"  Full breakdown:")
        for cls, cnt in counts.items():
            print(f"    {cls:30s} {cnt:5d}  ({cnt/n:.0%})")

    # Cross-tab: folders (rows) x predicted classes (columns) — the full picture at a glance
    crosstab = pd.crosstab(df["source_folder"], df["predicted_class"])
    crosstab.to_csv("blob_structural_crosstab.csv")
    print("\n\nFull cross-tab saved to blob_structural_crosstab.csv")
    print(crosstab.to_string())

    # Which trained classes never got matched by anything in this dataset?
    all_trained = set(clf.classes)
    matched = set(df["predicted_class"].unique())
    unmatched = all_trained - matched
    if unmatched:
        print(f"\nTrained classes that NONE of your images matched to ({len(unmatched)}):")
        for c in sorted(unmatched):
            print(f"  - {c}")

    return df


if __name__ == "__main__":
    classify_folders(DATASET_ROOT, RULES_PATH)
