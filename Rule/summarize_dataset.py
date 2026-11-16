"""
Summarize the original blob dataset: which classes are present, how many
defects in each, and whether each one was actually covered by the trained
model.
=============================================================================
Answers three questions directly:
  1. What classes (folders) exist in the original dataset?
  2. How many defect images are in each?
  3. Was each class actually mapped to one of the 18 reference groups,
     or silently skipped (unmapped folder names never get tested)?

Works straight from the raw folder structure — so it catches folders that
build_blob_dataset.py silently skipped because they weren't in
FOLDER_TO_GROUP, not just the ones that made it into the CSV.

REQUIREMENTS
    pip install pandas

USAGE
    python summarize_dataset.py
    (edit DATASET_ROOT and FOLDER_TO_GROUP below to match your setup —
     copy them from build_blob_dataset.py so both scripts agree)
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIGURE — keep these identical to what you used in build_blob_dataset.py
# ---------------------------------------------------------------------------

DATASET_ROOT = "path/to/your/blob_folders"

FOLDER_TO_GROUP = {
    "bright_spot": "Bright Spot (family)",
    "dark_spot": "Dark Spot (family)",
    "hole": "Hole (family)",
    "scratch": "Streak/Scratch (family)",
    "wrinkle": "Wrinkle Long",
    "splice": "Splice",
    # <-- must match build_blob_dataset.py exactly
}

RULES_PATH = "class_rules.csv"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def summarize_dataset(dataset_root: str, rules_path: str = RULES_PATH) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"DATASET_ROOT does not exist: {dataset_root}")

    # Pull the reference class list from the rules table, so we can flag
    # folders mapped to something not in it (a typo, etc.)
    rules = pd.read_csv(rules_path, index_col=0)
    trained_classes = set(rules.index)

    rows = []
    folders = sorted([f for f in dataset_root.iterdir() if f.is_dir()])

    for folder in folders:
        n_images = len([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
        mapped_group = FOLDER_TO_GROUP.get(folder.name)

        if mapped_group is None:
            status = "NOT MAPPED — skipped entirely, never tested"
            in_trained_classes = False
        elif mapped_group not in trained_classes:
            status = f"MAPPED to '{mapped_group}' — but that's NOT a trained class (check spelling)"
            in_trained_classes = False
        else:
            status = "OK — mapped and covered by the classification rules"
            in_trained_classes = True

        rows.append({
            "folder_name": folder.name,
            "n_defects": n_images,
            "mapped_to_group": mapped_group or "(none)",
            "in_trained_classes": in_trained_classes,
            "status": status,
        })

    df = pd.DataFrame(rows).sort_values("n_defects", ascending=False).reset_index(drop=True)

    total = df["n_defects"].sum()
    covered = df.loc[df["in_trained_classes"], "n_defects"].sum()
    uncovered = total - covered

    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    print(f"Folders found: {len(df)}")
    print(f"Total defect images: {total}")
    print(f"  -> covered by the classification rules: {covered} ({covered/total:.1%})" if total else "")
    print(f"  -> NOT covered (unmapped/unknown): {uncovered} ({uncovered/total:.1%})" if total else "")
    print("=" * 80)
    print()
    print(df.to_string(index=False))

    if uncovered > 0:
        print("\n⚠ Some folders are not covered by the classification rules.")
        print("  These defects will not be testable until either:")
        print("  (a) you add a mapping for them in FOLDER_TO_GROUP, if they")
        print("      correspond to one of the 18 reference groups, or")
        print("  (b) they're a genuinely new defect type the model has never")
        print("      seen — there are no rules defined for it.")

    # Also show which of the 18 reference groups have NO representation
    # at all in this dataset — useful to know what you can't currently test.
    mapped_groups_present = set(df.loc[df["in_trained_classes"], "mapped_to_group"])
    missing_from_dataset = trained_classes - mapped_groups_present
    if missing_from_dataset:
        print(f"\nReference classes with ZERO images in your dataset ({len(missing_from_dataset)}):")
        for g in sorted(missing_from_dataset):
            print(f"  - {g}")

    df.to_csv("dataset_coverage_summary.csv", index=False)
    print("\nSaved to dataset_coverage_summary.csv")

    return df


if __name__ == "__main__":
    summarize_dataset(DATASET_ROOT, RULES_PATH)
