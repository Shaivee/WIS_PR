"""
Classify every blob image against all 18 trained classes — no manual
label mapping required.
=============================================================================
Instead of forcing you to map your folder names onto the model's 18 trained
groups up front, this script does the opposite: it runs every image through
the trained model and lets the model decide which of its 18 known classes
each defect structurally resembles most.

Your folder names are kept only as a reference label ("what I originally
called this batch of images") — never used to constrain or check the
prediction. The output shows, for each of your folders, the full breakdown
of which trained classes its images actually got matched to.

This answers: "if I didn't know anything about ISRA's taxonomy, which of
its 18 classes does my data naturally fall into?"

REQUIREMENTS
    pip install pandas numpy pillow scikit-image scikit-learn

USAGE
    python classify_blob_folders.py
    (edit DATASET_ROOT below — that's the only required setting)
"""

from pathlib import Path
import pandas as pd

from predict_defect import BlobDefectClassifier
from blob_features import compute_features_for_image


# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------

DATASET_ROOT = "/home/shaivee/PycharmProjects/WIS/DATA/defects_shubham/data/defect_img"
MODEL_PATH = "blob_model_v1.pkl"
CONFIDENCE_THRESHOLD = 0.5   # predictions below this get flagged as low-confidence
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
OUTPUT_CSV = "blob_structural_classification.csv"


# Segmentation + feature extraction now live in blob_features.py, shared
# with build_blob_dataset.py — see that file for the implementation and the
# segmentation-robustness notes.


# ---------------------------------------------------------------------------
# Main: classify every image, no manual mapping needed
# ---------------------------------------------------------------------------

def classify_folders(dataset_root: str, model_path: str = MODEL_PATH) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"DATASET_ROOT does not exist: {dataset_root}")

    clf = BlobDefectClassifier(model_path, confidence_threshold=CONFIDENCE_THRESHOLD)

    rows = []
    skipped = []
    folders = sorted([f for f in dataset_root.iterdir() if f.is_dir()])
    print(f"Found {len(folders)} folders: {[f.name for f in folders]}\n")

    for folder in folders:
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
        print(f"  {folder.name:20s} ({len(images)} images)")

        for img_path in images:
            try:
                feats = compute_features_for_image(img_path)
                result = clf.predict(feats)
                rows.append({
                    "source_folder": folder.name,       # your original label, reference only
                    "filename": img_path.name,
                    "predicted_class": result["group"],  # the model's own decision
                    "confidence": result["confidence"],
                    "high_confidence": result["accept"],
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
    classify_folders(DATASET_ROOT, MODEL_PATH)
