"""
Build a labeled test dataset from folders of blob images.
============================================================
Turns a directory of class-named folders full of defect blob images into a
single CSV of model-ready features, ready to evaluate against
`blob_model_v1.pkl` / `predict_defect.py`.

Expected folder structure:

    dataset_root/
        bright_spot/
            img001.png
            img002.jpg
            ...
        dark_spot/
            ...
        hole/
            ...
        ... (however many folders you have)

Images within and across folders can be ANY size — the whole point of the
4-feature blob-ready model is that every feature is scale-free (a ratio),
so a 20x20px blob and a 300x300px blob are handled identically. Nothing
here assumes a fixed image size.

Each image should be a cropped "blob" — the defect plus a visible margin of
surrounding background (not a tight crop with zero background, or brightness
comparison has nothing to measure against).

REQUIREMENTS
    pip install pillow numpy scikit-image pandas

USAGE
    python build_blob_dataset.py

    (edit FOLDER_TO_GROUP and DATASET_ROOT below first)
"""

from pathlib import Path
import pandas as pd

from blob_features import compute_features_for_image


# ---------------------------------------------------------------------------
# STEP 0 — CONFIGURE THIS FOR YOUR DATA
# ---------------------------------------------------------------------------

# Where your class-named folders live.
DATASET_ROOT = "path/to/your/blob_folders"

# Map YOUR folder names (left) to the model's group labels (right).
# The right-hand values MUST exactly match one of the model's 18 classes:
#   'Bright Spot (family)', 'Dark Spot (family)', 'Hole (family)', 'Hole Small',
#   'Streak/Scratch (family)', 'Edge Crack (family)', 'Wrinkle Long', 'Splice',
#   'Dirt or Debris', 'Punching', 'Lump', 'Dot', 'Slime Spot',
#   'Minor Surface Anomaly', 'Tranv.Crack', 'Ripple',
#   'Floating Debris Small', 'Floating Debris'
FOLDER_TO_GROUP = {
    "bright_spot": "Bright Spot (family)",
    "dark_spot": "Dark Spot (family)",
    "hole": "Hole (family)",
    "scratch": "Streak/Scratch (family)",
    "wrinkle": "Wrinkle Long",
    "splice": "Splice",
    # <-- edit/add lines above to match your actual 6 folder names
}

OUTPUT_CSV = "blob_test_dataset.csv"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Set to a folder path to save a binary mask + red-tinted overlay for every
# processed image (for visual QA of segmentation) — set to None to skip.
SAVE_MASKS_TO = "masks"


# Segmentation + feature extraction now live in blob_features.py, shared
# with classify_blob_folders.py so both scripts always use the same logic
# (and the same bugfixes) rather than two copies drifting apart.


# ---------------------------------------------------------------------------
# STEP 3 — walk the folders and build the dataset
# ---------------------------------------------------------------------------

def build_test_dataset(dataset_root: str, output_csv: str = OUTPUT_CSV) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"DATASET_ROOT does not exist: {dataset_root}")

    rows = []
    skipped = []

    folders = sorted([f for f in dataset_root.iterdir() if f.is_dir()])
    print(f"Found {len(folders)} folders: {[f.name for f in folders]}\n")

    for folder in folders:
        true_label = FOLDER_TO_GROUP.get(folder.name)
        if true_label is None:
            print(f"  SKIPPING folder '{folder.name}' — not in FOLDER_TO_GROUP mapping.")
            continue

        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
        print(f"  {folder.name:20s} -> {true_label:30s} ({len(images)} images)")

        for img_path in images:
            try:
                mask_dir = f"{SAVE_MASKS_TO}/{folder.name}" if SAVE_MASKS_TO else None
                feats = compute_features_for_image(img_path, save_mask_dir=mask_dir)
                feats["true_label"] = true_label
                feats["source_folder"] = folder.name
                feats["filename"] = img_path.name
                feats["filepath"] = str(img_path)
                rows.append(feats)
            except Exception as e:
                skipped.append((str(img_path), str(e)))

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"\nBuilt dataset: {len(df)} images -> {output_csv}")
    if skipped:
        print(f"Skipped {len(skipped)} images due to errors:")
        for path, err in skipped[:10]:
            print(f"  {path}: {err}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    if len(df):
        print("\nClass distribution:")
        print(df["true_label"].value_counts().to_string())
        print("\nBlob size range (sanity check — confirms varying sizes handled fine):")
        print(f"  width  px: {df['blob_width_px'].min()} - {df['blob_width_px'].max()}")
        print(f"  height px: {df['blob_height_px'].min()} - {df['blob_height_px'].max()}")

    return df


if __name__ == "__main__":
    build_test_dataset(DATASET_ROOT, OUTPUT_CSV)
