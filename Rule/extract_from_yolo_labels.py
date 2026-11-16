"""
Extract defect crops from YOLO-labeled images and compute model-ready
features — no auto-segmentation (Otsu) anywhere in this pipeline.
=============================================================================
Replaces the earlier Otsu / mask-based approach entirely. Your labels
already tell us exactly where each defect is (a YOLO bounding box) — so
instead of guessing where the blob is via thresholding, we use the
annotation directly, and get a more reliable feature set as a result:

    aspect_ratio     : computed directly from the YOLO box dimensions
                        (box_height / box_width) — exact, ground-truth,
                        no segmentation guesswork at all.

    PERC_BRIGHT       : compared against BACKGROUND STATISTICS computed from
    PERC_DARK           the REST OF THE SOURCE IMAGE — every pixel that
                        isn't inside any labeled box in that image. This is
                        real background, not a guess, and doesn't require
                        padding the saved crop.

    A_to_roi_ratio    : the proportion of pixels INSIDE the box that are
                        meaningfully brighter or darker than that real
                        background reference — i.e. how much of the
                        annotated box is actually "different from the
                        material" vs. incidental surrounding texture
                        that happened to be included in the box. This is
                        the closest honest equivalent to "solidity" you
                        can get without a true pixel mask.

WHY THIS IS BETTER THAN OTSU
    Otsu had to guess, blind, on every single image, with no ground truth
    to check against — and it failed unpredictably (the "incorrect masks"
    problem). This approach uses your real annotations for the one feature
    that's exact (aspect_ratio) and a real, verifiable background sample
    for the other two — nothing here is a blind guess.

CLASS FOLDERS
    Defects are saved into CLASSES-named subfolders, matching the
    convention build_blob_dataset.py / classify_blob_folders.py already
    expect: OUTPUT_ROOT/<class_name>/<image_stem>_<object_index>.png

REQUIREMENTS
    pip install pillow numpy pandas

USAGE
    python extract_from_yolo_labels.py
    (edit IMAGES_DIR, LABELS_DIR, OUTPUT_ROOT below first)
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------

IMAGES_DIR = "/home/shaivee/PycharmProjects/WIS/DATA/defects_shubham/data/image"
LABELS_DIR = "/home/shaivee/PycharmProjects/WIS/DATA/defects_shubham/data/label"          # YOLO .txt files, same stem as each image
OUTPUT_ROOT = "defect_crops"                # crops saved here, one subfolder per class
FEATURES_OUTPUT_CSV = "yolo_extracted_features.csv"

# Set to a folder path to save a QA visualization per extracted defect,
# showing exactly which pixels inside each box counted as bright / dark /
# neither for A_to_roi_ratio — set to None to skip.
SAVE_MASKS_TO = "masks"

CLASSES = {
    0: "grey-spot",
    1: "hole",
    2: "insect",
    3: "paper-cut",
    4: "water-droplet-scratch",
    5: "wrinkle",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def get_all_images_labels():
    """Pair every image in IMAGES_DIR with its same-stem label file in LABELS_DIR."""
    images_dir = Path(IMAGES_DIR)
    labels_dir = Path(LABELS_DIR)
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((str(img_path), str(label_path)))
        else:
            print(f"  WARNING: no label file for {img_path.name} (expected {label_path}) — skipped")
    return pairs


def parse_yolo_labels(label_path, W, H):
    """Read a YOLO label file, return list of (class_id, x1, y1, x2, y2) in pixel coords."""
    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            class_id, x_c, y_c, bw, bh = map(float, line.split())
            x_center, y_center = x_c * W, y_c * H
            box_width, box_height = bw * W, bh * H
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)
            # clamp to image bounds — a box computed from normalized coords
            # can round slightly outside the image at the edges
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            boxes.append((int(class_id), x1, y1, x2, y2))
    return boxes


def compute_background_stats(gray: np.ndarray, boxes):
    """Mean/std of every pixel in the image that is NOT inside any labeled box."""
    exclude_mask = np.zeros(gray.shape, dtype=bool)
    for _, x1, y1, x2, y2 in boxes:
        exclude_mask[y1:y2, x1:x2] = True
    background_pixels = gray[~exclude_mask]
    if len(background_pixels) == 0:
        # Pathological case: boxes cover the entire image. Fall back to
        # using the whole image as its own reference rather than crashing.
        background_pixels = gray.flatten()
    return float(background_pixels.mean()), float(background_pixels.std() + 1e-6)


def save_bright_dark_mask(defect_crop: np.ndarray, bright_mask: np.ndarray, dark_mask: np.ndarray,
                           output_path) -> None:
    """
    QA visualization: shows exactly which pixels inside the box were
    counted as bright (yellow), dark (blue), or neither (unchanged) when
    computing PERC_BRIGHT / PERC_DARK / A_to_roi_ratio. This is the closest
    equivalent to a segmentation mask in this pipeline — there's no
    separate binary blob/background split like the Otsu approach had,
    since every pixel in the crop IS inside the annotated box by
    definition; this instead shows the brightness classification that
    A_to_roi_ratio is actually built from.
    """
    rgb = np.stack([defect_crop, defect_crop, defect_crop], axis=-1).astype(np.uint8)
    overlay = rgb.copy()
    # bright pixels -> yellow tint
    overlay[bright_mask, 0] = 255
    overlay[bright_mask, 1] = 255
    overlay[bright_mask, 2] = (rgb[bright_mask, 2] * 0.2).astype(np.uint8)
    # dark pixels -> blue tint
    overlay[dark_mask, 0] = (rgb[dark_mask, 0] * 0.2).astype(np.uint8)
    overlay[dark_mask, 1] = (rgb[dark_mask, 1] * 0.2).astype(np.uint8)
    overlay[dark_mask, 2] = 255
    Image.fromarray(overlay).save(output_path)


def extract_and_featurize(file_list, output_root: str = OUTPUT_ROOT):
    output_root = Path(output_root)
    for cls_name in CLASSES.values():
        (output_root / cls_name).mkdir(parents=True, exist_ok=True)
    if SAVE_MASKS_TO:
        for cls_name in CLASSES.values():
            (Path(SAVE_MASKS_TO) / cls_name).mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []

    for image_path, label_path in file_list:
        try:
            img = Image.open(image_path).convert("L")
            gray = np.array(img, dtype=np.float64)
            H, W = gray.shape

            boxes = parse_yolo_labels(label_path, W, H)
            if not boxes:
                skipped.append((image_path, "no valid labels in file"))
                continue

            bg_mean, bg_std = compute_background_stats(gray, boxes)

            for obj_idx, (class_id, x1, y1, x2, y2) in enumerate(boxes):
                box_width_px = x2 - x1
                box_height_px = y2 - y1
                if box_width_px <= 0 or box_height_px <= 0:
                    skipped.append((f"{image_path} obj {obj_idx}", "degenerate box (zero width/height)"))
                    continue

                class_name = CLASSES.get(class_id, f"unknown_class_{class_id}")

                # Save the crop, matching your original function's behavior
                defect_crop = gray[y1:y2, x1:x2].astype(np.uint8)
                stem = Path(image_path).stem
                output_name = output_root / class_name / f"{stem}_{obj_idx}.png"
                Image.fromarray(defect_crop).save(output_name)

                # Features — no Otsu anywhere. aspect_ratio from the box
                # directly; brightness/solidity from the real background
                # reference computed above.
                aspect_ratio = box_height_px / box_width_px

                box_pixels = gray[y1:y2, x1:x2]
                bright_mask = box_pixels > (bg_mean + bg_std)
                dark_mask = box_pixels < (bg_mean - bg_std)
                perc_bright = float(bright_mask.mean())
                perc_dark = float(dark_mask.mean())
                # "solidity": how much of the box is actually different
                # from real background, vs. incidental included material
                a_to_roi_ratio = float((bright_mask | dark_mask).mean())

                if SAVE_MASKS_TO:
                    mask_path = Path(SAVE_MASKS_TO) / class_name / f"{stem}_{obj_idx}_mask.png"
                    save_bright_dark_mask(defect_crop, bright_mask, dark_mask, mask_path)

                rows.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_image": image_path,
                    "object_index": obj_idx,
                    "crop_path": str(output_name),
                    "box_width_px": box_width_px,
                    "box_height_px": box_height_px,
                    "aspect_ratio": aspect_ratio,
                    "PERC_BRIGHT": perc_bright,
                    "PERC_DARK": perc_dark,
                    "A_to_roi_ratio": a_to_roi_ratio,
                    "background_mean": bg_mean,
                    "background_std": bg_std,
                })

        except Exception as e:
            skipped.append((image_path, str(e)))

    df = pd.DataFrame(rows)
    df.to_csv(FEATURES_OUTPUT_CSV, index=False)

    print(f"\nExtracted {len(df)} defect crops from {len(file_list)} images.")
    print(f"Crops saved under: {output_root}/<class_name>/")
    print(f"Features saved to: {FEATURES_OUTPUT_CSV}")
    if skipped:
        print(f"\nSkipped {len(skipped)} items:")
        for item, err in skipped[:10]:
            print(f"  {item}: {err}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    if len(df):
        print("\nClass distribution:")
        print(df["class_name"].value_counts().to_string())

    return df


def run_extraction():
    file_list = get_all_images_labels()
    print(f"Found {len(file_list)} image/label pairs.\n")
    return extract_and_featurize(file_list, OUTPUT_ROOT)


if __name__ == "__main__":
    run_extraction()
