"""
Classify ONE blob with its mask, using the paths set below.
=============================================================================
1. Edit BLOB_PATH and MASK_PATH in the SETTINGS block.
2. Run:  python test_blob_classifier.py

Relative paths are looked up from the folder you run the command in, then
from the folder this file is in. Absolute paths (C:/data/blob.png) also work.

Needs, in the same folder as this file:
    blob_mask_classifier.py, classify_rules.py, class_rules.csv, subclass_rules.csv
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# SETTINGS - change these
# ---------------------------------------------------------------------------

BLOB_PATH = "/home/shaivee/PycharmProjects/WIS_RuleClassification/Rule/defect_crops/grey-spot/7_png.rf.51771077de0985c2395c2b19c0d476d7_14.png"     # the blob image
MASK_PATH = "/home/shaivee/PycharmProjects/WIS_RuleClassification/Rule/masks/grey-spot/7_png.rf.51771077de0985c2395c2b19c0d476d7_14_mask.png"     # its mask: same size, defect = the brighter value

SWAP_AXES = False                 # True if your blobs are rotated 90 degrees
SHOW_RULE_CHECK = True            # print the range check for the predicted class

# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent


def find(path_str):
    """Absolute path as given; relative path from the current folder, else from this file's folder."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), HERE):
        if (base / p).exists():
            return base / p
    return Path.cwd() / p


def main():
    blob, mask = find(BLOB_PATH), find(MASK_PATH)
    for label, path in (("blob", blob), ("mask", mask)):
        if not path.is_file():
            print(f"error: {label} file not found: {path}")
            print("Edit BLOB_PATH / MASK_PATH at the top of this file.")
            return 1

    from blob_mask_classifier import MaskBlobClassifier, _print_result

    clf = MaskBlobClassifier("class_rules.csv","subclass_rules.csv")

    print(f"Blob: {blob}")
    print(f"Mask: {mask}\n")
    try:
        result = clf.classify(blob, mask, swap_axes=SWAP_AXES)
    except ValueError as e:
        print(f"error: {e}")
        return 1

    _print_result(result, clf, SHOW_RULE_CHECK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
