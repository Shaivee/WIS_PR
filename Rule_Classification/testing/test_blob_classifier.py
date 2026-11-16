"""
Test the classifier on one blob and its mask, then check the production path.

Edit BLOB_PATH and MASK_PATH below and run:  python test_blob_classifier.py

Step 1 (test mode)      calculates the four parameters from the blob and mask and classifies.
Step 2                  prints those parameters: this is what production receives.
Step 3 (production)     classifies from the parameters alone and confirms the answer matches.

Relative paths are looked up from the current folder, then from this file's folder.
"""

import sys
from pathlib import Path

from blob_mask_classifier import MaskBlobClassifier, print_result
from classify_rules import FEATURE_COLS

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BLOB_PATH = "/home/shaivee/PycharmProjects/WIS_RuleClassification/Rule/defect_crops/grey-spot/7_png.rf.51771077de0985c2395c2b19c0d476d7_14.png"     # the blob image
MASK_PATH = "/home/shaivee/PycharmProjects/WIS_RuleClassification/Rule/masks/grey-spot/7_png.rf.51771077de0985c2395c2b19c0d476d7_14_mask.png"     # its mask: same size, defect = the brighter value

SHOW_RULE_CHECK = True        # print how each parameter compares with the predicted class's range
# ---------------------------------------------------------------------------


def find(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        if (base / path).exists():
            return base / path
    return Path.cwd() / path


def main():
    blob, mask = find(BLOB_PATH), find(MASK_PATH)
    for label, path in (("blob", blob), ("mask", mask)):
        if not path.is_file():
            print(f"error: {label} file not found: {path}\nEdit BLOB_PATH / MASK_PATH at the top of this file.")
            return 1

    clf = MaskBlobClassifier()

    print("STEP 1 - TEST MODE: parameters are calculated from the blob and its mask")
    print(f"  Blob: {blob}\n  Mask: {mask}\n")
    try:
        test = clf.classify(blob, mask)
    except ValueError as error:
        print(f"error: {error}")
        return 1
    print_result(test, clf, SHOW_RULE_CHECK)

    parameters = {name: test[name] for name in FEATURE_COLS}
    print("\nSTEP 2 - THE FOUR PARAMETERS (what production receives)")
    for name, value in parameters.items():
        print(f"  {name:15s} {value!r}")

    print("\nSTEP 3 - PRODUCTION MODE: classify from those parameters alone")
    production = clf.classify_parameters(parameters, blob=blob)
    print_result(production, clf)

    same = all(production[key] == test[key]
               for key in ("predicted_class", "confidence", "specific_match", "total_penalty"))
    print(f"\nProduction result matches test result: {'YES' if same else 'NO'}")
    return 0 if same else 2


if __name__ == "__main__":
    sys.exit(main())
