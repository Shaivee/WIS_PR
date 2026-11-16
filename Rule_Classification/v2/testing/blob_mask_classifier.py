"""
Rule-based defect classifier. Give it a blob image and its mask; it returns JSON with the
class the defect is classified as and the runner-up class.

    from blob_mask_classifier import MaskBlobClassifier
    MaskBlobClassifier().classify("blob.png", "mask.png")
    # '{"class": "Dark Spot (family)", "runner_up": "Lump"}'

blob, mask : image file paths, or 2-D arrays of the same size. The defect in the mask is the
             brighter value (255/0 or 1/0). If the mask has several regions, the one with the
             most pixels is the defect; the other regions are left out of the background.

If the four parameters are already calculated, classify_parameters() skips the image step:
    MaskBlobClassifier().classify_parameters(
        {"aspect_ratio": 1.0, "PERC_BRIGHT": 0.04, "PERC_DARK": 0.79, "A_to_roi_ratio": 0.83})

The parameters (ROI = the mask's bounding box; background = every pixel outside the mask):
    aspect_ratio = box_h / box_w          PERC_BRIGHT = share of ROI brighter than background + 1 std
    A_to_roi_ratio = share of ROI that is bright or dark          PERC_DARK = share darker than background - 1 std
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

RULES = Path(__file__).resolve().parent / "rules" / "class_rules.csv"
FEATURES = ["aspect_ratio", "PERC_BRIGHT", "PERC_DARK", "A_to_roi_ratio"]


def load(image):
    """File path or 2-D array -> 2-D float array."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("L")
    return np.asarray(image, dtype=float)


class MaskBlobClassifier:
    def __init__(self, rules_csv=RULES):
        rules = pd.read_csv(rules_csv, index_col=0)
        self.classes = list(rules.index)
        self.lo = rules[[f"{f}_lo" for f in FEATURES]].to_numpy()
        self.hi = rules[[f"{f}_hi" for f in FEATURES]].to_numpy()
        self.width = np.where(self.hi == self.lo, 0.01, self.hi - self.lo)

    def classify(self, blob, mask):
        """Blob image and mask in, JSON out."""
        gray, mask = load(blob), load(mask)
        if gray.shape != mask.shape:
            raise ValueError(f"blob {gray.shape} and mask {mask.shape} must be the same size")
        defect = mask > mask.max() / 2
        labels, regions = ndimage.label(defect)
        if regions == 0:
            raise ValueError("mask contains no defect")
        largest = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1     # the region with the most pixels

        ys, xs = np.where(largest)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        background = gray[~defect]
        if background.size < 50 or ((y1, x1, y2, x2) == (0, 0, *gray.shape) and largest.mean() > 0.5):
            raise ValueError("no usable background: the mask covers the image or is inverted "
                             "(the defect must be the brighter value)")

        roi = gray[y1:y2, x1:x2]
        mean, std = background.mean(), background.std() + 1e-6
        bright, dark = roi > mean + std, roi < mean - std
        return self.classify_parameters({"aspect_ratio": (y2 - y1) / (x2 - x1),
                                         "PERC_BRIGHT": bright.mean(), "PERC_DARK": dark.mean(),
                                         "A_to_roi_ratio": (bright | dark).mean()})

    def classify_parameters(self, parameters):
        """The four parameters (dict) in, JSON out."""
        p = {f: float(parameters[f]) for f in FEATURES}
        if (not np.isfinite(list(p.values())).all() or p["aspect_ratio"] <= 0
                or not 0 <= p["PERC_BRIGHT"] <= 1 or not 0 <= p["PERC_DARK"] <= 1):
            raise ValueError(f"invalid parameters: {p}")

        x = np.array([p[f] for f in FEATURES])
        outside = np.maximum(self.lo - x, 0) + np.maximum(x - self.hi, 0)
        penalty = ((outside / self.width) ** 2).sum(axis=1)           # 0 when x is inside a class's ranges
        centering = (((x - (self.lo + self.hi) / 2) / self.width) ** 2).sum(axis=1)   # breaks ties
        best, runner_up = np.argsort(penalty * 1e6 + centering)[:2]
        return json.dumps({"class_1": self.classes[best], "class_2": self.classes[runner_up]})


if  __name__=="__main__":
    classifier = MaskBlobClassifier()
    text = '{"aspect_ratio": 1.0, "PERC_BRIGHT": 0.04, "PERC_DARK": 0.79, "A_to_roi_ratio": 0.83}'

    print(classifier.classify("/home/shaivee/PycharmProjects/WIS/DATA/defects_shubham/defect_crops/grey-spot/1_png.rf.b9351c5e9cff00f1272392034ac548ac_9.png",
                              "/home/shaivee/PycharmProjects/WIS/DATA/defects_shubham/masks/grey-spot/1_png.rf.b9351c5e9cff00f1272392034ac548ac_9_mask.png"))

    # print(classifier.classify_parameters(json.loads(text)))
