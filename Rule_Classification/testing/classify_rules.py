"""
Range-based rule classifier (no trained model).

class_rules.csv holds, for each class and each of the four parameters, a lower
and an upper bound (the 25th and 75th percentile of that class in the ISRA
data). A defect is scored against every class:

    - a parameter inside the class's [lower, upper] range costs nothing;
    - a parameter outside it costs ((distance outside) / (range width)) ** 2;
    - the class with the smallest total cost wins; ties go to the class whose
      range centre the defect is closest to.

subclass_rules.csv (optional) does the same inside a family, to name the
specific ISRA class (for example "Dark Spot Small" within "Dark Spot (family)").
"""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = ["aspect_ratio", "PERC_BRIGHT", "PERC_DARK", "A_to_roi_ratio"]

# range_scores and best_class_index below are duplicated in training/rule_scoring.py so
# that each pipeline runs on its own. If you change the rule, change it in both places.

# Penalty decides the ranking; centering only separates classes with equal penalty.
PENALTY_WEIGHT = 1e6
MIN_RANGE_WIDTH = 0.01  # stands in for a zero-width range, avoiding division by zero


def range_scores(x, lo, hi):
    """
    Score defects against class ranges.

    x      : (n_features,) for one defect, or (n_defects, n_features)
    lo, hi : (n_classes, n_features) lower and upper bounds
    Returns (penalty, centering), each shaped (n_classes,) for one defect or
    (n_defects, n_classes) for several. Lower is better for both.
    """
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    x = np.atleast_2d(x)[:, None, :]
    width = np.where(hi - lo == 0, MIN_RANGE_WIDTH, hi - lo)
    centre = (hi + lo) / 2
    outside = np.maximum(lo - x, 0) + np.maximum(x - hi, 0)
    penalty = ((outside / width) ** 2).sum(axis=2)
    centering = (((x - centre) / width) ** 2).sum(axis=2)
    return (penalty[0], centering[0]) if single else (penalty, centering)


def best_class_index(penalty, centering):
    """Index of the winning class for each defect (penalty first, then centering)."""
    return np.argmin(penalty * PENALTY_WEIGHT + centering, axis=-1)


class RuleBasedClassifier:
    def __init__(self, rules_csv, subclass_rules_csv=None, confidence_threshold=0.35):
        rules = pd.read_csv(rules_csv, index_col=0)
        self.classes = list(rules.index)
        self.lo = rules[[f"{c}_lo" for c in FEATURE_COLS]].to_numpy()
        self.hi = rules[[f"{c}_hi" for c in FEATURE_COLS]].to_numpy()
        self.threshold = confidence_threshold

        self.subclass = None
        if subclass_rules_csv is not None and Path(subclass_rules_csv).exists():
            sub = pd.read_csv(subclass_rules_csv, index_col=0)
            self.subclass = {
                "names": list(sub.index),
                "family": sub["Family"].to_numpy(),
                "reliability": sub["Reliability"].to_numpy(),
                "lo": sub[[f"{c}_lo" for c in FEATURE_COLS]].to_numpy(),
                "hi": sub[[f"{c}_hi" for c in FEATURE_COLS]].to_numpy(),
            }

    def predict(self, features):
        """Classify one defect given a dict with the four parameters."""
        x = np.array([features[c] for c in FEATURE_COLS], dtype=float)
        penalty, centering = range_scores(x, self.lo, self.hi)
        combined = penalty * PENALTY_WEIGHT + centering
        order = np.argsort(combined)
        best = order[0]

        # Confidence is a transparent score, not a probability: half is how well
        # the defect fits the winning class's ranges, half is how far ahead of
        # the runner-up it is.
        fit = 1.0 / (1.0 + penalty[best])
        separation = 1.0 - combined[best] / (combined[order[1]] + 1e-6)
        confidence = 0.5 * fit + 0.5 * min(1.0, max(0.0, separation))

        result = {
            "group": self.classes[best],
            "confidence": round(float(confidence), 4),
            "accept": bool(confidence >= self.threshold),
            "in_range": bool(penalty[best] == 0),
            "total_penalty": float(penalty[best]),
            "top_3": [(self.classes[i], round(float(1.0 / (1.0 + penalty[i] + 0.01 * centering[i])), 4))
                      for i in order[:3]],
        }

        if self.subclass is not None:
            members = np.flatnonzero(self.subclass["family"] == self.classes[best])
            if members.size:
                sp, sc = range_scores(x, self.subclass["lo"][members], self.subclass["hi"][members])
                pick = members[int(best_class_index(sp, sc))]
                result["subclass"] = self.subclass["names"][pick]
                result["subclass_reliability"] = float(self.subclass["reliability"][pick])
        return result

    def explain(self, features, cls=None):
        """Text showing how each parameter compares with a class's range (default: the winner)."""
        x = np.array([features[c] for c in FEATURE_COLS], dtype=float)
        cls = cls or self.predict(features)["group"]
        idx = self.classes.index(cls)
        width = np.where(self.hi[idx] - self.lo[idx] == 0, MIN_RANGE_WIDTH, self.hi[idx] - self.lo[idx])

        lines, total = [f"Range check against '{cls}':"], 0.0
        for i, name in enumerate(FEATURE_COLS):
            lo, hi, v = self.lo[idx, i], self.hi[idx, i], x[i]
            if v < lo:
                penalty, status = ((lo - v) / width[i]) ** 2, f"BELOW range by {lo - v:.3f}"
            elif v > hi:
                penalty, status = ((v - hi) / width[i]) ** 2, f"ABOVE range by {v - hi:.3f}"
            else:
                penalty, status = 0.0, "inside range"
            total += penalty
            lines.append(f"  {name:15s} value={v:.3f}  range=[{lo:.3f}, {hi:.3f}]  {status}  penalty={penalty:.3f}")
        lines.append(f"  total penalty = {total:.3f}  (0 means it fits every range)")
        return "\n".join(lines)
