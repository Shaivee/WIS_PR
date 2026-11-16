"""
Classify a defect using ONLY the class_rules.csv range lookup table — no
fitted model, no ML library making the decision.
=============================================================================
THE RULE, IN FULL:

    For a new defect with features x = (aspect_ratio, PERC_BRIGHT,
    PERC_DARK, A_to_roi_ratio), and every class C with a known range
    [lo_i,C, hi_i,C] per feature i (the class's 25th-75th percentile,
    computed by compute_rules.py):

    For each feature i:
        if lo_i,C <= x_i <= hi_i,C:   penalty_i = 0            (fits inside the range)
        if x_i < lo_i,C:               penalty_i = ((lo_i,C - x_i) / width_i,C)^2
        if x_i > hi_i,C:               penalty_i = ((x_i - hi_i,C) / width_i,C)^2

    distance(x, C) = sum of penalty_i over all 4 features

    Classify x as the class C with the SMALLEST total distance. Ties
    (multiple classes where x fits perfectly inside the range for every
    feature) are broken by preferring the class whose range CENTER is
    closest to x — i.e. among equally plausible classes, the one x sits
    most centrally within, not just barely inside the boundary of.

That's the entire decision procedure — a fixed formula evaluated against a
lookup table of pre-computed class ranges (class_rules.csv). There is no
scikit-learn model object anywhere in this file, and nothing is fit to
data at classification time.

REQUIREMENTS
    pip install pandas numpy

USAGE
    from classify_rules import RuleBasedClassifier
    clf = RuleBasedClassifier("class_rules.csv")
    result = clf.predict({"aspect_ratio": 12.5, "PERC_BRIGHT": 0.1,
                           "PERC_DARK": 0.7, "A_to_roi_ratio": 0.25})
"""

import numpy as np
import pandas as pd

FEATURE_COLS = ["aspect_ratio", "PERC_BRIGHT", "PERC_DARK", "A_to_roi_ratio"]


class RuleBasedClassifier:
    def __init__(self, rules_csv: str = "class_rules.csv", confidence_threshold: float = 0.35,
                 subclass_rules_csv: str = "subclass_rules.csv"):
        rules = pd.read_csv(rules_csv, index_col=0)
        self.classes = list(rules.index)
        self.lo = rules[[f"{c}_lo" for c in FEATURE_COLS]].values
        self.hi = rules[[f"{c}_hi" for c in FEATURE_COLS]].values
        self.width = np.where(self.hi - self.lo == 0, 0.01, self.hi - self.lo)
        self.center = (self.hi + self.lo) / 2
        self.feature_names = FEATURE_COLS
        self.threshold = confidence_threshold

        # Optional: sub-class matching within families. If a family (e.g.
        # "Dark Spot (family)") was predicted, this looks up which specific
        # original ISRA class (e.g. "Dark Spot Medium") the sample matches
        # best among just that family's members — same range-check formula,
        # scoped to a smaller set of candidates.
        self.subclass = None
        try:
            sub = pd.read_csv(subclass_rules_csv, index_col=0)
            self.subclass = {
                "names": list(sub.index),
                "lo": sub[[f"{c}_lo" for c in FEATURE_COLS]].values,
                "hi": sub[[f"{c}_hi" for c in FEATURE_COLS]].values,
                "family": sub["Family"].to_dict(),
            }
            self.subclass["width"] = np.where(
                self.subclass["hi"] - self.subclass["lo"] == 0, 0.01,
                self.subclass["hi"] - self.subclass["lo"]
            )
            self.subclass["center"] = (self.subclass["hi"] + self.subclass["lo"]) / 2
        except FileNotFoundError:
            pass  # sub-class matching simply unavailable; predict() still works at family level

        # Measured sub-class accuracy per family (given the correct family
        # was already identified) — see README_rules.md for how this was
        # tested. Reported alongside every sub-class guess so low-reliability
        # families (Bright Spot, Hole) aren't presented with false confidence.
        self.subclass_reliability = {
            "Streak/Scratch (family)": 0.947,
            "Edge Crack (family)": 0.850,
            "Dark Spot (family)": 0.837,
            "Minor Surface Anomaly": 0.663,
            "Hole (family)": 0.593,
            "Bright Spot (family)": 0.458,
        }

    def _scores(self, x: np.ndarray):
        """The rule itself: range-membership penalty, then centering as tie-break."""
        below = np.maximum(self.lo - x, 0)
        above = np.maximum(x - self.hi, 0)
        penalty = (((below + above) / self.width) ** 2).sum(axis=1)        # shape: (n_classes,)
        centering = (((x - self.center) / self.width) ** 2).sum(axis=1)    # shape: (n_classes,)
        return penalty, centering

    def predict(self, features: dict) -> dict:
        x = np.array([features[c] for c in self.feature_names], dtype=float)
        penalty, centering = self._scores(x)

        # Lexicographic ranking: minimize penalty first (does it fit inside
        # the range at all), then centering (how central within the range)
        combined = penalty * 1e6 + centering
        order = np.argsort(combined)
        best_idx = order[0]
        best_class = self.classes[best_idx]
        best_penalty = penalty[best_idx]

        # Confidence proxy: 1 if the sample sits fully inside the winning
        # class's range on every feature (penalty=0), scaled down by how far
        # outside it falls otherwise, and further reduced if a very close
        # runner-up exists. Transparent function of the two scores, not a
        # probability from a fitted model.
        in_range_conf = 1.0 / (1.0 + best_penalty)
        second_combined = combined[order[1]] if len(order) > 1 else combined[order[0]] + 1
        separation = float(1.0 - combined[order[0]] / (second_combined + 1e-6))
        separation = max(0.0, min(1.0, separation))
        confidence = float(in_range_conf * 0.5 + separation * 0.5)

        top_3 = [(self.classes[i], round(float(1.0 / (1.0 + penalty[i] + 0.01 * centering[i])), 4))
                 for i in order[:3]]

        result = {
            "group": best_class,
            "confidence": round(confidence, 4),
            "accept": bool(confidence >= self.threshold),
            "threshold_used": self.threshold,
            "in_range": bool(best_penalty == 0),
            "top_3": top_3,
        }

        # Sub-class matching: which specific original class within the
        # predicted family does this sample resemble most? (No-op for
        # standalone classes that were never part of a family — they have
        # no members in subclass_rules.csv, so nothing extra is added.)
        if self.subclass is not None:
            members_idx = [i for i, name in enumerate(self.subclass["names"])
                            if self.subclass["family"][name] == best_class]
            if members_idx:
                slo = self.subclass["lo"][members_idx]
                shi = self.subclass["hi"][members_idx]
                swidth = self.subclass["width"][members_idx]
                scenter = self.subclass["center"][members_idx]
                sbelow = np.maximum(slo - x, 0)
                sabove = np.maximum(x - shi, 0)
                spenalty = (((sbelow + sabove) / swidth) ** 2).sum(axis=1)
                scentering = (((x - scenter) / swidth) ** 2).sum(axis=1)
                sorder = np.argsort(spenalty * 1e6 + scentering)
                best_sub_i = members_idx[sorder[0]]
                result["subclass"] = self.subclass["names"][best_sub_i]
                result["subclass_reliability"] = self.subclass_reliability.get(best_class)
                result["subclass_options"] = [self.subclass["names"][members_idx[j]] for j in sorder]

        return result

    def explain(self, features: dict, cls: str = None) -> str:
        """Print exactly how the score against a given class (default: the
        winning class) was computed — full transparency into the rule."""
        x = np.array([features[c] for c in self.feature_names], dtype=float)
        if cls is None:
            cls = self.predict(features)["group"]
        idx = self.classes.index(cls)

        lines = [f"Range check against '{cls}':"]
        total_penalty = 0.0
        for i, fname in enumerate(self.feature_names):
            lo, hi = self.lo[idx, i], self.hi[idx, i]
            w = self.width[idx, i]
            v = x[i]
            if v < lo:
                p = ((lo - v) / w) ** 2
                status = f"BELOW range by {lo - v:.3f}"
            elif v > hi:
                p = ((v - hi) / w) ** 2
                status = f"ABOVE range by {v - hi:.3f}"
            else:
                p = 0.0
                status = "inside range"
            total_penalty += p
            lines.append(
                f"  {fname:16s} value={v:.3f}  class_range=[{lo:.3f}, {hi:.3f}]  "
                f"{status}  penalty={p:.3f}"
            )
        lines.append(f"  TOTAL penalty = {total_penalty:.3f}  (0 = fits inside every feature's range)")
        return "\n".join(lines)


if __name__ == "__main__":
    clf = RuleBasedClassifier("class_rules.csv")
    print(f"Loaded rules for {len(clf.classes)} classes.\n")

    example = {"aspect_ratio": 12.5, "PERC_BRIGHT": 0.1, "PERC_DARK": 0.7, "A_to_roi_ratio": 0.25}
    result = clf.predict(example)
    print("Example prediction:", result)
    print()
    print(clf.explain(example))
