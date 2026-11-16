"""
Range scoring: how well a defect fits each class's [lower, upper] ranges.

A parameter inside a class's range costs nothing; outside it costs
((distance outside) / (range width)) ** 2. The class with the smallest total
cost wins; ties go to the class whose range centre the defect is closest to.

testing/classify_rules.py contains an identical copy of these functions, so
that the training and testing folders can each run without the other. If you
change the rule here, change it there too.
"""

import numpy as np

FEATURE_COLS = ["aspect_ratio", "PERC_BRIGHT", "PERC_DARK", "A_to_roi_ratio"]

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
