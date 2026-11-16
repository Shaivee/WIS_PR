"""
Visibility check for labelled defects (step 7).

For each defect: contrast-to-noise ratio (CNR) in the FINAL image, measured at the defect's own scale:
    signal = RMS deviation of the defect from the local background, minus the background's own
             variation (so an invisible defect scores ~0), after smoothing at the defect scale
    noise  = robust spread of the surrounding ring of paper (same smoothing)
Smoothing scale follows the defect's thickness (thin -> fine, large -> coarser); line defects are also
averaged along their length (as a detector can). Other defects are excluded from the background ring.
"""
import numpy as np, cv2

LINE_LIKE = {"bladeline", "Coating mark", "Crease", "Impression line", "blade scratch"}


def _cnr_at(f, m, ring, sig, sy):
    fs = cv2.GaussianBlur(f, (0, 0), sigmaX=sig, sigmaY=sy)
    bgv = fs[ring]; bg = np.median(bgv)
    noise = 1.4826 * np.median(np.abs(bgv - bg)) + 1e-3
    inside = fs[m > 0]
    # excess contrast beyond the paper's own variation (0 for an invisible defect)
    excess = np.sqrt(max(0.0, float(np.mean((inside - bg) ** 2)) - noise ** 2))
    return excess / noise


def defect_cnr(img, mask, all_defects=None, cls=None):
    """Best contrast-to-noise ratio over several scales (fine rims / relief edges up to broad stains)."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 3:
        return 0.0
    H, W = img.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    size = max(y1 - y0, x1 - x0)
    pad = int(np.clip(0.5 * size, 25, 200))
    Y0, Y1, X0, X1 = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
    f = img[Y0:Y1, X0:X1].astype(np.float32)
    m = mask[Y0:Y1, X0:X1].astype(np.uint8)
    thick = 2 * float(cv2.distanceTransform(m, cv2.DIST_L2, 3).max())
    k1 = max(5, int(0.1 * thick + 5) | 1); k2 = max(k1 + 6, int(0.4 * thick + 25) | 1)
    near = cv2.dilate(m, np.ones((k1, k1), np.uint8)); far = cv2.dilate(m, np.ones((k2, k2), np.uint8))
    ring = (far > 0) & (near == 0)
    if all_defects is not None:
        ring &= ~cv2.dilate(all_defects[Y0:Y1, X0:X1].astype(np.uint8), np.ones((k1, k1), np.uint8)).astype(bool)
    if ring.sum() < 30:
        return float("nan")
    best = 0.0
    for sig in (0.7, 1.5, 3.0, 6.0):
        if sig > max(0.7, 0.4 * thick):
            break
        sy = max(sig, 8.0) if cls in LINE_LIKE else sig                           # lines: average along their length
        best = max(best, _cnr_at(f, m, ring, sig, sy))
    return float(best)
