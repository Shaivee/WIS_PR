#!/usr/bin/env python3
"""
WIS synthetic defect dataset generator
Paper / board web  |  line-scan camera  |  3 lights (transmitted, reflected, dark-field)

Output images are grayscale, one per light (all pixel-aligned):
    T  = transmitted (backlit)   R = reflected (front-lit)   DF = dark-field
--mode separate (default) writes three datasets: <out>/T, <out>/R, <out>/DF.
--mode T | R | DF writes only that light.  --mode rgb writes the old 3-channel composite.
Each light only labels defects that are actually visible under that light
(e.g. wrinkles are not labelled in T, thin spots are not labelled in DF).
About 40% of frames (--irregular) also get 1-3 unlabelled background irregularities:
cd_profile, cloudy_patch, felt_mark, speed_variation, md_banding, uneven_light,
flutter_blur, exposure, pixel_columns (listed per frame in manifest.csv).
Machine direction (MD) = image vertical, cross direction (CD) = image horizontal.

Grades:   tissue, printing/writing
Defects:  hole, pinhole, edge_crack, dirt, shive, slime_spot,
          thin_spot, streak, wet_spot, wrinkle, crease, fold_over

Labels (all derived from one instance map, so they always agree):
    labels/       YOLO detection      cls cx cy w h                    (normalised)
    labels_obb/   YOLO oriented boxes cls x1 y1 x2 y2 x3 y3 x4 y4      (normalised)
    masks/        semantic PNG        pixel = class_id + 1, 0 = no defect
    instances/    16-bit PNG          pixel = instance id, 0 = no defect
    annotations/  COCO JSON           bbox + polygon segmentation (+ obb in attributes)

Usage:
    pip install numpy opencv-python
    python wis_generator.py --n 10000 --out wis_dataset --workers 8           # T, R and DF datasets
    python wis_generator.py --n 10000 --mode T                                # transmitted light only
    python wis_generator.py --n 10000 --slice 1024                            # also 1024x1024 crops
    python wis_generator.py --n 20 --out test --preview 8    # quick look
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

CLASSES = ["hole", "pinhole", "edge_crack", "dirt", "shive", "slime_spot",
           "thin_spot", "streak", "wet_spot", "wrinkle", "crease", "fold_over"]
EDGE_CLASSES = {"edge_crack", "fold_over"}
# relative sampling weights for defects away from the web edge (pinhole boosted)
WEIGHTS = {"hole": 0.09, "pinhole": 0.12, "dirt": 0.11, "shive": 0.09, "slime_spot": 0.08,
           "thin_spot": 0.08, "streak": 0.08, "wet_spot": 0.08, "wrinkle": 0.09, "crease": 0.09}
# local drawing window per defect (mm); None = full frame height
WINDOW_MM = {"hole": (40, 40), "pinhole": (10, 10), "dirt": (30, 30), "shive": (30, 30),
             "slime_spot": (40, 40), "thin_spot": (40, 40), "wet_spot": (50, 50),
             "streak": (None, 40), "wrinkle": (None, 120), "crease": (None, 150)}
GRADE_NAMES = ["tissue", "printing"]
OPEN_T, OPEN_R, OPEN_D = 250.0, 32.0, 3.0     # camera response where there is no paper

MM_PER_PX = 0.1


def _init_worker(mm):
    global MM_PER_PX
    MM_PER_PX = mm


def px(mm):
    return mm / MM_PER_PX


# ============================================================================ #
# Noise helpers
# ============================================================================ #
def lf(h, w, rng, scale):
    """Smooth unit-variance noise with features about `scale` px across."""
    scale = max(2, int(scale))
    s = rng.standard_normal((max(2, h // scale + 2), max(2, w // scale + 2))).astype(np.float32)
    b = cv2.resize(s, (w, h), interpolation=cv2.INTER_CUBIC)
    return b / (b.std() + 1e-6)


def fine(h, w, rng, sigma=0.9):
    g = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    return g / (g.std() + 1e-6)


def grid(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return yy.astype(np.float32), xx.astype(np.float32)


def U(rng, a, b):
    return float(rng.uniform(a, b))


# ============================================================================ #
# Paper background
# ============================================================================ #
GRADES = {
    "tissue":   dict(floc=(3, 8), amp=(0.10, 0.18), T=(155, 180), R=(195, 215), grain=1.0, crepe=(0.3, 0.7), fib=1.0),
    "printing": dict(floc=(1, 4), amp=(0.05, 0.10), T=(105, 140), R=(205, 225), grain=0.5, crepe=None, fib=0.4),
}


def fibre_layer(h, w, rng, n):
    c = np.zeros((h, w), np.uint8)
    x, y = rng.uniform(0, w, n), rng.uniform(0, h, n)
    L = px(rng.uniform(1.0, 3.0, n))
    a = rng.normal(np.pi / 2, 0.5, n)                      # slight MD orientation
    x2, y2 = x + L * np.cos(a), y + L * np.sin(a)
    val = rng.uniform(60, 160, n)
    for i in range(n):
        cv2.line(c, (int(x[i] * 16), int(y[i] * 16)), (int(x2[i] * 16), int(y2[i] * 16)),
                 int(val[i]), 1, cv2.LINE_AA, shift=4)
    return c.astype(np.float32) / 255.0


def crepe(h, w, rng, period_mm):
    p = px(U(rng, *period_mm))
    dphi = 2 * np.pi / p * (1 + 0.25 * lf(h, 1, rng, 20)[:, 0])
    phi = np.cumsum(dphi)[:, None] + 1.5 * lf(1, w, rng, 60) + 0.6 * lf(h, w, rng, 25)
    return (np.sin(phi) * (0.6 + 0.4 * np.abs(lf(h, w, rng, 15)))).astype(np.float32)


def paper_background(grade, h, w, rng):
    g = GRADES[grade]
    floc = px(U(rng, *g["floc"]))
    F = 0.5 * lf(h, w, rng, floc / 3) + lf(h, w, rng, floc) + 0.5 * lf(h, w, rng, floc * 3)
    F /= F.std()
    grain = fine(h, w, rng)
    fib = fibre_layer(h, w, rng, int(h * w / 400 * g["fib"]))
    height = 0.5 * F + g["grain"] * grain + 1.5 * fib
    thick = U(rng, *g["amp"]) * F + 0.02 * grain + 0.05 * fib
    if g["crepe"]:
        cr = crepe(h, w, rng, g["crepe"])
        height += 1.8 * cr
        thick += 0.03 * cr
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    t = np.exp(-thick)
    T = U(rng, *g["T"]) * t / t.mean()                     # auto-exposure
    R = U(rng, *g["R"]) + 2 * F + 2.5 * gy
    D = 14 + 4.5 * np.sqrt(gx ** 2 + gy ** 2)
    return [T.astype(np.float32), R.astype(np.float32), D.astype(np.float32)]


def sensor_model(h, w, rng):
    prnu = 1 + 0.006 * rng.standard_normal(w).astype(np.float32)[None, :]
    flicker = 1 + 0.004 * lf(h, 1, rng, 40)
    x = np.linspace(-1, 1, w, dtype=np.float32)
    return prnu * flicker * (1 - U(rng, 0.03, 0.10) * x ** 2)[None, :]


# ============================================================================ #
# Shared defect helpers (all work on float channel lists, no noise)
# ============================================================================ #
def ragged_blob(h, w, rng, cx, cy, rx, ry, ang, rough):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(rx)), max(1, int(ry))), float(ang), 0, 360, 255, -1)
    m = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), max(1.0, min(rx, ry) / 3))
    n = 0.7 * lf(h, w, rng, max(2, min(rx, ry) / 3)) + 0.3 * fine(h, w, rng, 1.6)
    out = ((m + rough * n) > 0.5) & (m > 0.03)
    k, lab, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8))
    if k > 2:
        out = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return out


def rim(A):
    g = np.sqrt(cv2.Sobel(A, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(A, cv2.CV_32F, 0, 1) ** 2)
    return g / (g.max() + 1e-6)


def apply_gain(ch, A, t_gain, r_delta, d_delta, d_gain=1.0):
    T, R, D = ch
    return [T * (1 + (t_gain - 1) * A), R + r_delta * A, D * (1 + (d_gain - 1) * A) + d_delta * A]


def comp_open(ch, missing, rng, thin_ring=True):
    """Replace missing paper with open background; torn-edge effects."""
    T, R, D = ch
    P = 1 - cv2.GaussianBlur(missing.astype(np.float32), (0, 0), 0.7)
    rv = cv2.GaussianBlur(np.abs(cv2.Laplacian(P, cv2.CV_32F)), (0, 0), 0.8)
    rv /= rv.max() + 1e-6
    if thin_ring:
        ring = cv2.GaussianBlur(missing.astype(np.float32), (0, 0), px(0.6)) * P
        T = T * (1 + 0.25 * ring)
    T = P * T + (1 - P) * OPEN_T
    R = P * R + (1 - P) * OPEN_R - 15 * rv
    D = P * D + (1 - P) * OPEN_D + U(rng, 40, 80) * rv
    return [T, R, D]


def random_center(h, w, rng, margin):
    mx, my = min(margin, w / 2 - 1), min(margin, h / 2 - 1)
    return U(rng, mx, w - mx), U(rng, my, h - my)


# ============================================================================ #
# Defects — each takes a window (list of 3 float arrays) and returns (channels, mask)
# ============================================================================ #
def d_hole(ch, rng):
    h, w = ch[0].shape
    r = px(U(rng, 1.5, 10))
    cx, cy = random_center(h, w, rng, r * 1.3)
    m = ragged_blob(h, w, rng, cx, cy, r * U(rng, 0.6, 1.3), r * U(rng, 0.6, 1.3), U(rng, 0, 180), 0.35)
    label = m.copy()
    ys, xs = np.nonzero(m)
    if len(xs):
        c = np.zeros((h, w), np.uint8)
        for _ in range(int(rng.integers(0, 6))):          # fibres bridging the hole
            i = rng.integers(len(xs)); a = U(rng, 0, np.pi); L = px(U(rng, 1, 3))
            cv2.line(c, (int(xs[i] - L * np.cos(a)), int(ys[i] - L * np.sin(a))),
                     (int(xs[i] + L * np.cos(a)), int(ys[i] + L * np.sin(a))), 255, 1, cv2.LINE_AA)
        m &= ~(c > 90)
    return comp_open(ch, m, rng), label


def d_pinhole(ch, rng):
    h, w = ch[0].shape
    r = px(U(rng, 1.0, 2.0)) / 2
    cx, cy = random_center(h, w, rng, r * 2)
    m = ragged_blob(h, w, rng, cx, cy, r, r * U(rng, 0.8, 1.2), 0, 0.15)
    return comp_open(ch, m, rng), m


def d_dirt(ch, rng):
    h, w = ch[0].shape
    A = np.zeros((h, w), np.float32)
    cx, cy = random_center(h, w, rng, px(6))
    cluster = rng.random() < 0.35
    for _ in range(int(rng.integers(3, 9)) if cluster else 1):
        r = px(U(rng, 0.5, 1.2) if cluster else U(rng, 0.5, 2.5))
        ox, oy = (rng.normal(0, px(2)), rng.normal(0, px(2))) if cluster else (0, 0)
        b = ragged_blob(h, w, rng, cx + ox, cy + oy, r * U(rng, 0.6, 1.3), r * U(rng, 0.6, 1.3),
                        U(rng, 0, 180), 0.3)
        A = np.maximum(A, b.astype(np.float32))
    A = cv2.GaussianBlur(A, (0, 0), 0.7)
    out = apply_gain(ch, A, 1 - U(rng, 0.55, 0.85), -U(rng, 70, 140), 0)
    raised = U(rng, 0, 1)
    out[2] = out[2] + raised * 50 * rim(A)                  # raised particles scatter
    return out, A > 0.3, {"raised": raised > 0.5}


def d_shive(ch, rng):
    h, w = ch[0].shape
    L, W = px(U(rng, 2, 10)), px(U(rng, 0.3, 1.0))
    ang = np.deg2rad(rng.normal(90, 20))
    cx, cy = random_center(h, w, rng, L / 2 + 4)
    yy, xx = grid(h, w)
    u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    s = np.clip(2 * np.abs(u) / L, 0, 1)
    half = W / 2 * np.sqrt(np.clip(1 - s ** 3, 0, 1))
    bend = px(U(rng, -0.4, 0.4)) * (2 * u / L) ** 2
    A = np.clip((half - np.abs(v - bend)) / 1.2 + 0.5, 0, 1) * np.clip(half / 0.8, 0, 1)
    stri = np.interp(v.ravel(), np.linspace(-W, W, 24), rng.uniform(0.6, 1.0, 24)).reshape(h, w)
    A = cv2.GaussianBlur((A * stri).astype(np.float32), (0, 0), 0.6)
    return apply_gain(ch, A, 1 - U(rng, 0.35, 0.6), -U(rng, 25, 60), U(rng, 35, 70)), A > 0.3


def d_slime(ch, rng):
    h, w = ch[0].shape
    r = px(U(rng, 1.5, 10))
    cx, cy = random_center(h, w, rng, r * 1.4 + 4)
    b = ragged_blob(h, w, rng, cx, cy, r * U(rng, 0.7, 1.3), r * U(rng, 0.7, 1.3), U(rng, 0, 180), 0.25)
    A = cv2.GaussianBlur(b.astype(np.float32), (0, 0), max(1.2, r / 12))
    A = np.clip(A * (0.85 + 0.15 * lf(h, w, rng, max(3, r / 3))), 0, 1)
    ring = np.clip(4 * A * (1 - A), 0, 1)
    T, R, D = apply_gain(ch, A, 1 + U(rng, 0.35, 0.65), -U(rng, 10, 25), 0, d_gain=U(rng, 0.3, 0.55))
    return [T * (1 - U(rng, 0.15, 0.3) * ring), R, D + U(rng, 8, 20) * ring], A > 0.3


def d_thin_spot(ch, rng):
    h, w = ch[0].shape
    r = px(U(rng, 1.5, 10))
    cx, cy = random_center(h, w, rng, r * 1.6)
    b = ragged_blob(h, w, rng, cx, cy, r * U(rng, 0.6, 1.1), r * U(rng, 0.9, 1.6), rng.normal(0, 15), 0.3)
    A = cv2.GaussianBlur(b.astype(np.float32), (0, 0), max(2.0, r / 5))
    A /= A.max() + 1e-6
    return apply_gain(ch, A, 1 + U(rng, 0.4, 0.75), -U(rng, 8, 20), 0, d_gain=U(rng, 0.75, 0.95)), A > 0.4


def d_streak(ch, rng):
    h, w = ch[0].shape
    width = px(U(rng, 1.5, 12))
    x0 = U(rng, width, w - width)
    yy = np.arange(h, dtype=np.float32)
    xc = x0 + px(U(rng, 0.1, 0.5)) * lf(h, 1, rng, 80)[:, 0]
    xx = np.arange(w, dtype=np.float32)[None, :]
    prof = np.exp(-(np.abs(xx - xc[:, None]) / (width / 2)) ** 4)
    if rng.random() < 0.6:
        along = np.ones(h, np.float32)
    else:
        y0 = U(rng, 0, 0.5 * h); y1 = U(rng, y0 + 0.3 * h, h)
        along = np.clip(np.minimum(yy - y0, y1 - yy) / px(3), 0, 1)
    along = along * (0.8 + 0.2 * lf(h, 1, rng, 40)[:, 0])
    A = np.clip(prof * along[:, None], 0, 1).astype(np.float32)
    if rng.random() < 0.7:                                  # basis-weight streak
        sign = 1 if rng.random() < 0.5 else -1
        out = apply_gain(ch, A, 1 + sign * U(rng, 0.28, 0.45), -sign * U(rng, 5, 12), 0)
        variant = "basis_weight"
    else:                                                   # calender / surface streak
        out = apply_gain(ch, A, 1 + U(rng, -0.03, 0.03), U(rng, 20, 35), 0, d_gain=U(rng, 0.25, 0.5))
        variant = "surface"
    return out, A > 0.4, {"variant": variant}


def d_wet_spot(ch, rng):
    h, w = ch[0].shape
    r = px(U(rng, 1.5, 6))
    cx, cy = random_center(h, w, rng, r * 2.5)
    c = np.zeros((h, w), np.uint8)
    cv2.ellipse(c, (int(cx), int(cy)), (int(r), int(r * U(rng, 1.0, 1.3))), 0, 0, 360, 255, -1)
    if rng.random() < 0.7:
        d = 1 if rng.random() < 0.5 else -1
        tail = r * U(rng, 1, 3)
        pts = np.array([[cx - r * 0.9, cy], [cx + r * 0.9, cy],
                        [cx + rng.normal(0, r * 0.2), cy + d * (r + tail)]], np.int32)
        cv2.fillConvexPoly(c, pts, 255)
    for _ in range(int(rng.integers(0, 5))):
        a, dist = U(rng, 0, 2 * np.pi), r * U(rng, 1.5, 3.5)
        cv2.circle(c, (int(cx + dist * np.cos(a)), int(cy + dist * np.sin(a))),
                   max(2, int(r * U(rng, 0.1, 0.3))), 255, -1)
    cf = c.astype(np.float32) / 255
    edge = 0.2 * fine(h, w, rng, 1.5) * (cv2.GaussianBlur(cf, (0, 0), 2) > 0.02)
    A = cv2.GaussianBlur(((cv2.GaussianBlur(cf, (0, 0), 1.2) + edge) > 0.5).astype(np.float32), (0, 0), 0.9)
    T, R, D = apply_gain(ch, A, 1 + U(rng, 0.3, 0.7), -U(rng, 15, 35), 0, d_gain=U(rng, 0.35, 0.6))
    if rng.random() < 0.5:
        T = T * (1 - 0.25 * rim(A))                         # tide line
    return [T, R, D], A > 0.4


def surface_response(ch, hmap, t_gain, r_gain, d_gain):
    T, R, D = ch
    gy = cv2.Sobel(hmap, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(hmap, cv2.CV_32F, 1, 0, ksize=3)
    return [T * (1 - t_gain * hmap / (hmap.max() + 1e-6)), R + r_gain * gy + 0.3 * r_gain * gx,
            D + d_gain * np.sqrt(gx ** 2 + gy ** 2)]


def rotated_coords(h, w, ang, cx, cy):
    yy, xx = grid(h, w)
    u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    return u, v


def d_wrinkle(ch, rng):
    h, w = ch[0].shape
    ang = np.deg2rad(rng.normal(90, 30) if rng.random() < 0.7 else U(rng, 0, 180))
    cx, cy = U(rng, 0.25 * w, 0.75 * w), U(rng, 0.25 * h, 0.75 * h)
    u, v = rotated_coords(h, w, ang, cx, cy)
    L = px(U(rng, 15, 60))
    hmap = np.zeros((h, w), np.float32)
    n = 1 if rng.random() < 0.6 else int(rng.integers(2, 4))
    gap = px(U(rng, 3, 8))
    for k in range(n):
        s = px(U(rng, 0.6, 2.5))
        vv = v - (k - (n - 1) / 2) * gap - px(U(rng, -1, 1)) * (u / L) ** 2
        env = np.exp(-(np.abs(u + U(rng, -0.1, 0.1) * L) / (L / 2)) ** 4)
        hmap += U(rng, 0.7, 1.0) * np.exp(-vv ** 2 / (2 * s * s)) * env
    out = surface_response(ch, hmap * U(rng, 1.5, 2.5), 0.05, U(rng, 30, 50), U(rng, 45, 75))
    return out, hmap > 0.3


def d_crease(ch, rng):
    h, w = ch[0].shape
    ang = np.deg2rad(U(rng, 0, 180))
    cx, cy = U(rng, 0.2 * w, 0.8 * w), U(rng, 0.2 * h, 0.8 * h)
    u, v = rotated_coords(h, w, ang, cx, cy)
    v = v + px(0.15) * lf(h, w, rng, 60)
    L = px(U(rng, 10, 80))
    env = np.clip((L / 2 - np.abs(u)) / px(1.5), 0, 1)
    width = px(U(rng, 0.3, 1.0))
    line = np.exp(-(v / (width / 2)) ** 2) * env
    broad = np.exp(-(v / (width * 3)) ** 2) * env
    T, R, D = ch
    T = T * (1 - U(rng, 0.2, 0.35) * line)
    R = R + U(rng, 8, 18) * np.sign(v) * broad * np.sign(np.sin(ang) + 1e-3) - U(rng, 10, 25) * line
    D = D + U(rng, 40, 90) * line + U(rng, 5, 12) * broad
    return [T, R, D], line > 0.3


# ---- edge defects: web lies to the right of edge e(y) (frame is flipped later if needed)
def d_edge_crack(ch, e, rng):
    h, w = ch[0].shape
    xx = np.arange(w, dtype=np.float32)[None, :]
    outside = xx < e[:, None]
    y0 = U(rng, 0.2 * h, 0.8 * h)
    L, w0 = px(U(rng, 3, 25)), px(U(rng, 0.8, 2.5))
    x, y, a = e[int(y0)] - 2, y0, rng.normal(0, 0.25)
    crack = np.zeros((h, w), np.uint8)
    seg = 3.0
    steps = max(2, int(L / seg))
    for i in range(steps):
        a = 0.6 * a + (rng.normal(0, 0.9) if rng.random() < 0.25 else rng.normal(0, 0.15))
        nx, ny = x + seg * np.cos(a), y + seg * np.sin(a)
        cv2.line(crack, (int(x), int(y)), (int(nx), int(ny)), 255, max(1, int(round(w0 * (1 - i / steps) ** 1.5))))
        x, y = nx, ny
    cf = cv2.GaussianBlur(crack.astype(np.float32) / 255, (0, 0), 1.0)
    crack = ((cf + 0.25 * fine(h, w, rng, 1.2)) > 0.5) & (cf > 0.1) & ~outside
    return comp_open(ch, crack | outside, rng), crack


def d_fold_over(ch, e, rng):
    h, w = ch[0].shape
    T, R, D = ch
    yy = np.arange(h, dtype=np.float32)
    xx = np.arange(w, dtype=np.float32)[None, :]
    Lf = px(U(rng, 15, 60)); y0 = U(rng, -0.3 * Lf, h - 0.7 * Lf)
    ta, tb = U(rng, 0.1, 0.3), U(rng, 0.1, 0.3)
    t = (yy - y0) / Lf
    f = px(U(rng, 2, 7)) * np.clip(np.minimum(t / ta, (1 - t) / tb), 0, 1)
    f = np.clip(f + px(0.15) * lf(h, 1, rng, 50)[:, 0] * (f > 0), 0, None)
    fold_x = (e + f)[:, None]
    outside = xx < e[:, None]
    gap = (xx >= e[:, None]) & (xx < fold_x)
    flap = (xx >= fold_x) & (xx < (e + 2 * f)[:, None])
    line = np.exp(-((xx - fold_x) / 1.2) ** 2) * (f[:, None] > 1)
    dbl = cv2.GaussianBlur(flap.astype(np.float32), (0, 0), 0.7)
    T = T * (1 - U(rng, 0.4, 0.55) * dbl)
    R = R + U(rng, -6, 6) * dbl
    D = D + U(rng, 60, 110) * line + 10 * np.abs(cv2.Sobel(dbl, cv2.CV_32F, 1, 0))
    return comp_open([T, R, D], gap | outside, rng, thin_ring=False), gap | flap


def visible(view, name, info):
    """Is this defect visible under the given light? Labels are only written if it is."""
    if view == "rgb":
        return True
    if view == "T":
        if name == "wrinkle":
            return False
        if name == "streak":
            return info.get("variant") == "basis_weight"
    if view == "R":
        if name == "streak":
            return info.get("variant") == "surface"
    if view == "DF":
        if name == "thin_spot":
            return False
        if name == "dirt":
            return info.get("raised", False)
        if name == "streak":
            return info.get("variant") == "surface"
    return True


DEFECT_FN = {"hole": d_hole, "pinhole": d_pinhole, "dirt": d_dirt, "shive": d_shive,
             "slime_spot": d_slime, "thin_spot": d_thin_spot, "streak": d_streak,
             "wet_spot": d_wet_spot, "wrinkle": d_wrinkle, "crease": d_crease,
             "edge_crack": d_edge_crack, "fold_over": d_fold_over}


# ============================================================================ #
# Background irregularities — normal process / imaging variation, never labelled
# ============================================================================ #
IRREGULARITIES = ["cd_profile", "cloudy_patch", "felt_mark", "speed_variation",
                  "md_banding", "uneven_light", "flutter_blur", "exposure", "pixel_columns"]
IRR_STAGE = {"cd_profile": "paper", "cloudy_patch": "paper", "felt_mark": "paper",
             "speed_variation": "paper", "md_banding": "optics", "uneven_light": "optics",
             "flutter_blur": "optics", "exposure": "sensor", "pixel_columns": "pixels"}


def irr_cd_profile(ch, rng):
    """Moisture / basis-weight profile across the web: wide and weak (unlike a streak)."""
    T, R, D = ch
    p = lf(1, T.shape[1], rng, px(U(rng, 50, 300)))
    a = U(rng, 0.04, 0.10)
    return [T * (1 + a * p), R - a * 25 * p, D]


def irr_cloudy_patch(ch, rng):
    """Large region of poorer formation, no defined edge."""
    T, R, D = ch
    H, W = T.shape
    r = px(U(rng, 50, 250))
    cx, cy = U(rng, 0, W), U(rng, 0, H)
    xx = ((np.arange(W, dtype=np.float32) - cx) / r) ** 2
    yy = ((np.arange(H, dtype=np.float32) - cy) / (r * U(rng, 0.5, 1.5))) ** 2
    M = np.exp(-(yy[:, None] + xx[None, :]))
    F2 = lf(H, W, rng, px(U(rng, 4, 12)))
    amp = U(rng, 0.05, 0.12)
    return [T * np.exp(-amp * F2 * M), R + 3 * F2 * M, D]


def irr_felt_mark(ch, rng):
    """Fine periodic twill pattern from wire / felt."""
    T, R, D = ch
    H, W = T.shape
    yy, xx = grid(H, W)
    p = px(U(rng, 0.6, 2.0))
    a = np.deg2rad(U(rng, 30, 60))
    pat = (np.sin(2 * np.pi * (xx * np.cos(a) + yy * np.sin(a)) / p) *
           np.sin(2 * np.pi * (xx * np.cos(a) - yy * np.sin(a)) / (p * U(rng, 0.8, 1.2))))
    pat *= 0.6 + 0.4 * np.abs(lf(H, W, rng, px(40)))
    return [T * (1 + U(rng, 0.015, 0.035) * pat), R + U(rng, 1.5, 3.5) * pat, D + U(rng, 0.5, 1.5) * pat]


def irr_speed_variation(ch, rng):
    """Web speed vs line rate mismatch: texture stretched / squeezed along MD."""
    H, W = ch[0].shape
    r = np.clip(1 + U(rng, 0.2, 0.4) * lf(H, 1, rng, px(U(rng, 20, 80)))[:, 0], 0.5, None)
    pos = np.cumsum(r)
    pos = ((pos - pos[0]) / (pos[-1] - pos[0]) * (H - 1)).astype(np.float32)
    map_y = np.repeat(pos[:, None], W, axis=1)
    map_x = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    return [cv2.remap(c, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for c in ch]


def irr_md_banding(ch, rng):
    """Lamp flicker / headbox pulsation: faint bands across the full width."""
    H = ch[0].shape[0]
    out = []
    for c in ch:
        if rng.random() < 0.5:
            b = lf(H, 1, rng, px(U(rng, 5, 30)))[:, 0]
        else:
            b = np.sin(2 * np.pi * np.arange(H, dtype=np.float32) / px(U(rng, 5, 40)) + U(rng, 0, 6.28))
        out.append(c * (1 + U(rng, 0.035, 0.08) * b)[:, None])
    return out


def irr_uneven_light(ch, rng):
    """LED ripple and lamp hot / cold zones across CD, independent per light."""
    W = ch[0].shape[1]
    x = np.arange(W, dtype=np.float32)
    out = []
    for c in ch:
        ripple = U(rng, 0.015, 0.04) * np.sin(2 * np.pi * x / px(U(rng, 20, 60)) + U(rng, 0, 6.28))
        zones = U(rng, 0.03, 0.06) * lf(1, W, rng, px(U(rng, 100, 300)))[0]
        out.append(c * (1 + ripple + zones)[None, :])
    return out


def irr_flutter_blur(ch, rng):
    """Web flutter: a band where the web leaves the depth of field."""
    H, W = ch[0].shape
    h = px(U(rng, 10, 60))
    yc = U(rng, 0, H)
    wy = np.exp(-(np.abs(np.arange(H, dtype=np.float32) - yc) / (h / 2)) ** 4)
    wx = np.clip(0.5 + 0.7 * lf(1, W, rng, px(200))[0], 0, 1)
    wgt = wy[:, None] * wx[None, :]
    sigma = U(rng, 1.5, 4.0)
    shifts = [1.0, 1 + U(rng, -0.05, 0.05), 1 + U(rng, -0.2, 0.2)]     # angle change affects R, DF
    return [(c * (1 - wgt) + cv2.GaussianBlur(c, (0, 0), sigma) * wgt) * (1 + (k - 1) * wgt)
            for c, k in zip(ch, shifts)]


def irr_exposure(ch, rng, strong):
    """Frame-to-frame gain / gamma drift (always mild; strong when selected)."""
    out = []
    for c in ch:
        g = U(rng, 0.82, 1.18) if strong else U(rng, 0.95, 1.05)
        gam = U(rng, 0.85, 1.15) if strong else 1.0
        out.append(255 * (np.clip(c * g, 0, 255) / 255) ** gam)
    return out


def irr_pixel_columns(ch, rng):
    """Dead / hot sensor pixels: 1-2 px columns, identical in all lights."""
    W = ch[0].shape[1]
    cols = [(int(rng.integers(0, W - 2)), int(rng.integers(1, 3)), rng.random() < 0.5,
             U(rng, 15, 40), U(rng, 0.3, 0.7)) for _ in range(int(rng.integers(1, 5)))]
    out = [c.copy() for c in ch]
    for x, w, hot, add, mul in cols:
        for c in out:
            if hot:
                c[:, x:x + w] += add
            else:
                c[:, x:x + w] *= mul
    return out


IRR_FN = {"cd_profile": irr_cd_profile, "cloudy_patch": irr_cloudy_patch, "felt_mark": irr_felt_mark,
          "speed_variation": irr_speed_variation, "md_banding": irr_md_banding,
          "uneven_light": irr_uneven_light, "flutter_blur": irr_flutter_blur,
          "pixel_columns": irr_pixel_columns}


def apply_irregularities(ch, names, stage, rng):
    for n in names:
        if IRR_STAGE[n] == stage and n in IRR_FN:
            ch = IRR_FN[n](ch, rng)
    return [c.astype(np.float32) for c in ch]


# ============================================================================ #
# Frame assembly
# ============================================================================ #
def feather(h, w, y0, x0, H, W, ramp=24):
    fy, fx = np.ones(h, np.float32), np.ones(w, np.float32)
    ry, rx = min(ramp, h // 4), min(ramp, w // 4)
    if y0 > 0 and ry:
        fy[:ry] = np.linspace(0, 1, ry)
    if y0 + h < H and ry:
        fy[-ry:] = np.minimum(fy[-ry:], np.linspace(1, 0, ry))
    if x0 > 0 and rx:
        fx[:rx] = np.linspace(0, 1, rx)
    if x0 + w < W and rx:
        fx[-rx:] = np.minimum(fx[-rx:], np.linspace(1, 0, rx))
    return fy[:, None] * fx[None, :]


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def render_frame(idx, cfg):
    rng = np.random.default_rng([cfg["seed"], idx])
    H, W = cfg["height"], cfg["width"]
    grade = cfg["grade"] if cfg["grade"] != "mixed" else GRADE_NAMES[int(rng.integers(2))]
    ch = paper_background(grade, H, W, rng)
    irng = np.random.default_rng([cfg["seed"], idx, 7])     # separate stream: same defects with/without
    irr = []
    if "force_irr" in cfg:
        irr = list(cfg["force_irr"])
    elif irng.random() < cfg["irregular"]:
        k = int(irng.integers(1, 4))
        irr = [IRREGULARITIES[i] for i in irng.choice(len(IRREGULARITIES), size=k, replace=False)]
    ch = apply_irregularities(ch, irr, "paper", irng)
    inst = np.zeros((H, W), np.uint16)
    cls_of, info_of = {}, {}

    edge = rng.random() < cfg["edge_frames"]
    e, xmin = None, 0
    if edge:
        e = (px(U(rng, 5, 150)) + px(0.4) * lf(H, 1, rng, 60)[:, 0]
             + px(0.1) * fine(H, 1, rng, 1.0)[:, 0]).astype(np.float32)
        xmin = int(e.max() + px(5))

    todo = []
    if rng.random() >= cfg["defect_free"]:
        n = int(rng.integers(1, cfg["max_defects"] + 1))
        if edge and rng.random() < 0.7:
            todo.append("edge_crack" if rng.random() < 0.5 else "fold_over")
            n -= 1
        names = list(WEIGHTS)
        p = np.array([WEIGHTS[k] for k in names]); p /= p.sum()
        todo += [names[i] for i in rng.choice(len(names), size=max(0, n), p=p)]

    placed, nid = [], 1
    for name in todo:
        if name in EDGE_CLASSES:
            ww = min(W, int(e.max() + px(40)))
            wh = H if name == "fold_over" else min(H, int(px(60)))
            y0, x0 = int(rng.integers(0, H - wh + 1)), 0
        else:
            mh, mw = WINDOW_MM[name]
            wh = H if mh is None else min(H, int(px(mh)))
            ww = min(W - xmin, int(px(mw)))
            for _ in range(8):
                y0 = int(rng.integers(0, H - wh + 1))
                x0 = int(rng.integers(xmin, W - ww + 1))
                if not any(overlaps((x0, y0, x0 + ww, y0 + wh), b) for b in placed):
                    break
        placed.append((x0, y0, x0 + ww, y0 + wh))
        sub = [c[y0:y0 + wh, x0:x0 + ww] for c in ch]
        if name in EDGE_CLASSES:
            res = DEFECT_FN[name]([s.copy() for s in sub], e[y0:y0 + wh], rng)
        else:
            res = DEFECT_FN[name]([s.copy() for s in sub], rng)
        new, m, info = res if len(res) == 3 else (*res, {})
        F = feather(wh, ww, y0, x0, H, W)
        for k in range(3):
            ch[k][y0:y0 + wh, x0:x0 + ww] = sub[k] + (new[k] - sub[k]) * F
        m = m & (F > 0.5)
        if m.sum() < cfg["min_area"]:
            continue
        inst[y0:y0 + wh, x0:x0 + ww][m] = nid
        cls_of[nid] = CLASSES.index(name)
        info_of[nid] = info
        nid += 1

    if edge:                                                # open area beyond the web edge
        cw = min(W, int(e.max() + 20))
        xx = np.arange(cw, dtype=np.float32)[None, :]
        outside = xx < e[:, None]
        sub = comp_open([c[:, :cw] for c in ch], outside, rng, thin_ring=False)
        for k in range(3):
            ch[k][:, :cw] = sub[k]
        inst[:, :cw][outside] = 0

    ch = apply_irregularities(ch, irr, "optics", irng)
    S = sensor_model(H, W, rng)
    ch = irr_exposure([c * S for c in ch], irng, "exposure" in irr)
    sd = [U(rng, 1.2, 2.0), U(rng, 1.0, 1.8), U(rng, 0.8, 1.4)]
    ch = [c + rng.normal(0, s, c.shape).astype(np.float32) for c, s in zip(ch, sd)]
    ch = apply_irregularities(ch, irr, "pixels", irng)
    out = [np.clip(c, 0, 255).astype(np.uint8) for c in ch]
    if rng.random() < 0.5:                                  # web edge on the right side
        out = [c[:, ::-1] for c in out]
        inst = inst[:, ::-1]
    out = {"T": np.ascontiguousarray(out[0]), "R": np.ascontiguousarray(out[1]),
           "DF": np.ascontiguousarray(out[2])}
    return out, np.ascontiguousarray(inst), cls_of, info_of, grade, edge, irr


# ============================================================================ #
# Labels and writing
# ============================================================================ #
def labels_from_instances(inst, cls_of, min_area):
    anns = []
    for i in np.unique(inst):
        if i == 0:
            continue
        ys, xs = np.nonzero(inst == i)
        if len(xs) < min_area:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        sub = (inst[y0:y1 + 1, x0:x1 + 1] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys, pts = [], []
        for c in cnts:
            pts.append(c.reshape(-1, 2) + [x0, y0])
            a = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) + [x0, y0]
            if len(a) >= 3:
                polys.append(a.astype(float).flatten().tolist())
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(np.concatenate(pts).astype(np.float32))
        box = cv2.boxPoints(((cx, cy), (max(rw, 2.0), max(rh, 2.0)), ang))
        anns.append({"cls": int(cls_of[int(i)]), "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                     "area": int(len(xs)), "segmentation": polys, "obb": box.tolist()})
    return anns


def write_sample(root, split, name, img, inst, cls_of, cfg):
    H, W = inst.shape
    cv2.imwrite(os.path.join(root, "images", split, name + ".png"), img)
    lut = np.zeros(max(cls_of, default=0) + 1, np.uint8)
    for i, c in cls_of.items():
        lut[i] = c + 1
    cv2.imwrite(os.path.join(root, "masks", split, name + ".png"), lut[inst])
    cv2.imwrite(os.path.join(root, "instances", split, name + ".png"), inst)
    anns = labels_from_instances(inst, cls_of, cfg["min_area"])
    with open(os.path.join(root, "labels", split, name + ".txt"), "w") as f:
        for a in anns:
            x, y, bw, bh = a["bbox"]
            f.write(f"{a['cls']} {(x + bw / 2) / W:.6f} {(y + bh / 2) / H:.6f} {bw / W:.6f} {bh / H:.6f}\n")
    with open(os.path.join(root, "labels_obb", split, name + ".txt"), "w") as f:
        for a in anns:
            p = " ".join(f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}" for x, y in a["obb"])
            f.write(f"{a['cls']} {p}\n")
    return {"file": name + ".png", "split": split, "width": W, "height": H, "anns": anns}


def tile_starts(L, s, ov):
    if L <= s:
        return [0]
    st = list(range(0, L - s + 1, max(1, s - ov)))
    if st[-1] != L - s:
        st.append(L - s)
    return st


def export_view(root, split, name, img, inst, cls_of, cfg, rng):
    meta = write_sample(root, split, name, img, inst, cls_of, cfg)
    sliced = []
    sz = cfg["slice"]
    if sz:
        H, W = inst.shape
        with_def, empty = [], []
        for y in tile_starts(H, sz, cfg["slice_overlap"]):
            for x in tile_starts(W, sz, cfg["slice_overlap"]):
                ids, cnt = np.unique(inst[y:y + sz, x:x + sz], return_counts=True)
                ok = any(i != 0 and c >= cfg["min_area"] for i, c in zip(ids, cnt))
                (with_def if ok else empty).append((y, x))
        f = cfg["defect_free"]
        k = 1 if not with_def else int(np.floor(f * len(with_def) / (1 - f) + rng.random()))
        keep = with_def + [empty[i] for i in rng.permutation(len(empty))[:k]]
        for y, x in keep:
            sliced.append(write_sample(os.path.join(root, "sliced"), split, f"{name}_y{y:04d}_x{x:05d}",
                                       np.ascontiguousarray(img[y:y + sz, x:x + sz]),
                                       np.ascontiguousarray(inst[y:y + sz, x:x + sz]), cls_of, cfg))
    return meta, sliced


def process(job):
    idx, split, cfg = job
    chans, inst, cls_of, info_of, grade, edge, irr = render_frame(idx, cfg)
    name = f"wis_{idx:06d}"
    results = {}
    for view in cfg["views"]:
        keep = {i: c for i, c in cls_of.items() if visible(view, CLASSES[c], info_of[i])}
        lut = np.zeros(max(cls_of, default=0) + 1, np.uint16)
        for i in keep:
            lut[i] = i
        inst_v = lut[inst]
        img = cv2.merge([chans["DF"], chans["R"], chans["T"]]) if view == "rgb" else chans[view]
        rng = np.random.default_rng([cfg["seed"], idx, 1])
        meta, sliced = export_view(cfg["roots"][view], split, name, img, inst_v, keep, cfg, rng)
        for m in [meta] + sliced:
            m.update(idx=idx, grade=grade, edge=edge, irregularities=";".join(irr))
        results[view] = (meta, sliced)
    return results


def write_coco(metas, split, root):
    images, anns, aid = [], [], 1
    for img_id, m in enumerate(sorted((m for m in metas if m["split"] == split), key=lambda d: d["file"]), 1):
        images.append({"id": img_id, "file_name": m["file"], "width": m["width"], "height": m["height"]})
        for a in m["anns"]:
            anns.append({"id": aid, "image_id": img_id, "category_id": a["cls"] + 1, "bbox": a["bbox"],
                         "area": a["area"], "iscrowd": 0, "segmentation": a["segmentation"],
                         "attributes": {"obb": a["obb"]}})
            aid += 1
    with open(os.path.join(root, "annotations", f"instances_{split}.json"), "w") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}, f)


def write_meta_files(root, metas):
    for s in ("train", "val", "test"):
        write_coco(metas, s, root)
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    with open(os.path.join(root, "dataset.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(root)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}")
    with open(os.path.join(root, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")
    with open(os.path.join(root, "manifest.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "split", "grade", "edge_frame", "irregularities", "n_defects", "classes"])
        for m in sorted(metas, key=lambda d: d["file"]):
            wr.writerow([m["file"], m["split"], m["grade"], int(m["edge"]), m["irregularities"],
                         len(m["anns"]), ";".join(CLASSES[a["cls"]] for a in m["anns"])])


def make_preview(root, metas, n, width=2048):
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255), (0, 255, 0), (255, 128, 0),
              (128, 255, 128), (255, 255, 255), (0, 200, 255), (200, 100, 255), (100, 255, 255), (255, 80, 80)]
    rows = []
    for m in sorted(metas, key=lambda d: d["idx"])[:n]:
        im = cv2.imread(os.path.join(root, "images", m["split"], m["file"]), cv2.IMREAD_COLOR)
        for a in m["anns"]:
            x, y, bw, bh = a["bbox"]
            col = colors[a["cls"]]
            cv2.rectangle(im, (x - 4, y - 4), (x + bw + 4, y + bh + 4), col, 6)
            cv2.putText(im, CLASSES[a["cls"]], (x, max(40, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 3, cv2.LINE_AA)
        sc = width / im.shape[1]
        im = cv2.resize(im, (width, int(im.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        bar = np.zeros((26, width, 3), np.uint8)
        txt = f"{m['file']} | {m['grade']}{' | edge' if m['edge'] else ''}" + \
              (f" | irregular: {m['irregularities']}" if m['irregularities'] else "") + " | " + \
              (", ".join(CLASSES[a['cls']] for a in m["anns"]) or "defect-free")
        cv2.putText(bar, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        rows += [bar, im, np.full((6, width, 3), 255, np.uint8)]
    cv2.imwrite(os.path.join(root, "preview.png"), np.vstack(rows))


def main():
    p = argparse.ArgumentParser(description="WIS synthetic defect generator (paper, line-scan, 3 lights)")
    p.add_argument("--n", type=int, default=500, help="number of frames")
    p.add_argument("--out", default="wis_dataset")
    p.add_argument("--width", type=int, default=8196, help="frame width in px (cross direction)")
    p.add_argument("--height", type=int, default=1024, help="frame height in px (machine direction)")
    p.add_argument("--mm-per-px", type=float, default=0.1)
    p.add_argument("--grade", default="mixed", choices=["mixed"] + GRADE_NAMES)
    p.add_argument("--defect-free", type=float, default=0.10)
    p.add_argument("--max-defects", type=int, default=10)
    p.add_argument("--edge-frames", type=float, default=0.20, help="share of frames showing the web edge")
    p.add_argument("--irregular", type=float, default=0.4,
                   help="share of frames with 1-3 background irregularities (unlabelled)")
    p.add_argument("--min-area", type=int, default=30, help="min labelled defect area in px")
    p.add_argument("--slice", type=int, default=0, help="also export square crops of this size (e.g. 1024)")
    p.add_argument("--slice-overlap", type=int, default=128)
    p.add_argument("--mode", default="separate", choices=["separate", "T", "R", "DF", "rgb"],
                   help="separate = 3 grayscale datasets (T/, R/, DF/); T/R/DF = one light only; rgb = 3-channel composite")
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview", type=int, default=0, help="save preview.png of the first N frames")
    a = p.parse_args()

    _init_worker(a.mm_per_px)
    views = ["T", "R", "DF"] if a.mode == "separate" else [a.mode]
    roots = {v: (os.path.join(a.out, v) if a.mode == "separate" else a.out) for v in views}
    cfg = {"out": a.out, "seed": a.seed, "width": a.width, "height": a.height, "grade": a.grade,
           "defect_free": a.defect_free, "max_defects": a.max_defects, "edge_frames": a.edge_frames,
           "min_area": a.min_area, "slice": a.slice, "slice_overlap": a.slice_overlap,
           "irregular": a.irregular, "views": views, "roots": roots}
    all_roots = list(roots.values()) + ([os.path.join(r, "sliced") for r in roots.values()] if a.slice else [])
    for r in all_roots:
        for sub in ["images", "labels", "labels_obb", "masks", "instances"]:
            for s in ("train", "val", "test"):
                os.makedirs(os.path.join(r, sub, s), exist_ok=True)
        os.makedirs(os.path.join(r, "annotations"), exist_ok=True)

    frac = np.array(a.split) / sum(a.split)
    order = np.random.default_rng(a.seed).permutation(a.n)
    n_tr, n_va = int(frac[0] * a.n), int(frac[1] * a.n)
    split_of = {int(i): ("train" if r < n_tr else "val" if r < n_tr + n_va else "test") for r, i in enumerate(order)}
    jobs = [(i, split_of[i], cfg) for i in range(a.n)]

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    metas = {v: [] for v in views}
    sliced = {v: [] for v in views}
    with Pool(a.workers, initializer=_init_worker, initargs=(a.mm_per_px,)) as pool:
        it = pool.imap_unordered(process, jobs, chunksize=2)
        if tqdm:
            it = tqdm(it, total=a.n, desc="Frames")
        for k, res in enumerate(it, 1):
            for v, (m, sl) in res.items():
                metas[v].append(m)
                sliced[v] += sl
            if not tqdm and k % 100 == 0:
                print(f"  {k}/{a.n} frames")

    print(f"\nDone: {a.n} frames ({a.width}x{a.height}), mode = {a.mode}")
    for v in views:
        write_meta_files(roots[v], metas[v])
        if a.slice:
            write_meta_files(os.path.join(roots[v], "sliced"), sliced[v])
        if a.preview:
            make_preview(roots[v], metas[v], a.preview)
        counts = np.zeros(len(CLASSES), int)
        for m in metas[v]:
            for an in m["anns"]:
                counts[an["cls"]] += 1
        print(f"\n[{v}] -> {os.path.abspath(roots[v])}" + (f"  (+{len(sliced[v])} crops)" if a.slice else ""))
        print("  " + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, counts)))
        print(f"  defect-free frames: {sum(1 for m in metas[v] if not m['anns'])}")


if __name__ == "__main__":
    main()
