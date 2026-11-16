"""
Layer 4b - new, procedurally generated defects.

Every defect is a RELATIVE SIGNAL added to the paper reflectance (paper <- paper + s), so:
  * the synthetic paper texture continues unchanged inside the defect;
  * edges are soft with a physically realistic width (set in mm, independent of defect size);
  * defects are generated directly at their target size (no enlargement blur).
Relief defects (scratch, blister, wrinkle) are shaded with one light direction per frame.

Signals are drawn in physical (square) pixels at mm_per_px, then stretched along MD (image y)
by the frame's md_stretch, exactly like the defects transplanted from the samples.
"""
import numpy as np, cv2

TRANSLUCENT_CLASSES = ["oil spot", "fish eye", "slime spot", "slime hole"]
THIN_CLASSES = ["blade scratch", "calender cut", "felt hair"]
# transmission defects: openings through the sheet where light from behind the web shines through
# (bright in the image): pinholes, small holes, small tears
TRANSMISSION_CLASSES = ["transmission pinhole", "transmission hole", "transmission tear"]
NEW_CLASSES = (["scratch", "torn hole", "coating skip", "blister", "shive", "wrinkle", "dark spot"]
               + TRANSLUCENT_CLASSES + THIN_CLASSES + TRANSMISSION_CLASSES)

# size tiers per class: longest side in mm (a class may not exist in every tier)
NEW_SIZES = {
    "scratch":      {"small": (4, 10),   "medium": (10, 25), "large": (25, 60)},
    "torn hole":    {"small": (1.5, 5),  "medium": (5, 12)},
    "coating skip": {"small": (2, 5),    "medium": (5, 20),  "large": (20, 45)},
    "blister":      {"small": (1.5, 5),  "medium": (5, 12),  "large": (12, 25)},
    "shive":        {"small": (2, 5),    "medium": (5, 14),  "large": (14, 28)},
    "wrinkle":      {"medium": (15, 40), "large": (40, 90)},
    "dark spot":    {"small": (1.2, 5),  "medium": (5, 20),  "large": (20, 45)},
    "oil spot":     {"small": (2, 5),    "medium": (5, 20),  "large": (20, 35)},
    "fish eye":     {"small": (1, 4),    "medium": (4, 8)},
    "slime spot":   {"small": (2, 5),    "medium": (5, 18),  "large": (18, 30)},
    "slime hole":   {"small": (2, 5),    "medium": (5, 15)},
    # thin defects: size = length (blade scratches in continuous-web mode run through several frames)
    "blade scratch": {"small": (20, 50), "medium": (50, 120), "large": (120, 400)},
    "calender cut":  {"small": (5, 15),  "medium": (15, 40),  "large": (40, 90)},
    "felt hair":     {"small": (3, 8),   "medium": (8, 20),   "large": (20, 40)},
    # transmission defects: size = the opening (the glow of scattered light around it is larger)
    "transmission pinhole": {"small": (0.3, 1.2)},
    "transmission hole":    {"small": (1.0, 5.0), "medium": (5.0, 10.0)},
    "transmission tear":    {"small": (2.0, 5.0), "medium": (5.0, 15.0)},
}
# texture gain inside translucent defects: >0 = paper structure shows more (look-through),
# <0 = less (slime: non-fibrous area). Applied to the local paper texture by the defect layer.


# ---------------------------------------------------------------- helpers
def _lf(h, w, rng, scale):
    scale = max(2.0, float(scale))
    s = rng.standard_normal((max(2, int(h / scale) + 2), max(2, int(w / scale) + 2))).astype(np.float32)
    b = cv2.resize(s, (w, h), interpolation=cv2.INTER_CUBIC)
    return b / (b.std() + 1e-6)


def fbm(h, w, rng, scale, octaves=4):
    out, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    for i in range(octaves):
        out += amp * _lf(h, w, rng, scale / (2 ** i)); tot += amp * amp; amp *= 0.55
    return out / np.sqrt(tot)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def soft_alpha(mask, edge_px):
    """0..1 alpha with a smooth edge of ~edge_px (10-90%) centred on the mask boundary."""
    m = mask.astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    sd = din - dout                                   # >0 inside
    return smoothstep(sd / max(edge_px / 0.8, 0.5) + 0.5)


def relief_shading(height, light_angle):
    """Directional derivative of a height map (front light from light_angle, radians, 0 = from +y)."""
    height = np.asarray(height, np.float32)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return -(np.sin(light_angle) * gx + np.cos(light_angle) * gy)


def random_polyline(L, theta, curv, rng, step=1.0):
    n = max(3, int(L / step))
    a, x, y, pts = theta, 0.0, 0.0, []
    c = curv
    for i in range(n):
        pts.append((x, y)); a += c + rng.normal(0, abs(curv) * 0.5 + 1e-4); x += step * np.cos(a); y += step * np.sin(a)
        if rng.random() < 0.01: c = -c
    return np.array(pts, np.float32)


def canvas_for(pts, pad):
    mn = pts.min(0) - pad; size = (pts.max(0) - mn + pad).astype(int) + 1
    return mn, (int(size[1]), int(size[0]))


def dist_to_polyline(pts, mn, shape):
    """Distance map to a polyline, plus normalised position along the line (0..1) of the nearest vertex."""
    P = np.round(pts - mn).astype(np.int32)
    line = np.full(shape, 255, np.uint8)
    for i in range(len(P) - 1):
        cv2.line(line, tuple(P[i]), tuple(P[i + 1]), 0, 1)
    d = cv2.distanceTransform(line, cv2.DIST_L2, 5)
    verts = np.full(shape, 255, np.uint8)
    Pc = np.clip(P, 0, [shape[1] - 1, shape[0] - 1])
    verts[Pc[:, 1], Pc[:, 0]] = 0
    _, lab = cv2.distanceTransformWithLabels(verts, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # label -> vertex index: labels are assigned to zero pixels in raster order
    zy, zx = np.nonzero(verts == 0)
    lab_of_zero = lab[zy, zx]
    first_idx = {}
    for i, (x, y) in enumerate(Pc):
        first_idx.setdefault((y, x), i)
    lut = np.zeros(lab.max() + 1, np.float32)
    for (yy, xx), lb in zip(zip(zy, zx), lab_of_zero):
        lut[lb] = first_idx[(yy, xx)] / max(1, len(P) - 1)
    return d.astype(np.float32), lut[lab]


# ---------------------------------------------------------------- generator
class NewDefectGenerator:
    def __init__(self, mm_per_px=0.112):
        self.mmpx = mm_per_px

    def px(self, mm):
        return mm / self.mmpx

    def render(self, cls, tier, rng, md_stretch=1.8, frame_h=1024, light_angle=0.0, max_height_share=0.9, size_mm=None):
        if size_mm is None:
            lo, hi = NEW_SIZES[cls][tier]
            size_mm = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        self._tier = tier
        fn = getattr(self, "_" + cls.replace(" ", "_"))
        self._tex = None
        s = fn(self.px(size_mm), rng, light_angle)
        tex = self._tex                                              # (alpha, gain) for translucent defects
        # make sure it fits the frame height after the MD stretch (shrink if needed)
        max_h = max_height_share * frame_h / md_stretch
        if s.shape[0] > max_h:
            k = max_h / s.shape[0]
            s = cv2.resize(s, (max(3, int(s.shape[1] * k)), max(3, int(s.shape[0] * k))), interpolation=cv2.INTER_AREA)
            if tex is not None: tex = (cv2.resize(tex[0], (s.shape[1], s.shape[0]), interpolation=cv2.INTER_AREA), tex[1])
        # MD stretch
        s = cv2.resize(s, (s.shape[1], max(3, int(round(s.shape[0] * md_stretch)))), interpolation=cv2.INTER_LINEAR)
        if tex is not None:
            tex = (cv2.resize(tex[0], (s.shape[1], s.shape[0]), interpolation=cv2.INTER_LINEAR), tex[1])
        # final border taper (never a hard patch edge)
        h, w = s.shape
        def ramp(n, k=6):
            r = np.ones(n, np.float32); k = min(k, n // 3); t = smoothstep(np.linspace(0, 1, k)); r[:k] = t; r[-k:] = t[::-1]; return r
        taper = ramp(h)[:, None] * ramp(w)[None, :]
        self.last_tex = None if tex is None else ((tex[0] * taper).astype(np.float32), tex[1])
        return (s * taper).astype(np.float32)

    # --- scratch: thin groove, relief edge, varying along its length, tapered ends, sometimes dashed
    def _scratch(self, L, rng, light):
        theta = rng.uniform(0, np.pi)
        pts = random_polyline(L, theta, rng.uniform(0.0005, 0.004) * rng.choice([-1, 1]), rng)
        pad = self.px(1.5)
        mn, shape = canvas_for(pts, pad)
        d, pos = dist_to_polyline(pts, mn, shape)
        wid = self.px(rng.uniform(0.1, 0.3) if L < self.px(10) else rng.uniform(0.12, 0.4))
        groove = np.exp(-(d / wid) ** 2)
        h, w = shape
        short = L < self.px(12)
        var = 0.2 if short else 0.45                               # short scratches are more even
        along = np.clip(1 - var + var * fbm(h, w, rng, self.px(rng.uniform(3, 10)), 3), 0.08, 1.2)
        taper = smoothstep(np.minimum(pos, 1 - pos) / 0.15)
        if not short and rng.random() < 0.25:                       # interrupted scratch (longer ones only)
            along *= (fbm(h, w, rng, self.px(rng.uniform(1.5, 4)), 2) > -0.3)
        mod = along * taper
        shade = relief_shading(-groove, light)
        shade /= (np.abs(shade).max() + 1e-6)
        s = (-rng.uniform(0.05, 0.18) * groove + rng.uniform(0.04, 0.12) * shade) * mod
        return cv2.GaussianBlur(s, (0, 0), 0.5)

    # --- torn hole: irregular torn polygon (straight-ish torn segments, sharp corners), short radiating
    #     cracks, dark rim of torn fibres, fibres sticking out across the edge
    def _torn_hole(self, D, rng, light):
        R = D / 2
        pad = int(self.px(4) + 0.5 * R)
        n = int(2 * (R * 1.35 + pad)); c = n / 2
        nv = int(np.clip(rng.integers(7, 12) + R / self.px(2.0), 7, 26))     # bigger tears: more corners
        angs = np.sort(rng.uniform(0, 2 * np.pi, nv))
        rad = R * rng.uniform(0.6, 1.25, nv)
        asp = rng.uniform(1.0, 1.7); rot = rng.uniform(0, np.pi)
        pts = []
        for a_, r_ in zip(angs, rad):
            x, y = r_ * np.cos(a_), r_ * np.sin(a_) / asp
            pts.append((c + x * np.cos(rot) - y * np.sin(rot), c + x * np.sin(rot) + y * np.cos(rot)))
        poly = np.array(pts, np.float32)
        # subdivide edges and jitter slightly: torn, not perfectly straight
        fine = []
        for i in range(nv):
            p0, p1 = poly[i], poly[(i + 1) % nv]
            seg = max(2, int(np.hypot(*(p1 - p0)) / self.px(0.6)))
            for t in np.linspace(0, 1, seg, endpoint=False):
                q = p0 + t * (p1 - p0) + rng.normal(0, self.px(0.06) + 0.012 * R, 2)
                fine.append(q)
        fine = np.array(fine, np.float32)
        m8 = np.zeros((n, n), np.uint8)
        cv2.fillPoly(m8, [np.round(fine * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        hole = m8 > 127
        # 1-3 short cracks radiating from corners
        crack = np.zeros((n, n), np.uint8)
        for _ in range(int(rng.integers(0, 4))):
            i = int(rng.integers(nv)); p = poly[i]; d = p - c; d /= np.linalg.norm(d) + 1e-6
            L = self.px(rng.uniform(0.5, 2.5)) + 0.25 * R * rng.random()
            a0 = np.arctan2(d[1], d[0]) + rng.normal(0, 0.3); x, y = p
            for _k in range(int(L / 2)):
                a0 += rng.normal(0, 0.25); nx, ny = x + 2 * np.cos(a0), y + 2 * np.sin(a0)
                cv2.line(crack, (int(x), int(y)), (int(nx), int(ny)), 255, max(1, int(self.px(0.12))), cv2.LINE_AA); x, y = nx, ny
        opening = hole | (crack > 100)
        edge = self.px(rng.uniform(0.1, 0.25))
        a_in = soft_alpha(opening, edge)
        din = cv2.distanceTransform((~opening).astype(np.uint8), cv2.DIST_L2, 5)
        rim_w = self.px(rng.uniform(0.25, 0.7))
        rim = np.exp(-(din / rim_w) ** 2) * (1 - a_in) * np.clip(0.6 + 0.5 * fbm(n, n, rng, self.px(0.6), 2), 0, 1.3)
        fib = np.zeros((n, n), np.uint8)
        ys, xs = np.nonzero(cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))
        for _ in range(int(rng.integers(4, 18))):
            if len(xs) == 0: break
            i = int(rng.integers(len(xs))); a = np.arctan2(ys[i] - c, xs[i] - c) + rng.normal(0, 0.9) + (np.pi if rng.random() < 0.5 else 0)
            Lf = self.px(rng.uniform(0.4, 1.8))
            cv2.line(fib, (int(xs[i]), int(ys[i])), (int(xs[i] + Lf * np.cos(a)), int(ys[i] + Lf * np.sin(a))), 255, 1, cv2.LINE_AA)
        fib = cv2.GaussianBlur(fib.astype(np.float32) / 255, (0, 0), 0.6)
        # --- light passing through the opening is not uniform:
        din_in = cv2.distanceTransform(opening.astype(np.uint8), cv2.DIST_L2, 5)
        mw = min(self.px(rng.uniform(0.3, 0.8)), 0.2 * R)                  # margin scales with small holes
        margin = 1 - np.exp(-din_in / max(mw, 1.0))                         # thin translucent paper at the torn edge
        centre = 1 - 0.25 * np.clip(1 - din_in / (din_in.max() + 1e-6), 0, 1) ** 2   # brighter towards the middle
        trans = np.clip(0.8 + 0.2 * fbm(n, n, rng, self.px(rng.uniform(1.0, 3.0)), 3), 0.5, 1.1)
        # residual fibres spanning the opening (semi-transparent strands)
        web = np.zeros((n, n), np.uint8)
        hy, hx = np.nonzero(hole)
        for _ in range(int(rng.integers(2, 6) + R / self.px(1.5))):
            if len(hx) == 0: break
            i = int(rng.integers(len(hx))); a = rng.uniform(0, np.pi); Lw = R * rng.uniform(0.6, 2.0)
            p0 = (int(hx[i] - Lw * np.cos(a)), int(hy[i] - Lw * np.sin(a))); p1 = (int(hx[i] + Lw * np.cos(a)), int(hy[i] + Lw * np.sin(a)))
            cv2.line(web, p0, p1, int(rng.uniform(80, 200)), 1, cv2.LINE_AA)
        web = cv2.GaussianBlur(web.astype(np.float32) / 255, (0, 0), 0.7)
        bright = rng.uniform(0.28, 0.45)                                   # ~+50-85 GL, like the real holes
        inside = a_in * margin * centre * trans * (1 - 0.7 * web) * (1 - 0.8 * fib)
        return bright * inside - rng.uniform(0.05, 0.14) * rim - rng.uniform(0.04, 0.1) * fib * (1 - a_in)

    # --- coating skip: irregular patch, slightly darker (or lighter), rougher base texture showing through
    def _coating_skip(self, D, rng, light):
        a = D / 2; b = a / rng.uniform(1.0, 2.5)
        pad = int(self.px(3)); n_h, n_w = int(2 * a + 2 * pad), int(2 * a + 2 * pad)
        yy, xx = np.mgrid[0:n_h, 0:n_w].astype(np.float32); cy, cx = n_h / 2, n_w / 2
        rot = rng.uniform(0, np.pi)
        ex = (xx - cx) * np.cos(rot) + (yy - cy) * np.sin(rot); ey = -(xx - cx) * np.sin(rot) + (yy - cy) * np.cos(rot)
        base = 1 - np.sqrt((ex / a) ** 2 + (ey / b) ** 2)
        field = base + 0.35 * fbm(n_h, n_w, rng, max(a, b) / 1.5, 4)
        mask = field > 0.1
        n_lab, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if n_lab > 2: mask = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
        if rng.random() < 0.5:                                     # islands of coating left inside
            mask &= ~(fbm(n_h, n_w, rng, self.px(rng.uniform(0.8, 2.5)), 3) > 1.3)
        alpha = soft_alpha(mask, self.px(rng.uniform(0.3, 1.0)))
        tone = rng.uniform(-0.10, -0.03) if rng.random() < 0.75 else rng.uniform(0.02, 0.05)
        rough = rng.uniform(0.006, 0.018) * fbm(n_h, n_w, rng, self.px(rng.uniform(0.4, 1.2)), 3)
        return alpha * (tone + rough)

    # --- blister: raised dome lit from one side, faint rim
    def _blister(self, D, rng, light):
        R = D / 2; asp = rng.uniform(1.0, 1.35)
        pad = int(self.px(2)); n = int(2 * R * asp + 2 * pad)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32); c = n / 2
        rot = rng.uniform(0, np.pi)
        ex = (xx - c) * np.cos(rot) + (yy - c) * np.sin(rot); ey = -(xx - c) * np.sin(rot) + (yy - c) * np.cos(rot)
        r = np.sqrt((ex / (R * asp)) ** 2 + (ey / R) ** 2) * (1 + 0.04 * _lf(n, n, rng, R / 2))
        dome = np.sqrt(np.clip(1 - r ** 2, 0, 1)) * R * rng.uniform(0.15, 0.35)
        dome = cv2.GaussianBlur(dome, (0, 0), self.px(rng.uniform(0.08, 0.2)))
        shade = relief_shading(dome, light)
        shade = shade / (np.abs(shade).max() + 1e-6)
        ring = np.exp(-((r - 1) / (self.px(0.25) / R)) ** 2)
        inside = soft_alpha(r < 1, self.px(0.2))
        return rng.uniform(0.06, 0.16) * shade - rng.uniform(0.02, 0.06) * ring + rng.uniform(-0.03, 0.03) * inside

    # --- shive: dark spindle-shaped fibre bundle with lengthwise streaks, aligned mostly along MD
    def _shive(self, L, rng, light):
        W = L / rng.uniform(5, 18)
        W = float(np.clip(W, self.px(0.25), self.px(1.6)))
        theta = np.pi / 2 + rng.normal(0, 0.35)                      # MD = image y
        pad = int(W * 2 + self.px(1)); n_h = int(abs(L * np.sin(theta)) + W * 2 + 2 * pad); n_w = int(abs(L * np.cos(theta)) + W * 2 + 2 * pad)
        yy, xx = np.mgrid[0:n_h, 0:n_w].astype(np.float32); cy, cx = n_h / 2, n_w / 2
        u = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
        v = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
        v = v - rng.uniform(-0.08, 0.08) * (u ** 2) / max(L / 2, 1)   # slight bend
        half = W / 2 * np.clip(1 - (2 * u / L) ** 2, 0, 1) ** 0.7
        mask = np.abs(v) < half
        alpha = soft_alpha(mask, self.px(rng.uniform(0.08, 0.2)))
        streaks = np.interp(v.ravel(), np.linspace(-W, W, 16), rng.uniform(0.55, 1.0, 16)).reshape(v.shape)
        mottle = 0.85 + 0.15 * fbm(n_h, n_w, rng, self.px(0.5), 2)
        return -rng.uniform(0.15, 0.45) * alpha * streaks * mottle

    # --- wrinkle: long soft ridge (or set of ridges) with light/dark sides
    def _wrinkle(self, L, rng, light):
        theta = np.pi / 2 + rng.normal(0, 0.5) if rng.random() < 0.6 else rng.uniform(0, np.pi)
        pts = random_polyline(L, theta, rng.uniform(0.0005, 0.003) * rng.choice([-1, 1]), rng, step=2.0)
        sig = self.px(rng.uniform(0.6, 2.5))
        nset = 1 if rng.random() < 0.65 else int(rng.integers(2, 4))
        gap = sig * rng.uniform(2.5, 4.0)
        pad = sig * 4 + gap * nset
        mn, shape = canvas_for(pts, pad)
        h, w = shape
        height = np.zeros(shape, np.float32)
        nrm = np.array([-np.sin(theta), np.cos(theta)], np.float32)
        for j in range(nset):
            off = (j - (nset - 1) / 2) * gap
            d, pos = dist_to_polyline(pts + off * nrm, mn, shape)
            env = smoothstep(np.minimum(pos, 1 - pos) / 0.2)
            height += rng.uniform(0.7, 1.0) * np.exp(-(d / sig) ** 2) * env
        height *= np.clip(0.75 + 0.35 * fbm(h, w, rng, self.px(8), 2), 0.3, 1.2)
        shade = relief_shading(height * sig, light)
        shade = shade / (np.abs(shade).max() + 1e-6)
        return rng.uniform(0.03, 0.09) * shade - rng.uniform(0.0, 0.02) * height

    # --- dark spot, calibrated on the Basler frame 0008:
    #     smudge  : soft oval stain, flat interior 8-20 % darker, edge ~0.15 x size (2 mm for a 14 mm spot),
    #               gentle mottling, sometimes a darker core
    #     speck   : tiny sharp particle core(s) 15-40 % darker inside a faint 2-5 mm halo 2.5-7 % darker
    def _dark_spot(self, D, rng, light):
        if D < self.px(5) and rng.random() < 0.85:                       # small ones: mostly halo + core
            return self._dark_speck(D, rng)
        a = D / 2; b = a / rng.uniform(1.0, 1.6)
        pad = int(self.px(3) + 0.25 * a); n = int(2 * a + 2 * pad)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32); c = n / 2
        rot = rng.uniform(0, np.pi)
        ex = (xx - c) * np.cos(rot) + (yy - c) * np.sin(rot); ey = -(xx - c) * np.sin(rot) + (yy - c) * np.cos(rot)
        r = np.sqrt((ex / a) ** 2 + (ey / b) ** 2)
        field = 1 - r + 0.07 * fbm(n, n, rng, max(a, b) / 1.2, 3)            # smooth oval, gently irregular
        mask = field > 0
        k, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if k > 2: mask = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
        edge = float(np.clip(rng.uniform(0.2, 0.3) * D, self.px(0.6), self.px(6.0)))   # real: ~2.2 mm on a 9 mm spot
        alpha = soft_alpha(mask, edge)
        depth = rng.uniform(0.08, 0.18) * (0.6 if D < self.px(5) else 1.0)
        mottle = 1 + rng.uniform(0.04, 0.10) * fbm(n, n, rng, self.px(rng.uniform(2.0, 5.0)), 3)
        s = -depth * alpha * mottle
        if rng.random() < 0.35:                                        # darker core
            sig = 0.45 * min(a, b)
            s -= depth * rng.uniform(0.3, 0.8) * np.exp(-((ex / (sig * a / b)) ** 2 + (ey / sig) ** 2) / 2) * alpha
        return s.astype(np.float32)

    def _dark_speck(self, D, rng):
        R = D / 2
        pad = int(self.px(2)); n = int(2 * R + 2 * pad); c = n / 2
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        r = np.hypot(xx - c, (yy - c) * rng.uniform(0.8, 1.25)) / R
        halo_mask = (1 - r + 0.25 * fbm(n, n, rng, R / 1.5, 3)) > 0
        halo = soft_alpha(halo_mask, float(np.clip(0.3 * R, self.px(0.5), self.px(1.2))))
        s = -rng.uniform(0.02, 0.05) * halo * (1 + 0.2 * fbm(n, n, rng, self.px(0.8), 2))
        for _ in range(int(rng.choice([1, 1, 1, 2, 3]))):                # particle core(s)
            cr = self.px(rng.uniform(0.1, 0.35)); off = rng.normal(0, 0.25 * R, 2)
            core = np.zeros((n, n), np.uint8)
            cv2.ellipse(core, (int(c + off[0]), int(c + off[1])), (max(1, int(cr * rng.uniform(0.7, 1.3))), max(1, int(cr * rng.uniform(0.7, 1.3)))),
                        rng.uniform(0, 180), 0, 360, 255, -1, cv2.LINE_AA)
            s -= rng.uniform(0.12, 0.32) * cv2.GaussianBlur(core.astype(np.float32) / 255, (0, 0), self.px(0.08))
        return s.astype(np.float32)

    # ------------------------------------------------------------------ translucent defects
    # Authored for FRONT lighting: translucent = less reflected light (some passes through to the
    # dark space behind the web) -> darker, but less dark than a real opening.

    def _drop_shape(self, D, rng, irregular=0.05, aspect=(1.0, 1.35)):
        a = D / 2; b = a / rng.uniform(*aspect)
        pad = int(self.px(2.5) + 0.3 * a); n = int(2 * a + 2 * pad); c = n / 2
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32); rot = rng.uniform(0, np.pi)
        ex = (xx - c) * np.cos(rot) + (yy - c) * np.sin(rot); ey = -(xx - c) * np.sin(rot) + (yy - c) * np.cos(rot)
        r = np.sqrt((ex / a) ** 2 + (ey / b) ** 2)
        field = 1 - r + irregular * fbm(n, n, rng, max(a, b) / 1.2, 3)
        mask = field > 0
        k, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        if k > 2: mask = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
        return mask, n, c, r

    # oil / liquid-drop spot: regular (round) shape, even translucency, faint darker "coffee ring",
    # sometimes satellite droplets; paper structure shows through more strongly
    def _oil_spot(self, D, rng, light):
        mask, n, c, r = self._drop_shape(D, rng, irregular=0.04)
        for _ in range(int(rng.choice([0, 0, 1, 2, 3]))):            # satellite droplets
            rr = D / 2 * rng.uniform(0.08, 0.25); ang = rng.uniform(0, 2 * np.pi); dist = D / 2 * rng.uniform(1.1, 1.6)
            cx, cy = int(c + dist * np.cos(ang)), int(c + dist * np.sin(ang))
            m2 = np.zeros((n, n), np.uint8); cv2.circle(m2, (cx, cy), max(1, int(rr)), 1, -1)
            mask |= m2.astype(bool)
        alpha = soft_alpha(mask, self.px(rng.uniform(0.2, 0.6)))
        din = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        ring = np.exp(-din / self.px(rng.uniform(0.3, 0.8))) * alpha
        t = rng.uniform(0.08, 0.25)
        s = -t * alpha * (1 + 0.05 * fbm(n, n, rng, self.px(3), 2)) - rng.uniform(0.0, 0.5) * t * ring
        self._tex = (alpha, rng.uniform(0.6, 1.5))
        return s.astype(np.float32)

    # fish eye: small glazed translucent spot, crisp edge, small bright glint from the glazed surface
    def _fish_eye(self, D, rng, light):
        mask, n, c, r = self._drop_shape(D, rng, irregular=0.03, aspect=(1.0, 1.2))
        alpha = soft_alpha(mask, self.px(rng.uniform(0.08, 0.2)))
        t = rng.uniform(0.10, 0.25)
        s = -t * alpha
        gy, gx = c + rng.normal(0, 0.15) * D / 2 * np.array([np.cos(light), np.sin(light)])
        glint = np.zeros((n, n), np.float32)
        cv2.circle(glint, (int(gx), int(gy)), max(1, int(D * rng.uniform(0.06, 0.15))), 1.0, -1, cv2.LINE_AA)
        g_ = cv2.GaussianBlur(glint, (0, 0), max(0.8, D * 0.04)); g_ /= g_.max() + 1e-6
        s += (t + rng.uniform(0.03, 0.12)) * g_ * alpha                 # glint slightly brighter than the paper
        self._tex = (alpha, rng.uniform(0.2, 0.6))
        return s.astype(np.float32)

    def _slime_shape(self, D, rng):
        mask, n, c, r = self._drop_shape(D, rng, irregular=0.22, aspect=(1.0, 1.8))
        return mask, n, c

    # slime spot: irregular, lobed, translucent deposit; densest in the middle and thinning out
    # gradually towards an uneven edge (no cut-out outline), patchy inside
    def _slime_spot(self, D, rng, light):
        mask, n, c = self._slime_shape(D, rng)
        edge = float(np.clip(rng.uniform(0.12, 0.25) * D, self.px(0.6), self.px(2.0)))
        din = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        dout = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        sd = (din - dout) / edge
        dens = 1 / (1 + np.exp(-2.2 * (sd - 0.3)))                           # gradual fall-off across the edge
        dens *= 0.7 + 0.3 * np.clip(din / (din.max() + 1e-6), 0, 1) ** 0.5    # denser towards the middle
        dens *= np.clip(1 + 0.3 * fbm(n, n, rng, self.px(rng.uniform(0.8, 2.5)), 3), 0.4, 1.5)   # patchy
        s = -rng.uniform(0.10, 0.28) * dens
        self._tex = (np.clip(dens, 0, 1), -rng.uniform(0.1, 0.3))            # mild loss of paper structure
        return s.astype(np.float32)

    # slime hole: a ragged opening (dark space behind the web) with translucent slime around it that is
    # densest at the hole edge and thins out gradually - no outline, extent varies by direction, patchy
    def _slime_hole(self, D, rng, light):
        n = int(D * 1.6 + self.px(6)); c = n / 2
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        hr = D / 2 * rng.uniform(0.35, 0.6)                                   # opening radius
        rot = rng.uniform(0, np.pi); asp = rng.uniform(1.0, 1.7)
        ox, oy = rng.normal(0, 0.12 * D, 2)                                   # opening off-centre
        ex = (xx - c - ox) * np.cos(rot) + (yy - c - oy) * np.sin(rot); ey = -(xx - c - ox) * np.sin(rot) + (yy - c - oy) * np.cos(rot)
        field = 1 - np.sqrt((ex / hr) ** 2 + (ey * asp / hr) ** 2) + 0.4 * fbm(n, n, rng, max(3.0, hr / 1.5), 3)
        hole = field > 0
        kk, lab, st, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8))
        if kk > 2: hole = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
        a_hole = soft_alpha(hole, self.px(rng.uniform(0.08, 0.2)))
        # slime: decays with distance from the opening; how far it reaches varies around the hole
        d_out = cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 5)
        reach = min(self.px(rng.uniform(0.5, 2.5)), 0.5 * D) * np.clip(1 + 0.7 * fbm(n, n, rng, max(4.0, D / 3), 2), 0.2, 2.5)
        dens = np.exp(-(d_out / reach) ** rng.uniform(1.0, 1.6))
        dens *= np.clip(1 + 0.4 * fbm(n, n, rng, self.px(rng.uniform(0.4, 1.2)), 3), 0.2, 1.7)      # patchy deposit
        slime = np.clip(dens, 0, 1.3) * (1 - a_hole)
        fib = np.zeros((n, n), np.uint8)
        ey_, ex_ = np.nonzero(cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))
        for _ in range(int(rng.integers(1, 6))):
            if len(ex_) == 0: break
            i = int(rng.integers(len(ex_))); a = rng.uniform(0, np.pi); Lf = self.px(rng.uniform(0.3, 1.2))
            cv2.line(fib, (int(ex_[i] - Lf * np.cos(a)), int(ey_[i] - Lf * np.sin(a))), (int(ex_[i] + Lf * np.cos(a)), int(ey_[i] + Lf * np.sin(a))), 255, 1, cv2.LINE_AA)
        fib = cv2.GaussianBlur(fib.astype(np.float32) / 255, (0, 0), 0.5)
        s = -rng.uniform(0.12, 0.30) * slime - rng.uniform(0.65, 0.85) * a_hole * (1 - 0.6 * fib)
        self._tex = (np.clip(slime, 0, 1), -rng.uniform(0.3, 0.7))
        return s.astype(np.float32)

    # ------------------------------------------------------------------ thin defects
    def _blade_scratch(self, L, rng, light):
        """Very fine hair-like groove along MD from the coating blade: faint relief edges, slight wander,
        strength fading in and out along its length, sometimes interrupted."""
        width = self.px({"small": rng.uniform(0.08, 0.18), "medium": rng.uniform(0.15, 0.3),
                         "large": rng.uniform(0.25, 0.6)}.get(getattr(self, "_tier", "medium"), 0.2))
        n = int(L) + 1
        wander = self.px(rng.uniform(0.05, 0.6)) * _lf(n, 1, rng, rng.uniform(80, 600))[:, 0]
        pad = int(width * 4 + abs(wander).max() + 4); wdt = 2 * pad + 1
        xx = np.arange(wdt, dtype=np.float32)[None, :] - pad - wander[:, None]
        groove = np.exp(-(xx / width) ** 2)
        along = np.clip(0.7 + 0.35 * _lf(n, 1, rng, rng.uniform(40, 300))[:, 0], 0.15, 1.2)
        if rng.random() < 0.3:                                        # interrupted in places
            along *= (_lf(n, 1, rng, rng.uniform(30, 120))[:, 0] > -0.8)
        along = cv2.GaussianBlur(along[:, None].astype(np.float32), (0, 0), 3)[:, 0]
        shade = relief_shading(-groove, light); shade /= (np.abs(shade).max() + 1e-6)
        thin = float(np.clip(self.px(0.3) / max(width, 1e-6), 1.0, 2.5))        # thinner scratches need more contrast to be visible
        s = (-rng.uniform(0.05, 0.12) * thin * groove + rng.uniform(0.02, 0.07) * shade) * along[:, None]
        return s.astype(np.float32)

    def _calender_cut(self, L, rng, light):
        """Wrinkle crushed and cut in the calender: sharp dark cut line, diagonal, with a narrow glazed /
        blackened band along it, tapering at the ends."""
        theta = np.pi / 2 + rng.choice([-1, 1]) * rng.uniform(np.deg2rad(20), np.deg2rad(70))
        pts = random_polyline(L, theta, rng.uniform(0.0002, 0.0015) * rng.choice([-1, 1]), rng, step=1.0)
        band_w = min(self.px(rng.uniform(0.2, 1.2)), 0.03 * L); core_w = self.px(rng.uniform(0.08, 0.2))
        mn, shape = canvas_for(pts, band_w * 3 + 4)
        d, pos = dist_to_polyline(pts, mn, shape)
        taper = smoothstep(np.minimum(pos, 1 - pos) / 0.12)
        h, w = shape
        vary = np.clip(0.75 + 0.3 * fbm(h, w, rng, self.px(rng.uniform(3, 10)), 2), 0.3, 1.2)
        core = np.exp(-(d / core_w) ** 2)
        band = np.exp(-(d / band_w) ** 2)
        s = (-rng.uniform(0.25, 0.55) * core - rng.uniform(0.04, 0.14) * band) * taper * vary
        return s.astype(np.float32)

    def _felt_hair(self, L, rng, light):
        """Filament from the machine felt embedded in the surface: thin, curly / wavy, longer and more
        visible than normal paper fibres, fading where it dips below the surface."""
        md = rng.random() < 0.5
        theta = np.pi / 2 + rng.normal(0, 0.4) if md else rng.uniform(0, np.pi)
        # wavy, not coiled: small curvature with a slow wave on top
        pts = random_polyline(L, theta, rng.uniform(0.002, 0.012) * rng.choice([-1, 1]), rng, step=1.0)
        seg = np.arange(len(pts), dtype=np.float32)
        wave = self.px(rng.uniform(0.15, 0.6)) * _lf(len(pts), 1, rng, self.px(rng.uniform(3, 10)))[:, 0]   # irregular meander
        dirv = np.gradient(pts, axis=0); nrm = np.stack([-dirv[:, 1], dirv[:, 0]], 1); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6
        pts = pts + nrm * wave[:, None]
        wid = self.px(rng.uniform(0.05, 0.15))
        mn, shape = canvas_for(pts, 6)
        d, pos = dist_to_polyline(pts, mn, shape)
        h, w = shape
        line = np.exp(-(d / max(wid, 0.45)) ** 2)
        dips = np.clip(0.6 + 0.5 * fbm(h, w, rng, self.px(rng.uniform(1.0, 4.0)), 2), 0.05, 1.1)   # in / out of surface
        taper = smoothstep(np.minimum(pos, 1 - pos) / 0.08)
        sign = -1 if rng.random() < 0.8 else 1                          # mostly darker, some lighter filaments
        thin = float(np.clip(self.px(0.15) / max(wid, 1e-6), 1.0, 2.0))       # thinner filaments need more contrast
        s = sign * rng.uniform(0.12, 0.30) * thin * line * dips * taper
        return cv2.GaussianBlur(s.astype(np.float32), (0, 0), 0.4)

    # ------------------------------------------------------------------ transmission defects
    # Light from behind the web shines through the opening: saturated bright core, soft glow of light
    # scattered in the sheet around it, ragged / fibrous edges, sometimes fibres across the opening.
    def _glow(self, opening, rng, reach_mm):
        g = cv2.GaussianBlur(opening.astype(np.float32), (0, 0), self.px(reach_mm))
        return g / (g.max() + 1e-6)

    def _finish_transmission(self, opening, rng, glow_mm, strands=0):
        edge = self.px(rng.uniform(0.04, 0.1))
        a = soft_alpha(opening, max(edge, 0.6))
        if strands:
            h, w = opening.shape; fib = np.zeros((h, w), np.uint8)
            ys, xs = np.nonzero(opening)
            for _ in range(strands):
                if len(xs) == 0: break
                i = int(rng.integers(len(xs))); ang = rng.uniform(0, np.pi); L = self.px(rng.uniform(0.5, 2.0))
                cv2.line(fib, (int(xs[i] - L * np.cos(ang)), int(ys[i] - L * np.sin(ang))), (int(xs[i] + L * np.cos(ang)), int(ys[i] + L * np.sin(ang))), 255, 1, cv2.LINE_AA)
            a = a * (1 - 0.7 * cv2.GaussianBlur(fib.astype(np.float32) / 255, (0, 0), 0.5))
        core = rng.uniform(0.6, 1.3) * a                                        # usually saturates at 255
        glow = rng.uniform(0.08, 0.25) * self._glow(opening, rng, glow_mm)
        return (core + glow * (1 - a)).astype(np.float32)

    def _transmission_pinhole(self, D, rng, light):
        pad = int(self.px(2.5)); n = int(D + 2 * pad) | 1; c = n // 2
        op = np.zeros((n * 4, n * 4), np.uint8)
        cv2.ellipse(op, (c * 4 + 2, c * 4 + 2), (max(1, int(D * 2 * rng.uniform(0.7, 1.0))), max(1, int(D * 2 * rng.uniform(0.7, 1.0)))),
                    rng.uniform(0, 180), 0, 360, 255, -1, cv2.LINE_AA)
        op = cv2.resize(op.astype(np.float32) / 255, (n, n), interpolation=cv2.INTER_AREA) > 0.35
        if not op.any(): op[c, c] = True
        return self._finish_transmission(op, rng, rng.uniform(0.3, 0.8))

    def _transmission_hole(self, D, rng, light):
        R = D / 2; pad = int(self.px(3) + 0.5 * R); n = int(2 * R + 2 * pad); c = n / 2
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        ang = np.arctan2(yy - c, xx - c); rho = np.hypot(yy - c, (xx - c) * rng.uniform(0.8, 1.25))
        k = int(rng.integers(3, 7)); amps = rng.uniform(0.3, 1.0, k); amps *= rng.uniform(0.08, 0.2) / amps.sum()
        lobes = sum(amps[j] * np.cos((j + 2) * ang + rng.uniform(0, 6.28)) for j in range(k))
        field = R * (1 + lobes) - rho + 0.05 * R * fbm(n, n, rng, max(3.0, R / 2), 3)
        op = cv2.GaussianBlur(field, (0, 0), 0.6) > 0
        return self._finish_transmission(op, rng, rng.uniform(0.4, 1.2), strands=int(rng.integers(0, 4)))

    def _transmission_tear(self, L, rng, light):
        theta = rng.uniform(0, np.pi)
        pts = random_polyline(L, theta, rng.uniform(0.003, 0.02) * rng.choice([-1, 1]), rng, step=1.0)
        w0 = self.px(rng.uniform(0.2, 1.5))
        mn, shape = canvas_for(pts, w0 + self.px(3))
        d, pos = dist_to_polyline(pts, mn, shape)
        h, w = shape
        width = w0 * (np.clip(np.sin(np.pi * np.clip(pos, 0, 1)), 0, 1) ** 0.7) * np.clip(1 + 0.35 * fbm(h, w, rng, self.px(1.0), 2), 0.3, 1.6)
        op = d < np.maximum(width / 2, 0.3)
        return self._finish_transmission(op, rng, rng.uniform(0.3, 0.9), strands=int(rng.integers(0, 3)))
