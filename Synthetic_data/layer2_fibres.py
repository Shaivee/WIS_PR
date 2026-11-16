"""
Layer 2 — fibres, specks and faint spots on/in the paper surface.

Works on the paper reflectance map (1 = average paper) BEFORE the camera layer:
    paper <- paper - depth * alpha
Subtracting (rather than multiplying) keeps the paper texture inside a feature exactly the
same as in the surrounding paper; alpha has soft, blurred edges (edge_softness).
Every feature then receives the same shading, PRNU/DSNU, noise and blur as the paper.

Three feature types, measured on your Basler frame:
  speck : small compact dark dirt particle (1-4 px across, strong: 15-60% darker)
  fibre : thin curved fibre, sub-pixel to ~1.5 px wide, 10-60 px long, faint (5-15% darker),
          contrast fading along its length (fibre dipping in and out of the surface)
  faint : soft low-contrast blotch a few px across (5-12% darker)
Features are drawn in physical (square-pixel) space, then stretched along MD (y) by
md_stretch (Basler: ~1.8), as the real line-scan images are.
"""
import numpy as np, cv2

DEFAULTS = dict(
    md_stretch=(1.6, 2.0),               # Basler: specks ~1.8x taller than wide
    density_scale=(0.6, 1.5),            # per-frame multiplier on all densities (paper cleanliness varies)
    speck_per_mpx=36.0, speck_radius=(0.45, 1.3), speck_depth=(0.22, 0.65),
    fibre_per_mpx=54.0, fibre_length=(4, 20), fibre_width=(0.35, 0.8), fibre_depth=(0.055, 0.14),
    fibre_md_align=0.75,                 # 0 = random orientation, 1 = all along MD
    long_fibre_share=0.12, long_fibre_length=(30, 90),   # occasional long, curly fibres
    speck_halo_share=0.25, speck_tail_share=0.15,        # speck variety
    faint_per_mpx=28.0, faint_sigma=(0.5, 1.2), faint_depth=(0.05, 0.10),
    speck_edge_softness=(0.5, 0.85),     # edge blur in px (soft edges)
    fibre_edge_softness=(0.5, 0.9),
)
# Calibrated so that the detector in layer2_detect.py finds the same statistics on synthetic
# frames as on the real Basler frame (per Mpx: specks ~21, fibres ~40, faint ~90 incl. texture).
SS = 4  # supersampling for anti-aliased sub-pixel drawing


def _loguni(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


class FibreSpeckLayer:
    def __init__(self, **overrides):
        self.cfg = dict(DEFAULTS, **overrides)

    def sample_params(self, rng):
        c = self.cfg
        return dict(md_stretch=float(rng.uniform(*c["md_stretch"])),
                    density=float(rng.uniform(*c["density_scale"])))

    # ---- drawing helpers: returns alpha patch (0..1) already stretched along MD
    def _finish(self, canvas, stretch, blur_px):
        a = canvas.astype(np.float32) / 255.0
        if blur_px > 0:
            a = cv2.GaussianBlur(a, (0, 0), blur_px * SS)
        h, w = a.shape
        return cv2.resize(a, (max(1, w // SS), max(1, int(round(h * stretch / SS)))), interpolation=cv2.INTER_AREA)

    def _speck(self, rng, stretch):
        r = _loguni(rng, *self.cfg["speck_radius"])
        n = int(np.ceil(r * 3 + 3)) * SS
        c = np.zeros((2 * n, 2 * n), np.uint8)
        # irregular particle: a few overlapping small ellipses
        for _ in range(int(rng.integers(1, 4))):
            ox, oy = rng.normal(0, r * 0.35, 2) * SS
            ax_ = r * rng.uniform(0.6, 1.0) * SS; ay_ = r * rng.uniform(0.6, 1.0) * SS
            cv2.ellipse(c, (int(n + ox), int(n + oy)), (max(1, int(ax_)), max(1, int(ay_))), float(rng.uniform(0, 180)), 0, 360, 255, -1, cv2.LINE_AA)
        if rng.random() < self.cfg["speck_tail_share"]:           # short fibre tail stuck to the particle
            ang = rng.uniform(0, 2 * np.pi); L = r * rng.uniform(2, 5) * SS
            cv2.line(c, (n, n), (int(n + L * np.cos(ang)), int(n + L * np.sin(ang))), 90, max(1, int(0.8 * SS)), cv2.LINE_AA)
        a = self._finish(c, stretch, rng.uniform(*self.cfg["speck_edge_softness"]))
        a = a / max(a.max(), 0.7)                                  # small specks may stay slightly lighter
        if rng.random() < self.cfg["speck_halo_share"]:           # soft stained halo around the particle
            a = np.maximum(a, 0.18 * cv2.GaussianBlur(a, (0, 0), 1.5 * r + 0.5) / (a.max() + 1e-6) * 4).clip(0, 1)
        return a, _loguni(rng, *self.cfg["speck_depth"])

    def _fibre(self, rng, stretch):
        c = self.cfg
        long_ = rng.random() < c["long_fibre_share"]
        L = _loguni(rng, *(c["long_fibre_length"] if long_ else c["fibre_length"]))
        # orientation: mixture of MD-aligned and random
        th = rng.normal(np.pi / 2, 0.35) if rng.random() < c["fibre_md_align"] else rng.uniform(0, np.pi)
        curv = (rng.uniform(0.03, 0.10) if long_ else rng.uniform(0.01, 0.08)) * (1 if rng.random() < 0.5 else -1)
        n = int(L / 1.0) + 2
        pts, x, y, a = [], 0.0, 0.0, th
        for i in range(n):
            pts.append((x, y)); a += curv + rng.normal(0, 0.04)
            if long_ and rng.random() < 0.03: curv = -curv                 # long fibres change bending direction
            x += np.cos(a); y += np.sin(a)
        pts = np.array(pts)
        pad = 4
        mn = pts.min(0) - pad; size = (pts.max(0) - mn + pad).astype(int) + 1
        canvas = np.zeros((size[1] * SS, size[0] * SS), np.uint8)
        width = rng.uniform(*c["fibre_width"])
        # contrast varies along the fibre (dips in / out of the surface)
        prof = np.clip(0.55 + 0.45 * np.sin(np.linspace(0, rng.uniform(1, 4) * np.pi, n) + rng.uniform(0, 6.3)), 0.15, 1)
        prof *= np.minimum(1, np.minimum(np.arange(n), n - 1 - np.arange(n)) / 3 + 0.3)   # tapered ends
        P = ((pts - mn) * SS).astype(np.int32)
        for i in range(n - 1):
            cv2.line(canvas, tuple(P[i]), tuple(P[i + 1]), int(255 * prof[i]), max(1, int(round(width * SS))), cv2.LINE_AA)
        a = self._finish(canvas, stretch, rng.uniform(*c["fibre_edge_softness"]))
        a = a / (a.max() + 1e-6) * prof.max()
        return a, _loguni(rng, *c["fibre_depth"])

    def _faint(self, rng, stretch):
        s = rng.uniform(*self.cfg["faint_sigma"])
        n = int(np.ceil(s * 4 + 2)) * SS
        c = np.zeros((2 * n, 2 * n), np.uint8)
        cv2.ellipse(c, (n, n), (max(1, int(s * SS * rng.uniform(0.7, 1.3))), max(1, int(s * SS * rng.uniform(0.7, 1.3)))),
                    float(rng.uniform(0, 180)), 0, 360, 255, -1, cv2.LINE_AA)
        a = self._finish(c, stretch, s * 0.6)
        a /= a.max() + 1e-6
        return a, _loguni(rng, *self.cfg["faint_depth"])

    def apply(self, paper, rng, params=None, exclude=None):
        """paper: reflectance map (modified copy returned). Returns (paper, params, features)."""
        h, w = paper.shape
        p = params or self.sample_params(rng)
        out = paper.copy()
        mpx = h * w / 1e6
        feats = []
        for kind, dens, fn in (("faint", self.cfg["faint_per_mpx"], self._faint),
                               ("fibre", self.cfg["fibre_per_mpx"], self._fibre),
                               ("speck", self.cfg["speck_per_mpx"], self._speck)):
            for _ in range(rng.poisson(dens * mpx * p["density"])):
                a, depth = fn(rng, p["md_stretch"])
                ph, pw = a.shape
                y0, x0 = int(rng.integers(-ph // 2, h - ph // 2)), int(rng.integers(-pw // 2, w - pw // 2))
                ys, xs = max(0, y0), max(0, x0); ye, xe = min(h, y0 + ph), min(w, x0 + pw)
                if ye <= ys or xe <= xs: continue
                sub = a[ys - y0:ye - y0, xs - x0:xe - x0]
                # subtractive darkening: the paper texture continues unchanged through the feature
                out[ys:ye, xs:xe] -= depth * sub
                feats.append(dict(kind=kind, x=xs, y=ys, w=xe - xs, h=ye - ys, depth=depth))
        return out, p, feats
