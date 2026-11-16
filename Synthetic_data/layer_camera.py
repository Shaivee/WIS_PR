"""
Camera layers — illumination, optics and sensor, applied on top of the paper layers.

Signal chain for one line-scan frame (rows = machine direction, columns = sensor pixels):

    I = white_level * paper(x, y) * shading(x) * irregular_light(x, y)      # light reaching the sensor
    I = I * (1 + PRNU(x))                                                   # per-pixel sensitivity
    I = I + DSNU(x)                                                         # per-pixel dark offset
    I = I + temporal_noise(I)                                               # shot + read noise
    image = clip(round(I), 0, 255)

On a line-scan sensor every row is read by the same pixels, so PRNU and DSNU are
column patterns (vertical stripes), and shading/vignetting varies only across the web (x).

Calibrated on the Basler racer 2 L 8192 frame:
  shading    : off-centre, 51% at left edge, peak ~75% across, 75% at right edge
  PRNU+DSNU  : ~0.6 grey levels of fine column pattern, nearly constant with brightness (DSNU-dominated)
  irregular  : ~1.6% lamp structure across CD, ~0.2% flicker along MD
  noise      : shot + read noise, then a small post-capture blur (0.45 x 0.55 px) that reproduces
               the pixel-to-pixel correlation of the real frame's grain (calibrated: fine texture
               2.8 GL, column lines 0.65 GL, correlation vertical 0.41 / horizontal 0.21)
Each effect is randomised per frame and can be switched off per frame (see DEFAULTS).
"""
import numpy as np, cv2

DEFAULTS = {
    # probability that the effect is present in a frame, and its strength range
    "shading_prob": 0.85,            # otherwise: flat-field corrected (nearly flat, <=2% residual)
    "shading_strength": (0.5, 1.2),  # 1.0 = as strong as the measured Basler shading
    "lamp_zones": True,              # broad lamp hot/cold zones across the web
    "led_ripple": False,             # periodic vertical bands (100-400 px) from LED line lights - removed
    "prnu_prob": 0.9,
    "prnu_std": (0.001, 0.003),      # relative per-column gain spread (calibrated with post-blur: ~0.2%)
    "dsnu_prob": 0.9,
    "dsnu_std": (0.5, 1.2),          # grey levels per-column offset spread (calibrated with post-blur: ~0.85)
    "dsnu_oddeven_prob": 0.15,       # extra odd/even column offset
    "dsnu_segments_prob": 0.15,      # extra offsets between sensor readout segments (taps)
    "irregular_light_prob": 0.3,     # lamp hot/cold zones, LED ripple, MD flicker
    "white_level": (170, 215),       # brightness of the brightest part of the paper
    "read_noise": (0.9, 1.4),        # grey levels, BEFORE post-capture blur
    "shot_noise_k": (0.066, 0.100),  # variance per grey level, BEFORE post-capture blur
    # post-capture blur: in the real frame even the sensor noise is correlated between neighbouring
    # pixels (vertical 0.41, horizontal 0.21), i.e. the image is softened after read-out
    # (camera / acquisition processing). Applied last, to everything including noise and PRNU/DSNU.
    "post_blur_x": (0.40, 0.50),     # px, across the web
    "post_blur_y": (0.47, 0.57),     # px, along the web
}


def _smooth1d(n, rng, scale):
    k = max(2, n // max(2, int(scale)) + 2)
    s = cv2.resize(rng.standard_normal((1, k)).astype(np.float32), (n, 1), interpolation=cv2.INTER_CUBIC)[0]
    return s / (s.std() + 1e-9)


class CameraModel:
    def __init__(self, path, **overrides):
        z = np.load(path)
        self.fp_amp = z["fixed_pattern_amp"]
        self.fp_len = int(z["fixed_pattern_len"])
        self.shading_basler = z["shading_basler"]
        xs = np.linspace(-1, 1, len(self.shading_basler))
        self._shade_poly = np.polyfit(xs, self.shading_basler, 6)     # smooth, can be evaluated slightly shifted
        self.cfg = dict(DEFAULTS, **overrides)

    # ------------------------------------------------------------------ components
    @staticmethod
    def _post_blur(I, p):
        sx, sy = p.get("post_blur_x", 0.0), p.get("post_blur_y", 0.0)
        if sx <= 0 and sy <= 0:
            return I
        return cv2.GaussianBlur(I.astype(np.float32), (5, 5), sigmaX=max(sx, 1e-3), sigmaY=max(sy, 1e-3))
    def fixed_pattern(self, w, rng):
        """Column pattern with the same spatial character as the measured Basler one, unit std."""
        amp = np.interp(np.linspace(0, len(self.fp_amp) - 1, w // 2 + 1), np.arange(len(self.fp_amp)), self.fp_amp)
        x = np.fft.irfft(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, len(amp))), w)
        return (x / (x.std() + 1e-12)).astype(np.float32)

    def shading(self, w, rng, strength):
        """Smooth fall-off across the web: measured Basler shape or a lens model with an off-centre lamp."""
        x = np.linspace(-1, 1, w, dtype=np.float32)
        if rng.random() < 0.5:
            sc = rng.uniform(0.88, 1.0)                                   # small shift / stretch of the peak,
            xx = x * sc + rng.uniform(-(1 - sc), 1 - sc)                  # staying inside the measured range
            if rng.random() < 0.5:
                xx = -xx                                                  # mirrored
            base = np.polyval(self._shade_poly, xx).astype(np.float32)
        else:
            fov = rng.uniform(0.6, 1.0)                                 # lens: cos^4 of the field angle
            lens = np.cos(np.arctan(x * fov)) ** 4
            c, wd = rng.uniform(-0.6, 0.6), rng.uniform(0.9, 2.0)       # lamp: broad, off-centre
            lamp = np.exp(-((x - c) / wd) ** 2)
            base = lens * lamp
            base = base / base.max()
            drop = rng.uniform(0.25, 0.55)                               # edge fall-off similar to the Basler one
            base = 1 - (1 - base) * drop / max(1e-6, (1 - base).max())
        base = base / base.max()
        return np.clip(1 - strength * (1 - base), 0.3, 1.0).astype(np.float32)

    def irregular_light(self, h, w, rng, parts=False):
        """Optional: lamp hot/cold zones and LED ripple across CD, flicker along MD.
        parts=True returns (md, cd) separately."""
        x = np.arange(w, dtype=np.float32)
        cd = np.ones(w, np.float32)
        if self.cfg.get("lamp_zones", False):                          # broad hot / cold zones (read as vignetting)
            cd += rng.uniform(0.005, 0.02) * _smooth1d(w, rng, rng.uniform(300, 1200))
            for _ in range(int(rng.integers(0, 4))):
                cx, sw = rng.uniform(0, w), rng.uniform(100, 800)
                cd += rng.uniform(-0.05, 0.05) * np.exp(-((x - cx) / sw) ** 2)
        if self.cfg.get("led_ripple", False) and rng.random() < 0.5:   # LED ripple: regular vertical bands
            cd += rng.uniform(0.003, 0.015) * np.sin(2 * np.pi * x / rng.uniform(100, 400) + rng.uniform(0, 6.3))
        md = 1.0 + rng.uniform(0.001, 0.004) * _smooth1d(h, rng, rng.uniform(5, 60))   # Basler ~0.0018
        if parts:
            return md, cd
        return md[:, None] * cd[None, :]

    # ------------------------------------------------------------------ sampling
    def sample_params(self, rng):
        c = self.cfg
        p = dict(
            shading=rng.random() < c["shading_prob"],
            prnu=rng.random() < c["prnu_prob"],
            dsnu=rng.random() < c["dsnu_prob"],
            irregular_light=rng.random() < c["irregular_light_prob"],
            white_level=float(rng.uniform(*c["white_level"])),
            read_noise=float(rng.uniform(*c["read_noise"])),
            shot_k=float(rng.uniform(*c["shot_noise_k"])),
            post_blur_x=float(rng.uniform(*c["post_blur_x"])),
            post_blur_y=float(rng.uniform(*c["post_blur_y"])),
        )
        p["shading_strength"] = float(rng.uniform(*c["shading_strength"])) if p["shading"] else float(rng.uniform(0, 0.04))
        p["prnu_std"] = float(rng.uniform(*c["prnu_std"])) if p["prnu"] else 0.0
        p["dsnu_std"] = float(rng.uniform(*c["dsnu_std"])) if p["dsnu"] else 0.0
        p["dsnu_oddeven"] = float(rng.uniform(0.2, 0.8)) if p["dsnu"] and rng.random() < c["dsnu_oddeven_prob"] else 0.0
        p["dsnu_segments"] = int(rng.choice([2, 4, 8])) if p["dsnu"] and rng.random() < c["dsnu_segments_prob"] else 0
        return p

    # ------------------------------------------------------------------ apply
    def apply(self, paper, rng, params=None, return_parts=False):
        """paper: relative reflectance map (1 = average paper), shape (H, W). Returns uint8 image."""
        h, w = paper.shape
        p = params or self.sample_params(rng)
        parts = {}
        shade = self.shading(w, rng, p["shading_strength"])
        parts["shading"] = shade
        light = p["white_level"] * shade[None, :]
        if p["irregular_light"]:
            il = self.irregular_light(h, w, rng)
            parts["irregular_light"] = il
            light = light * il
        I = paper * light
        if p["prnu_std"] > 0:
            prnu = p["prnu_std"] * (0.6 * self.fixed_pattern(w, rng) + 0.8 * rng.standard_normal(w).astype(np.float32))
            prnu *= p["prnu_std"] / (prnu.std() + 1e-12)
            parts["prnu"] = prnu
            I = I * (1 + prnu)[None, :]
        if p["dsnu_std"] > 0:
            dsnu = p["dsnu_std"] * self.fixed_pattern(w, rng)
            if p["dsnu_oddeven"]:
                dsnu[1::2] += p["dsnu_oddeven"]
            if p["dsnu_segments"]:
                seg = np.repeat(rng.uniform(-1.0, 1.0, p["dsnu_segments"]), int(np.ceil(w / p["dsnu_segments"])))[:w]
                dsnu += seg.astype(np.float32)
            parts["dsnu"] = dsnu
            I = I + dsnu[None, :]
        sigma = np.sqrt(p["read_noise"] ** 2 + p["shot_k"] * np.clip(I, 0, None))
        I = I + sigma * rng.standard_normal(I.shape).astype(np.float32)
        I = self._post_blur(I, p)
        img = np.clip(np.round(I), 0, 255).astype(np.uint8)
        return (img, p, parts) if return_parts else (img, p)


# ---------------------------------------------------------------------- variant sets
    def sample_variant_params(self, rng):
        """Like sample_params, but every effect is ON (strength drawn from its range)."""
        c = self.cfg
        return dict(
            white_level=float(rng.uniform(*c["white_level"])),
            read_noise=float(rng.uniform(*c["read_noise"])),
            shot_k=float(rng.uniform(*c["shot_noise_k"])),
            post_blur_x=float(rng.uniform(*c["post_blur_x"])),
            post_blur_y=float(rng.uniform(*c["post_blur_y"])),
            shading_strength=float(rng.uniform(*c["shading_strength"])),
            prnu_std=float(rng.uniform(*c["prnu_std"])),
            dsnu_std=float(rng.uniform(*c["dsnu_std"])),
            dsnu_oddeven=float(rng.uniform(0.2, 0.8)) if rng.random() < c["dsnu_oddeven_prob"] else 0.0,
            dsnu_segments=int(rng.choice([2, 4, 8])) if rng.random() < c["dsnu_segments_prob"] else 0,
            irregular_light=bool(rng.random() < c["irregular_light_prob"]),   # combined image only
        )

    def apply_variants(self, paper, rng, params=None, include_reference=False, mult=None, pattern_seed=None):
        """Render the same paper four ways with shared random draws:
        'shading', 'prnu', 'dsnu' (one effect each) and 'combined' (all three,
        plus illumination irregularities if drawn). Optionally 'reference' (none).
        mult: strength multipliers, e.g. {"shading": 1, "prnu": 5, "dsnu": 5}
              (1 = realistic; >1 exaggerates the effect to make it clearly visible).
        Returns (dict name -> uint8 image, params, parts)."""
        h, w = paper.shape
        p = dict(params or self.sample_variant_params(rng))
        m = {"shading": 1.0, "prnu": 1.0, "dsnu": 1.0, **(mult or {})}
        p["shading_strength"] *= m["shading"]
        p["prnu_std"] *= m["prnu"]
        p["dsnu_std"] *= m["dsnu"]
        p["dsnu_oddeven"] *= m["dsnu"]
        p["dsnu_segment_scale"] = m["dsnu"]
        p.update({f"mult_{k}": v for k, v in m.items()})
        prng = np.random.default_rng(pattern_seed) if pattern_seed is not None else rng   # camera patterns fixed per sequence
        shade = self.shading(w, prng, p["shading_strength"])
        prnu = 0.6 * self.fixed_pattern(w, prng) + 0.8 * prng.standard_normal(w).astype(np.float32)
        prnu *= p["prnu_std"] / (prnu.std() + 1e-12)
        dsnu = p["dsnu_std"] * self.fixed_pattern(w, prng)
        if p["dsnu_oddeven"]:
            dsnu[1::2] += p["dsnu_oddeven"]
        if p["dsnu_segments"]:
            seg = np.repeat(prng.uniform(-1.0, 1.0, p["dsnu_segments"]), int(np.ceil(w / p["dsnu_segments"])))[:w]
            dsnu = dsnu + p["dsnu_segment_scale"] * seg.astype(np.float32)
        # lamp zones across the web are fixed for the installation (pattern generator); flicker changes per frame
        if p["irregular_light"]:
            _, cd = self.irregular_light(h, w, prng, parts=True)          # lamp zones: fixed for the installation
            md, _ = self.irregular_light(h, w, rng, parts=True)            # flicker: changes from frame to frame
            il = md[:, None] * cd[None, :]
        else:
            il = 1.0
        z = rng.standard_normal((h, w)).astype(np.float32)                # one noise draw shared by all images
        S = p["white_level"] * paper
        signals = {
            "shading": S * shade[None, :],
            "prnu": S * (1 + prnu)[None, :],
            "dsnu": S + dsnu[None, :],
            "combined": S * shade[None, :] * il * (1 + prnu)[None, :] + dsnu[None, :],
        }
        if include_reference:
            signals = {"reference": S, **signals}
        out = {}
        for k, I in signals.items():
            sigma = np.sqrt(p["read_noise"] ** 2 + p["shot_k"] * np.clip(I, 0, None))
            out[k] = np.clip(np.round(self._post_blur(I + sigma * z, p)), 0, 255).astype(np.uint8)
        parts = dict(shading=shade, prnu=prnu, dsnu=dsnu, irregular_light=il)
        return out, p, parts
