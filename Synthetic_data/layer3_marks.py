"""
Layer 3 - harmless look-alikes: normal paper structure that is NOT a defect and is NEVER labelled.
They teach a detector what not to flag. Applied to the paper reflectance map after layers 1-2 and
before the defects (so defects sit on top), and kept weaker / more regular / more diffuse than defects.

  wire mark     : forming-wire weave pressed into the sheet - very fine regular diamond / twill texture
  couch mark    : drilled-hole pattern of a suction roll copied into the sheet - staggered grid of soft dots
  seam mark     : light band across the web from a seamed wire / felt (once every several metres)
  cloudy patch  : area of poorer formation - the normal cloudy texture is stronger there, no edges

Wire, couch and seam marks come from the machine's fabrics and rolls, so they are planned per
stretch of web and continue from frame to frame; cloudy patches are drawn per frame.
"""
import numpy as np, cv2

DEFAULTS = dict(
    wire_prob=0.4, wire_amp=(0.006, 0.015), wire_period_mm=(0.3, 0.8),
    couch_prob=0.25, couch_amp=(0.01, 0.03), couch_pitch_mm=(3.0, 6.0), couch_dot_mm=(1.5, 3.5),
    seam_prob=0.15, seam_amp=(0.015, 0.04), seam_width_mm=(2.0, 10.0),
    cloudy_prob=0.3, cloudy_gain=(0.5, 1.5), cloudy_size_mm=(30.0, 150.0),
    mm_per_px=0.112,
)


def _smooth_noise(h, w, rng, scale):
    s = rng.standard_normal((max(2, int(h / scale) + 2), max(2, int(w / scale) + 2))).astype(np.float32)
    b = cv2.resize(s, (w, h), interpolation=cv2.INTER_CUBIC)
    return b / (b.std() + 1e-6)


class LookalikeLayer:
    def __init__(self, **overrides):
        self.cfg = dict(DEFAULTS, **overrides)

    def px(self, mm):
        return mm / self.cfg["mm_per_px"]

    def plan(self, n_frames, H, W, rng, md_stretch=1.8):
        c = self.cfg; p = {}
        if rng.random() < c["wire_prob"]:
            per = self.px(rng.uniform(*c["wire_period_mm"]))
            p["wire"] = dict(amp=float(rng.uniform(*c["wire_amp"])), per=per, ang=float(rng.uniform(0.35, 0.75)),
                             twill=bool(rng.random() < 0.5), ph=rng.uniform(0, 2 * np.pi, 2).tolist())
        if rng.random() < c["couch_prob"]:
            pitch = self.px(rng.uniform(*c["couch_pitch_mm"]))
            band = None
            if rng.random() < 0.5:                                  # only a band across part of the web
                x0 = rng.uniform(0, W * 0.7); band = (x0, x0 + rng.uniform(0.15, 0.5) * W)
            p["couch"] = dict(amp=float(rng.uniform(*c["couch_amp"])) * (1 if rng.random() < 0.6 else -1), pitch=pitch,
                              dot=self.px(rng.uniform(*c["couch_dot_mm"])), off=rng.uniform(0, pitch, 2).tolist(), band=band)
        if rng.random() < c["seam_prob"]:
            p["seam"] = dict(amp=float(rng.uniform(*c["seam_amp"])), y=float(rng.uniform(0, n_frames * H)),
                             w=self.px(rng.uniform(*c["seam_width_mm"])) * md_stretch)
        return p

    def apply(self, paper, plan, fi, rng, md_stretch=1.8):
        """Returns (paper, names of the look-alikes present in this frame)."""
        H, W = paper.shape; out = paper.copy(); names = []
        yy = (np.arange(H, dtype=np.float32) + fi * H)[:, None]    # strip coordinates: patterns continue across frames
        xx = np.arange(W, dtype=np.float32)[None, :]
        if "wire" in plan:
            q = plan["wire"]; k = 2 * np.pi / q["per"]; ys = yy / md_stretch
            a = np.sin(k * (xx * np.cos(q["ang"]) + ys * np.sin(q["ang"])) + q["ph"][0])
            b = np.sin(k * (xx * np.cos(q["ang"]) - ys * np.sin(q["ang"])) + q["ph"][1])
            pat = a * b if q["twill"] else 0.5 * (a + b)
            pat *= np.clip(0.7 + 0.3 * _smooth_noise(H, W, rng, 200), 0.3, 1.2)       # weave not perfectly even
            out += q["amp"] * pat.astype(np.float32); names.append("wire mark")
        if "couch" in plan:
            q = plan["couch"]; ys = yy / md_stretch
            gx = (xx - q["off"][0]) / q["pitch"]; gy = (ys - q["off"][1]) / (q["pitch"] * 0.866)
            row = np.floor(gy)
            dx = (gx + 0.5 * (row % 2)) % 1.0 - 0.5; dy = gy - row - 0.5               # staggered holes
            r = np.sqrt((dx * q["pitch"]) ** 2 + (dy * q["pitch"] * 0.866) ** 2)
            dots = np.exp(-(r / (q["dot"] / 2)) ** 2)
            if q["band"] is not None:
                x0, x1 = q["band"]; ramp = self.px(15)
                dots = dots * np.clip(np.minimum(xx - x0, x1 - xx) / ramp, 0, 1)
            out += q["amp"] * dots.astype(np.float32); names.append("couch mark")
        if "seam" in plan:
            q = plan["seam"]; d = (yy - q["y"]) / (q["w"] / 2)
            if np.abs(d).min() < 3:
                band = np.exp(-d ** 4) * np.clip(1 + 0.15 * _smooth_noise(1, W, rng, 400), 0.6, 1.3)
                out += q["amp"] * band.astype(np.float32); names.append("seam mark")
        if rng.random() < self.cfg["cloudy_prob"]:
            size = self.px(rng.uniform(*self.cfg["cloudy_size_mm"]))
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            m = np.exp(-(((xx - cx) / size) ** 2 + ((yy - fi * H - cy) / (size * md_stretch * rng.uniform(0.5, 1.2))) ** 2))
            tex = out - cv2.GaussianBlur(out, (0, 0), 25)                               # the paper's own cloudy texture
            out += rng.uniform(*self.cfg["cloudy_gain"]) * m.astype(np.float32) * tex; names.append("cloudy patch")
        return out.astype(np.float32), names
