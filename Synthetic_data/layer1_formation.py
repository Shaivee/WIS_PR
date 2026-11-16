"""
Layer 1 — base sheet formation (fibre mass distribution).

Output: relative mass variation map (mean 0), dimensionless. A value of +0.03 means
3% more fibre than average at that pixel.

Default type "basler": measured on your Basler racer 2 L 8192 frame after removing
the camera's column pattern, sensor noise and lighting profile. Weak (strength ~0.01),
fine-scale formation with little large-scale cloudiness.
Optional types "calm" / "cloudy": measured on the 2365-px sample images (stronger, cloudier).

  * spectral synthesis with the measured 2-D power spectrum of the chosen type
  * brightness distribution matched to the real per-patch distribution
  * strength drawn from the measured range (widened by strength_spread for variety)
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

MM_PER_PX_REF = 0.112          # pixel size of the images the model was measured on


class FormationModel:
    def __init__(self, path):
        z = np.load(path)
        self.freq, self.quant = z["freq"], z["quantiles"]
        self.kinds = [str(k) for k in z["kinds"]]
        all_kinds = self.kinds + [str(k) for k in z.get("optional_kinds", [])]
        self.logpsd = {k: z[f"logpsd_{k}"] for k in all_kinds}
        self.strengths = {k: z[f"strengths_{k}"] for k in all_kinds}
        self.interp = {k: RegularGridInterpolator((self.freq, self.freq), self.logpsd[k],
                                                  bounds_error=False, fill_value=None) for k in all_kinds}

    def _psd(self, h, w, kind, other, mix, scale):
        fy = np.fft.fftfreq(h)[:, None] * scale
        fx = np.fft.fftfreq(w)[None, :] * scale
        fmax, step = self.freq.max(), self.freq[1] - self.freq[0]
        pts = np.stack([np.clip(np.broadcast_to(fy, (h, w)), -fmax, fmax).ravel(),
                        np.clip(np.broadcast_to(fx, (h, w)), -fmax, fmax).ravel()], 1)
        lp = (1 - mix) * self.interp[kind](pts) + mix * self.interp[other](pts)
        p = np.exp(lp).reshape(h, w)
        f = np.sqrt(fx ** 2 + fy ** 2)
        return p * np.clip(f / (2 * step), 0, 1) ** 2        # scales above ~57 mm belong to later layers

    def _histmatch(self, x):
        order = np.argsort(x.ravel())
        out = np.empty(x.size, np.float32)
        out[order] = np.interp(np.linspace(0, 1, x.size), np.linspace(0, 1, len(self.quant)), self.quant)
        return out.reshape(x.shape)

    def sample_params(self, rng, kind="basler", strength_spread=0.2):
        """strength_spread: extra +/- relative variation around the measured range
        (the Basler model comes from a single frame, so its measured range is narrow)."""
        s = self.strengths[kind]
        strength = float(np.interp(rng.random(), np.linspace(0, 1, len(s)), s))
        strength *= float(rng.uniform(1 - strength_spread, 1 + strength_spread))
        return dict(kind=kind, other=kind, mix=0.0, strength=strength)

    def generate(self, h, w, rng, mm_per_px=MM_PER_PX_REF, params=None, iters=3):
        """Returns (formation map, params). mm_per_px sets the physical scale of the output pixels."""
        p = params or self.sample_params(rng)
        scale = MM_PER_PX_REF / mm_per_px
        amp = np.sqrt(self._psd(h, w, p["kind"], p["other"], p["mix"], scale))
        x = np.fft.ifft2(np.fft.fft2(rng.standard_normal((h, w))) * amp).real
        x = (x - x.mean()) / (x.std() + 1e-12)
        for _ in range(iters):                                # marginal shape, then spectrum (last)
            x = self._histmatch(x)
            x = np.fft.ifft2(amp * np.exp(1j * np.angle(np.fft.fft2(x)))).real
            x = (x - x.mean()) / (x.std() + 1e-12)
        return (x * p["strength"]).astype(np.float32), p
