"""
Layer 4 — defects, transplanted from your labelled sample images.

Each library entry is the defect's own relative signal (brightness change vs. the paper
around it), with the sample paper's texture and noise removed and edges fading to 0.
It is ADDED to the synthetic paper reflectance before the camera layer:
    paper <- paper + scale * signal
so the synthetic paper texture continues unchanged inside the defect, edges are soft,
and the defect gets the same shading, PRNU/DSNU, noise as the rest of the frame.

Per placement: random left/right flip, size scale, strength scale, MD stretch (same as the
frame), and for MD line defects (blade line, coating mark, crease, impression line) a
random window along the line so it runs through the whole frame height.
"""
import numpy as np, cv2

from layer4b_new_defects import NewDefectGenerator, NEW_CLASSES, NEW_SIZES, TRANSLUCENT_CLASSES, THIN_CLASSES, TRANSMISSION_CLASSES

LIB_CLASSES = ["pinhole", "Layers missing", "blotches", "punching defect", "stamping", "Impression line",
               "Dirt", "Deposit", "starch spot", "wax spot", "Crease", "bladeline", "Coating mark", "Grease mark"]
EDGE_CLASSES = ["edge crack", "edge cut"]    # only produced in web-edge stretches
CLASSES = (LIB_CLASSES + [c for c in NEW_CLASSES if c not in TRANSLUCENT_CLASSES + THIN_CLASSES + TRANSMISSION_CLASSES]
           + EDGE_CLASSES + TRANSLUCENT_CLASSES + THIN_CLASSES + TRANSMISSION_CLASSES)
# ids: 0-13 samples, 14-20 procedural, 21-22 edge, 23-26 translucent, 27-29 thin,
#      30-32 transmission (light shining through: pinhole, small hole, small tear)
MD_LINES = {"bladeline", "Coating mark", "Crease", "Impression line", "blade scratch"}
# surface-only / relief defects: strong in reflected light, weak in transmitted light
RELIEF = {"wrinkle", "blister", "scratch", "Crease", "Impression line", "bladeline", "Coating mark", "blade scratch", "coating skip"}
# straight MD lines: shape from the sample, strength (peak relative change) from the range of
# the clearly visible real lines (some labelled lines average out to <0.2% and are not learnable)
LINE_PEAK = {"bladeline": (0.008, 0.02), "Coating mark": (0.035, 0.07), "Impression line": (0.02, 0.035)}
NO_VFLIP = {"blotches"}                      # comet tail direction follows the web direction
# compact, single-tone defects whose outline can be rebuilt when enlarged (mixed / multi-part
# defects such as blotches with tails, stamping, punching, grease dots keep their own edges)
# physical size limits per defect type (longest side, mm): a pinhole is tiny by definition
# (real ones: ~1.5 mm), missing-layer spots are small (~3 mm); large bright areas are blotches
CLASS_MAX_MM = {"pinhole": 1.6, "Layers missing": 6.0}
# ("Layers missing" is excluded: it is a naturally soft glow with a ~1.4 mm edge)
# defects where light passes THROUGH the paper. Front-lit: the camera sees the dark space behind
# the web through them, so the transmitted (bright) part of the signal becomes dark.
HOLE_LIKE = {"pinhole", "blotches", "torn hole", "Layers missing"}
EDGE_FIX_CLASSES = {"pinhole", "Dirt", "Deposit", "starch spot", "wax spot"}

DEFAULTS = dict(
    defects_per_frame=(8, 14),
    defect_free=0.10,
    size_scale=(0.8, 1.25),                  # used only when size_tiers is None
    # size tiers: longest side of the visible defect, in mm (converted with mm_per_px)
    size_tiers={"small": (1.0, 5.0), "medium": (5.0, 20.0), "large": (20.0, 80.0)},
    mix_sizes=True,                          # frames with >= 3 defects get every tier at least once
    line_width_tiers={"small": (0.4, 0.8), "medium": (0.8, 1.5), "large": (1.5, 2.5)},   # x real line width
    line_max_width_mm={"bladeline": 6.0, "Crease": 15.0, "Impression line": 20.0, "Coating mark": 25.0},
    max_height_share=0.9,
    max_lines_per_frame=2,                   # MD line defects (blade line, coating mark, crease, impression line)                    # non-line defects always fit inside the frame (no cut edges)
    scale_limits=(0.15, 5.0),                # never shrink/enlarge a real defect beyond this
    strength_scale=(0.75, 1.25),
    min_size_mm=1.0, mm_per_px=0.112,
    label_threshold=0.2,                     # box = pixels above 20% of the defect's peak change
    strength_mult=1.0,                       # global multiplier on all defect strengths
    class_strength={},                       # per-class multiplier, e.g. {"Impression line": 1.5}
    class_weights={},                        # per-class sampling weight (default: all classes equal)
    new_defects=True,                        # include the procedurally generated classes (scratch, torn hole, ...)
    keep_edge_width=True,                    # enlarged sample defects keep their real edge width
    light_angle_sd=0.5,                      # per-frame light direction spread (radians) for relief shading
    lighting="front",                        # "front" (reflected; your Basler frames) or "back" (transmitted)
    # defects' own internal texture (real Basler defects are 1.25-1.5x more textured inside than the paper):
    defect_texture_mid=(0.007, 0.014),       # MD-streaky mottling (0.3-1 mm across), relative to paper brightness
    defect_texture_fine=(0.002, 0.006),      # fine grain
)


class DefectLayer:
    def __init__(self, library_path, **overrides):
        self.cfg = dict(DEFAULTS, **overrides)
        # dark spots (big and small, as in your Basler frames) are 3x as frequent unless set otherwise
        self.cfg["class_weights"] = {"dark spot": 3.0, **self.cfg["class_weights"]}
        lib = list(np.load(library_path, allow_pickle=True))
        self.by_class = {c: [e for e in lib if e["cls"] == c] for c in CLASSES}
        self.by_class = {c: v for c, v in self.by_class.items() if v}
        self.classes = [c for c in self.by_class if c not in NEW_CLASSES]
        self.real_dark = self.by_class.get("dark spot", [])
        self.gen = NewDefectGenerator(self.cfg["mm_per_px"])
        self.new_classes = list(NEW_CLASSES) if self.cfg["new_defects"] else []
        # visible extent of each library defect (longest side, px), used to hit the target size
        for es in self.by_class.values():
            for e in es:
                sg = np.abs(e["signal"].astype(np.float32)); m = sg > 0.2 * sg.max()
                ys, xs = np.nonzero(m)
                e["extent"] = (float(ys.max() - ys.min() + 1), float(xs.max() - xs.min() + 1)) if len(ys) else (1.0, 1.0)
                e["edge_px"] = self._edge_width_px(e["signal"])
                ys5, xs5 = np.nonzero(cv2.GaussianBlur(sg, (0, 0), 1.0) > 0.55 * sg.max())
                e["extent_core"] = (float(np.ptp(ys5) + 1), float(np.ptp(xs5) + 1)) if len(ys5) else e["extent"]

    def _scale_for(self, e, tier, rng, clip=True):
        cfg = self.cfg
        if tier is None or cfg["size_tiers"] is None:
            return rng.uniform(*cfg["size_scale"])
        if e["cls"] in MD_LINES:                                      # lines: tier scales the real width
            k = float(rng.uniform(*cfg["line_width_tiers"][tier]))
            wmax = cfg["line_max_width_mm"].get(e["cls"], 1e9) / cfg["mm_per_px"]
            return min(k, wmax / max(e["extent"][1], 1.0))
        lo, hi = cfg["size_tiers"][tier]
        lo = max(lo, cfg["min_size_mm"] * 1.3)                        # margin so the visible size stays >= min
        hi = min(hi, CLASS_MAX_MM.get(e["cls"], hi)); lo = min(lo, hi * 0.8)
        target_px = np.exp(rng.uniform(np.log(lo), np.log(max(hi, lo * 1.01)))) / cfg["mm_per_px"]
        ey, ex = e["extent_core"] if (cfg["keep_edge_width"] and "extent_core" in e) else e["extent"]
        k = target_px / max(ey, ex, 1.0)
        return float(np.clip(k, *cfg["scale_limits"])) if clip else float(k)

    def _feasible(self, e, tier, frame_h=1024, md_stretch=2.0):
        """Can this example reach the tier's size range within scale limits (and fit the frame)?"""
        if tier is None or e["cls"] in MD_LINES:
            return True
        cfg = self.cfg
        lo, hi = cfg["size_tiers"][tier]; lo = max(lo, cfg["min_size_mm"] * 1.3)
        if lo > CLASS_MAX_MM.get(e["cls"], 1e9):
            return False
        ext = max(e["extent_core"] if (cfg["keep_edge_width"] and "extent_core" in e) else e["extent"]) + 1e-6
        k_lo, k_hi = lo / cfg["mm_per_px"] / ext, hi / cfg["mm_per_px"] / ext
        kmax = min(cfg["scale_limits"][1], cfg["max_height_share"] * frame_h / (e["signal"].shape[0] * md_stretch))
        return k_hi >= cfg["scale_limits"][0] and k_lo <= kmax

    def _pick(self, rng, tier, allow_lines=True, used=()):
        """Pick a class evenly among the classes that can produce this size, then a fitting example
        (an example already used in this frame is avoided whenever another one fits)."""
        classes = [c for c in self.classes if allow_lines or c not in MD_LINES]
        ok = {c: [e for e in self.by_class[c] if self._feasible(e, tier)] for c in classes}
        ok = {c: ([e for e in v if e["ann_id"] not in used] or v) for c, v in ok.items()}
        cands = [c for c in classes if ok[c]] or classes
        cands += [c for c in self.new_classes if (tier is None or tier in NEW_SIZES[c]) and (allow_lines or c not in MD_LINES)]
        wts = np.array([self.cfg["class_weights"].get(k, 1.0) for k in cands], np.float64)
        c = cands[int(rng.choice(len(cands), p=wts / wts.sum()))]
        if c in NEW_CLASSES:
            return c, None
        pool = ok.get(c) or self.by_class[c]
        return c, pool[int(rng.integers(len(pool)))]

    @staticmethod
    def _edge_width_px(sig):
        """Real edge width of a library defect (10-90 %, in its own pixels) from nested contour areas."""
        a = np.abs(sig.astype(np.float32)); a = cv2.GaussianBlur(a, (0, 0), 0.7); a /= a.max() + 1e-9
        r = lambda lvl: np.sqrt((a > lvl).sum() / np.pi)
        w = (r(0.2) - r(0.8)) / 0.75                     # 20-80 % distance scaled to 10-90 %
        return float(np.clip(w, 1.5, 40.0))

    @staticmethod
    def _restore_edges(s, edge_px, thr=0.55, flatten=0.6):
        """Rebuild the outline of an enlarged defect with its REAL edge width.
        Shape and internal structure come from the enlarged signal; only the boundary is redrawn."""
        from layer4b_new_defects import soft_alpha
        peak = np.abs(s).max() + 1e-9
        a = cv2.GaussianBlur(np.abs(s), (0, 0), 1.0) / peak
        mask = a > thr
        if mask.sum() < 9:
            return s
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        keep = np.zeros_like(mask)
        big = st[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):                                  # keep main parts, drop crumbs
            if st[i, cv2.CC_STAT_AREA] >= 0.05 * big: keep |= lab == i
        alpha = soft_alpha(keep, edge_px)
        sgn = np.sign(cv2.GaussianBlur(s, (0, 0), 2.0) + 1e-12)
        # interior: keep its pattern, but flatten the enlarged slope towards the core (real larger
        # defects are not wide V-shapes); fill up to the outline at the threshold level
        core = sgn * thr * peak + (s - sgn * thr * peak) * flatten
        inner = np.where(a > thr, core, sgn * thr * peak)
        inner = cv2.GaussianBlur(inner, (0, 0), 0.6)
        return (inner * alpha).astype(np.float32)

    @staticmethod
    def _vary_shape(s, rng, max_rot=15.0):
        """Per-placement variety: small rotation, aspect jitter and a gentle smooth warp,
        so the same real example never appears twice identically."""
        h, w = s.shape
        p = int(0.12 * max(h, w)) + 2
        s = cv2.copyMakeBorder(s, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        H, W = s.shape
        ang = rng.uniform(-max_rot, max_rot); sx, sy = rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0); M[:, :2] *= np.array([[sx], [sy]])
        M[:, 2] += (np.array([W / 2, H / 2]) - M[:, :2] @ np.array([W / 2, H / 2]))
        s = cv2.warpAffine(s, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        amp = 0.025 * max(h, w)                                    # gentle elastic warp
        def field():
            f = cv2.resize(rng.standard_normal((4, 4)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
            return f * amp
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        s = cv2.remap(s, gx + field(), gy + field(), cv2.INTER_LINEAR, borderValue=0)
        ys, xs = np.nonzero(np.abs(s) > 1e-5)
        if len(ys):
            s = s[max(0, ys.min() - 3):ys.max() + 4, max(0, xs.min() - 3):xs.max() + 4]
        return s

    NO_OWN_TEXTURE = {"pinhole", "torn hole", "edge crack", "edge cut", "calender cut", "felt hair"} | set(MD_LINES) | set(TRANSMISSION_CLASSES)

    def _add_texture(self, s, c, rng, md_stretch=1.8):
        """Give a defect its own internal texture (mottling + fine grain) on top of the paper texture.
        Strength follows the local defect depth, so it fades out towards the edges."""
        if c in self.NO_OWN_TEXTURE or s.size < 16:
            return s
        from layer4b_new_defects import _lf
        h, w = s.shape
        peak = np.abs(s).max() + 1e-9
        wgt = np.clip(np.abs(cv2.GaussianBlur(s, (0, 0), 1.0)) / peak, 0, 1)
        wgt = wgt * wgt * (3 - 2 * wgt)
        # like the real defects' texture: grain streaked along the machine direction (3-4x elongated)
        asp = rng.uniform(3.0, 4.0)
        hs = max(3, int(round(h / (md_stretch * asp / 1.8))))
        mid = _lf(hs, w, rng, rng.uniform(2.5, 8)); mid = cv2.resize(mid, (w, h), interpolation=cv2.INTER_CUBIC)
        hf = max(3, int(round(h / 2.0)))
        fine = cv2.GaussianBlur(rng.standard_normal((hf, w)).astype(np.float32), (0, 0), 0.7)
        fine = cv2.resize(fine / (fine.std() + 1e-9), (w, h), interpolation=cv2.INTER_LINEAR)
        depth = float(np.clip(peak / 0.12, 0.6, 1.6))                # deeper defects are more uneven inside
        a_mid = rng.uniform(*self.cfg["defect_texture_mid"]) * depth
        a_fine = rng.uniform(*self.cfg["defect_texture_fine"]) * depth
        return (s + wgt * (a_mid * mid / (mid.std() + 1e-9) + a_fine * fine)).astype(np.float32)

    def _apply_lighting(self, s, c, rng):
        """Convert a defect signal to the configured lighting ("front" = reflected, "back" = transmitted).
        Uses its own random stream (one draw from rng), so front- and backlit runs place identical defects."""
        rng = np.random.default_rng(int(rng.integers(1 << 31)))
        back = self.cfg["lighting"] == "back"
        if c in TRANSMISSION_CLASSES:                  # light from behind the web shines through: bright in any image
            return s
        if c in TRANSLUCENT_CLASSES or c in ("slime hole", "calender cut"):
            return s if not back else (-0.8 * s).astype(np.float32)    # front-authored: through-light -> brighter
        if back and c in RELIEF:
            return (rng.uniform(0.25, 0.45) * s).astype(np.float32)   # surface relief barely shows in transmission
        if back and c in ("Dirt", "shive", "Deposit", "dark spot", "felt hair"):
            return (rng.uniform(1.1, 1.4) * s).astype(np.float32)     # opaque matter: darker in transmission
        if self.cfg["lighting"] != "front" or c not in HOLE_LIKE:
            return s
        pos, neg = np.clip(s, 0, None), np.clip(s, None, 0)
        if c == "Layers missing":                      # thinner paper, not an opening: slightly darker
            g = rng.uniform(0.4, 0.8)
            return np.clip(neg - g * pos, -0.9, None).astype(np.float32)
        # openings: where the paper is fully open the camera sees the dark space behind the web
        # (real cut in frame 0187: down to ~35 GL, i.e. ~80 % darker); partial openings scale with it.
        from layer4b_new_defects import fbm
        h, w = s.shape
        o = cv2.GaussianBlur(pos / (pos.max() + 1e-9), (0, 0), 1.0)             # graded edge, not a hard outline
        o = np.clip(o / (o.max() + 1e-9), 0, 1)
        var = np.clip(1 + 0.04 * fbm(h, w, rng, max(6.0, 0.3 * min(h, w)), 2), 0.9, 1.08)     # very slight unevenness
        if c == "blotches":
            # a blotch's bright part (backlit) is thin, translucent sheet rather than a clean opening:
            # under front light it is moderately darker, comparable to its tail, so the two read as one defect
            open_dark = rng.uniform(0.15, 0.35) * o ** 0.8 * var
        else:                                                                       # true openings
            open_dark = rng.uniform(0.65, 0.85) * o ** 0.8 * var
        tail_dark = cv2.GaussianBlur(-neg, (0, 0), 0.6)
        if c != "blotches":
            # the dark torn rim of a backlit pinhole / torn hole is denser fibre that blocks transmitted light;
            # under front light it does not show as a separate shadow beside the opening
            tail_dark *= 0.0
        # one continuous darkening: each point takes the stronger of tail / opening, blended smoothly
        # (a sum would stack a black patch on top of the grey tail)
        p = 3.0
        dark = (tail_dark ** p + open_dark ** p) ** (1 / p)
        # torn fibres around the opening catch the front light: thin, faint lighter rim,
        # suppressed where the dark tail joins so it never outlines the join
        core = (o > 0.5).astype(np.uint8)
        k = max(3, int(2 * rng.uniform(0.25, 0.55) / self.cfg["mm_per_px"]) | 1)
        ring = np.clip(cv2.GaussianBlur((cv2.dilate(core, np.ones((k, k), np.uint8)) - core).astype(np.float32), (0, 0), 0.8), 0, 1)
        tail_n = tail_dark / (tail_dark.max() + 1e-9)
        rim = rng.uniform(0.02, 0.06) * ring * (1 - o) * np.clip(1 - 1.5 * tail_n, 0, 1)
        return np.clip(-dark + rim, -0.9, None).astype(np.float32)

    def _prepare(self, e, rng, md_stretch, frame_h, tier=None):
        s = e["signal"].astype(np.float32)
        c = e["cls"]
        if rng.random() < 0.5: s = s[:, ::-1]
        if c not in NO_VFLIP and rng.random() < 0.5: s = s[::-1, :]
        if c not in MD_LINES:
            s = self._vary_shape(s, rng)
        k = self._scale_for(e, tier, rng)
        if c not in MD_LINES:                                         # must fit the frame height, uncut
            k = min(k, self.cfg["max_height_share"] * frame_h / (s.shape[0] * md_stretch))
        h, w = s.shape
        ky = 1.0 if c in MD_LINES else k                              # MD lines keep their length
        nw, nh = max(3, int(round(w * k))), max(3, int(round(h * ky * md_stretch)))
        s = cv2.resize(s, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        if self.cfg["keep_edge_width"] and k > 1.2 and c in EDGE_FIX_CLASSES:
            s = self._restore_edges(s, e["edge_px"] * rng.uniform(0.9, 1.3))
        if c in LINE_PEAK:
            s *= rng.uniform(*LINE_PEAK[c]) / (np.abs(s).max() + 1e-9)
        else:
            s *= rng.uniform(*self.cfg["strength_scale"])
        s *= self.cfg["strength_mult"] * self.cfg["class_strength"].get(c, 1.0)
        if s.shape[0] > frame_h:
            if c in MD_LINES:                                   # any window along the line
                y0 = int(rng.integers(0, s.shape[0] - frame_h + 1))
            else:                                               # window that contains the defect content
                e_row = np.abs(s).sum(1); cs = np.concatenate([[0], np.cumsum(e_row)])
                win = cs[frame_h:] - cs[:-frame_h]
                good = np.nonzero(win >= 0.6 * win.max())[0]
                y0 = int(rng.choice(good))
            s = s[y0:y0 + frame_h]
        return s

    def apply(self, paper, rng, md_stretch=1.8, n=None):
        """Adds defects to a copy of paper. Returns (paper, annotations)."""
        H, W = paper.shape
        out = paper.copy()
        cfg = self.cfg
        if n is None:
            n = 0 if rng.random() < cfg["defect_free"] else int(rng.integers(cfg["defects_per_frame"][0], cfg["defects_per_frame"][1] + 1))
        anns, boxes = [], []
        tiers = list(cfg["size_tiers"]) if cfg["size_tiers"] else [None]
        if cfg["size_tiers"] and cfg["mix_sizes"] and n >= len(tiers):
            plan = tiers + [tiers[int(rng.integers(len(tiers)))] for _ in range(n - len(tiers))]
        else:
            plan = [tiers[int(rng.integers(len(tiers)))] for _ in range(n)]
        plan = [plan[i] for i in rng.permutation(len(plan))]
        n_lines = 0
        used = set()
        light = float(rng.normal(0, cfg["light_angle_sd"]))              # one lamp direction per frame
        for tier in plan:
            c, e = self._pick(rng, tier, allow_lines=n_lines < cfg["max_lines_per_frame"], used=used)
            if e is not None: used.add(e["ann_id"])
            n_lines += c in MD_LINES
            real_ok = [x for x in self.real_dark if self._feasible(x, tier) and x["ann_id"] not in used] if c == "dark spot" else []
            if c == "dark spot" and real_ok and rng.random() < 0.4:      # real dark spot from your Basler frame
                e = real_ok[int(rng.integers(len(real_ok)))]; used.add(e["ann_id"])
                s = self._prepare(e, rng, md_stretch, H, tier)
            elif c in NEW_CLASSES:
                s = self.gen.render(c, tier, rng, md_stretch, H, light, cfg["max_height_share"])
                s *= cfg["strength_mult"] * cfg["class_strength"].get(c, 1.0)
                e = {"ann_id": -1}
            else:
                s = self._prepare(e, rng, md_stretch, H, tier)
            s = self._apply_lighting(s, c, rng)
            s = self._add_texture(s, c, rng, md_stretch)
            ph, pw = s.shape
            peak = float(np.abs(s).max())
            m = np.abs(s) > cfg["label_threshold"] * peak
            if not m.any(): continue
            ys, xs = np.nonzero(m)
            if c not in MD_LINES and max(xs.max() - xs.min() + 1, (ys.max() - ys.min() + 1) / md_stretch) * cfg["mm_per_px"] < cfg["min_size_mm"]:
                continue                                              # below the minimum defect size
            by0, by1, bx0, bx1 = ys.min(), ys.max(), xs.min(), xs.max()
            for _try in range(40):                                          # avoid overlapping defects
                if c in MD_LINES:
                    y0 = (H - ph) // 2 if ph <= H else 0
                else:
                    y0 = int(rng.integers(-by0, H - by1)) if H - by1 > -by0 else 0
                x0 = int(rng.integers(-bx0, W - bx1))
                box = (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)
                if all(box[2] < b[0] - 20 or box[0] > b[2] + 20 or box[3] < b[1] - 20 or box[1] > b[3] + 20 for b in boxes):
                    break
            ys0, xs0 = max(0, y0), max(0, x0); ye, xe = min(H, y0 + ph), min(W, x0 + pw)
            if ye <= ys0 or xe <= xs0: continue
            out[ys0:ye, xs0:xe] += s[ys0 - y0:ye - y0, xs0 - x0:xe - x0]
            mm = m[ys0 - y0:ye - y0, xs0 - x0:xe - x0]
            yy, xx = np.nonzero(mm)
            bx = (int(xs0 + xx.min()), int(ys0 + yy.min()), int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1))
            boxes.append((bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]))
            full = np.zeros((H, W), bool); full[ys0:ye, xs0:xe] = mm
            size_mm = max(bx[2], bx[3] / md_stretch) * cfg["mm_per_px"]
            anns.append(dict(cls=c, cls_id=CLASSES.index(c), bbox=bx, src_ann=int(e["ann_id"]), mask=full, peak=peak,
                             tier=tier, size_mm=round(float(size_mm), 1)))
        return out, anns

    # ======================================================================================
    # Continuous-web mode: plan defects along a stretch of consecutive frames (as the camera
    # sees the running web), then cut the stretch into frames.
    #   * defects crossing a frame boundary appear partly in both frames (each part labelled)
    #   * repeating roll marks: same defect at the same CD position every roll circumference
    #   * MD lines run through several frames and may start / stop mid-stretch
    #   * clusters: some small spots get 2-5 neighbours nearby (may overlap)
    # ======================================================================================
    REPEAT_CLASSES = ["dark spot", "Dirt", "pinhole", "torn hole", "blister", "coating skip", "starch spot"]
    CLUSTER_CLASSES = {"Dirt", "dark spot", "starch spot", "Deposit", "shive", "pinhole"}

    def _make_signal(self, c, e, tier, rng, md_stretch, fit_h, light, used):
        if c == "dark spot":
            real_ok = [x for x in self.real_dark if self._feasible(x, tier) and x["ann_id"] not in used]
            if real_ok and rng.random() < 0.4:
                e = real_ok[int(rng.integers(len(real_ok)))]; used.add(e["ann_id"])
                s = self._apply_lighting(self._prepare(e, rng, md_stretch, fit_h, tier), c, rng)
                return self._add_texture(s, c, rng, md_stretch), e
        if c in NEW_CLASSES:
            s = self.gen.render(c, tier, rng, md_stretch, fit_h, light, self.cfg["max_height_share"])
            s *= self.cfg["strength_mult"] * self.cfg["class_strength"].get(c, 1.0)
            tex = self.gen.last_tex
            return self._add_texture(self._apply_lighting(s, c, rng), c, rng, md_stretch), {"ann_id": -1, "tex": tex}
        s = self._apply_lighting(self._prepare(e, rng, md_stretch, fit_h, tier), c, rng)
        return self._add_texture(s, c, rng, md_stretch), e

    def _long_line(self, e, rng, md_stretch, span):
        """A line defect `span` rows long: windows of the (long) library line joined with cross-fades."""
        parts, total = [], 0
        while total < span:
            p = self._prepare(e, rng, md_stretch, max(64, min(span - total + 200, 4096)), None)
            parts.append(p); total += p.shape[0] - (200 if len(parts) > 1 else 0)
        w = max(p.shape[1] for p in parts)
        parts = [np.pad(p, ((0, 0), (0, w - p.shape[1]))) for p in parts]
        out = parts[0]
        for p in parts[1:]:
            o = min(200, out.shape[0] // 2, p.shape[0] // 2)
            t = np.linspace(0, 1, o, dtype=np.float32)[:, None]
            out = np.vstack([out[:-o], out[-o:] * (1 - t) + p[:o] * t, p[o:]])
        s = out[:span]
        ramp = np.ones(s.shape[0], np.float32); k = min(120, s.shape[0] // 4)
        ramp[:k] = np.linspace(0, 1, k); ramp[-k:] = np.linspace(1, 0, k)        # line starts / stops softly
        return s * ramp[:, None]

    def plan_web(self, n_frames, H, W, rng, md_stretch=1.8, repeats_mean=1.5, cluster_prob=0.15, max_lines=2):
        """Plan every defect of a stretch of n_frames consecutive frames (strip coordinates)."""
        cfg = self.cfg; L = n_frames * H
        light = float(rng.normal(0, cfg["light_angle_sd"]))
        if rng.random() < cfg["defect_free"]:                              # a completely clean stretch of web
            return [], light
        items, boxes, used, gid = [], [], set(), 0
        tiers = list(cfg["size_tiers"])

        def place(s, c, e, tier, x0=None, y_centre=None, group=None, kind="random", check=True):
            nonlocal gid
            ph, pw = s.shape; peak = float(np.abs(s).max())
            if peak <= 0: return None
            m = np.abs(s) > cfg["label_threshold"] * peak
            ys, xs = np.nonzero(m)
            if not len(ys): return None
            for _t in range(40 if check else 1):
                xx0 = x0 if x0 is not None else int(rng.integers(-xs.min(), W - xs.max()))
                yy0 = int(y_centre - ph // 2) if y_centre is not None else int(rng.integers(-ph // 2, L - ph // 2))
                box = (xx0 + xs.min(), yy0 + ys.min(), xx0 + xs.max(), yy0 + ys.max())
                if not check or all(box[2] < b[0] - 20 or box[0] > b[2] + 20 or box[3] < b[1] - 20 or box[1] > b[3] + 20 for b in boxes):
                    break
                y_centre = None if kind == "random" else y_centre
            boxes.append(box); gid += 1
            it = dict(s=s, m=m, x0=xx0, y0=yy0, cls=c, cls_id=CLASSES.index(c), tier=tier, peak=peak,
                      tex=(e or {}).get("tex") if c in TRANSLUCENT_CLASSES else None,
                      src_ann=int(e["ann_id"]) if e else -1, gid=gid, group=group if group is not None else gid, kind=kind)
            items.append(it); return it

        # 1) random defects, drawn per frame (keeps per-frame density and defect-free frames)
        n_lines = 0
        for fi in range(n_frames):
            n = int(rng.integers(cfg["defects_per_frame"][0], cfg["defects_per_frame"][1] + 1))
            plan = (tiers + [tiers[int(rng.integers(len(tiers)))] for _ in range(n - len(tiers))]) if n >= len(tiers) else \
                   [tiers[int(rng.integers(len(tiers)))] for _ in range(n)]
            for tier in [plan[i] for i in rng.permutation(len(plan))]:
                c, e = self._pick(rng, tier, allow_lines=n_lines < max_lines, used=used)
                if e is not None: used.add(e["ann_id"])
                if c in MD_LINES:                                  # a line running through several frames
                    n_lines += 1
                    span = int(rng.integers(H // 2, L + 1))
                    if c in NEW_CLASSES:                            # procedural line (blade scratch)
                        s = self.gen.render(c, tier, rng, md_stretch, 10 ** 9, light, 1.0, size_mm=span / md_stretch * cfg["mm_per_px"])
                        s = s[:span]
                        y_c = int(rng.integers(span // 2, L - span // 2 + 1)) if span < L else L // 2
                        place(s, c, None, tier, y_centre=y_c, kind="line")
                        continue
                    k = self._scale_for(e, tier, rng)
                    s = self._long_line(e, rng, md_stretch, span)
                    s = cv2.resize(s, (max(3, int(s.shape[1] * k)), s.shape[0]), interpolation=cv2.INTER_LINEAR) if abs(k - 1) > 0.05 else s
                    if c in LINE_PEAK: s *= rng.uniform(*LINE_PEAK[c]) / (np.abs(s).max() + 1e-9)
                    y_c = int(rng.integers(span // 2, L - span // 2 + 1)) if span < L else L // 2
                    place(s, c, e, tier, y_centre=y_c, kind="line")
                    continue
                s, e = self._make_signal(c, e, tier, rng, md_stretch, H, light, used)
                it = place(s, c, e, tier, y_centre=int(fi * H + rng.integers(0, H)))
                # clusters of small spots
                if it is not None and tier == "small" and c in self.CLUSTER_CLASSES and rng.random() < cluster_prob:
                    cx = it["x0"] + it["s"].shape[1] // 2; cy = it["y0"] + it["s"].shape[0] // 2
                    for _ in range(int(rng.integers(2, 6))):
                        s2, e2 = self._make_signal(c, e if e and e.get("ann_id", -1) >= 0 else None, "small", rng, md_stretch, H, light, used) \
                            if c in NEW_CLASSES else self._make_signal(c, e, "small", rng, md_stretch, H, light, used)
                        r = rng.uniform(5, 30) / cfg["mm_per_px"]; a = rng.uniform(0, 2 * np.pi)
                        place(s2, c, e2, "small", x0=int(np.clip(cx + r * np.cos(a) - s2.shape[1] // 2, 0, W - s2.shape[1])),
                              y_centre=int(cy + r * np.sin(a) * md_stretch), group=it["group"], kind="cluster", check=False)
        # 2) repeating roll marks: same CD position, pitch = roll circumference (60-250 mm)
        for _ in range(rng.poisson(repeats_mean)):
            c = self.REPEAT_CLASSES[int(rng.integers(len(self.REPEAT_CLASSES)))]
            tier = "small" if rng.random() < 0.6 else "medium"
            e = None if c in NEW_CLASSES else (self._pick(rng, tier, used=used)[1] if c not in self.by_class else
                                                ([x for x in self.by_class[c] if self._feasible(x, tier)] or self.by_class[c])[0])
            if c not in NEW_CLASSES and e is None: continue
            base, e = self._make_signal(c, e, tier, rng, md_stretch, H, light, used)
            pitch = rng.uniform(60, 250) / cfg["mm_per_px"] * md_stretch                  # rows (roll circumference)
            x0 = int(rng.integers(0, max(1, W - base.shape[1]))); y = rng.uniform(0, pitch); grp = None
            while y < L:
                s = base * rng.uniform(0.85, 1.15)                                         # each turn slightly different
                it = place(s, c, e, tier, x0=x0, y_centre=int(y), group=grp, kind="repeat", check=False)
                if it is not None and grp is None: grp = it["group"]
                y += pitch
        return items, light

    def render_web_frame(self, paper, items, fi, md_stretch=1.8):
        """Add the part of every planned defect that falls inside frame fi; return (paper, annotations)."""
        H, W = paper.shape; top = fi * H
        out = paper.copy(); anns = []
        for it in items:
            s, m = it["s"], it["m"]; ph, pw = s.shape
            y0 = it["y0"] - top; x0 = it["x0"]
            ys0, xs0 = max(0, y0), max(0, x0); ye, xe = min(H, y0 + ph), min(W, x0 + pw)
            if ye <= ys0 or xe <= xs0: continue
            if it.get("tex") is not None:                              # translucent: paper structure shows more / less
                ta, tg = it["tex"]
                loc = paper[ys0:ye, xs0:xe]
                hp = loc - cv2.GaussianBlur(loc, (0, 0), 15)
                out[ys0:ye, xs0:xe] += tg * ta[ys0 - y0:ye - y0, xs0 - x0:xe - x0] * hp
            out[ys0:ye, xs0:xe] += s[ys0 - y0:ye - y0, xs0 - x0:xe - x0]
            mm = m[ys0 - y0:ye - y0, xs0 - x0:xe - x0]
            if not mm.any(): continue
            yy, xx = np.nonzero(mm)
            bx = (int(xs0 + xx.min()), int(ys0 + yy.min()), int(np.ptp(xx) + 1), int(np.ptp(yy) + 1))
            size_mm = max(bx[2], bx[3] / md_stretch) * self.cfg["mm_per_px"]
            if size_mm < self.cfg["min_size_mm"] and it["cls"] not in MD_LINES: continue   # sliver of a cut defect
            full = np.zeros((H, W), bool); full[ys0:ye, xs0:xe] = mm
            cut = bool(y0 < 0 or y0 + ph > H) and it["kind"] != "line"
            anns.append(dict(cls=it["cls"], cls_id=it["cls_id"], bbox=bx, src_ann=it["src_ann"], mask=full, peak=it["peak"],
                             tier=it["tier"], size_mm=round(float(size_mm), 1), gid=it["gid"], group=it["group"],
                             kind=it["kind"], cut=cut))
        return out, anns


    # ======================================================================================
    # Web edge (outermost camera): a continuous edge through the stretch, dark space beyond it
    # (front-lit), and edge cracks / edge cuts running in from the edge.
    # ======================================================================================
    def plan_edge(self, n_frames, H, W, rng, md_stretch=1.8, edge_prob=0.25):
        if rng.random() >= edge_prob:
            return None
        from layer4b_new_defects import _lf, soft_alpha
        L = n_frames * H; mm = self.cfg["mm_per_px"]
        side = "left" if rng.random() < 0.5 else "right"             # side where the web ENDS
        base = rng.uniform(0.04, 0.45) * W                            # distance of the edge from that frame side
        kind = "slit" if rng.random() < 0.6 else "deckle"
        wander = rng.uniform(0.3, 2.0) / mm * _lf(L, 1, rng, rng.uniform(300, 3000))[:, 0]
        if kind == "slit":
            fine = rng.uniform(0.02, 0.06) / mm * _lf(L, 1, rng, rng.uniform(10, 30))[:, 0]      # slit: nearly straight
            fine = cv2.GaussianBlur(fine[:, None].astype(np.float32), (0, 0), 2.0)[:, 0]
        else:
            fine = rng.uniform(0.2, 0.6) / mm * _lf(L, 1, rng, rng.uniform(15, 60))[:, 0] + 0.04 / mm * _lf(L, 1, rng, 4)[:, 0]
            fine = cv2.GaussianBlur(fine[:, None].astype(np.float32), (0, 0), 1.5)[:, 0]
        edge_x = (base + wander + fine).astype(np.float32)          # distance of the edge from the frame side, per row
        # edge defects
        openings = []
        for _ in range(int(rng.integers(2, 7))):
            cls = "edge crack" if rng.random() < 0.6 else "edge cut"
            yc = rng.uniform(0, L)
            m = self._edge_opening(cls, rng, md_stretch)             # patch: column 0 = at the edge, grows into the web
            openings.append(dict(cls=cls, m=m, yc=yc))
        return dict(side=side, edge_x=edge_x, kind=kind, openings=openings, rim=float(rng.uniform(-0.06, 0.08)), rim_w=float(rng.uniform(1.0, 3.0)),
                    bg=float(rng.uniform(0.12, 0.35)), bg_seed=int(rng.integers(1 << 30)))

    def _edge_opening(self, cls, rng, md):
        """Opening mask (float 0..1) in a patch whose column 0 lies on the edge; +x = into the web."""
        mm = self.cfg["mm_per_px"]; px = lambda v: v / mm
        if cls == "edge crack":
            Lc = px(rng.uniform(2, 25)); w0 = px(rng.uniform(0.3, 2.0))
            n = int(Lc) + 1; a = rng.normal(0, 0.35); x, y = 0.0, 0.0; pts = [(x, y)]
            for i in range(n):
                a = 0.7 * a + rng.normal(0, 0.18); x += np.cos(a); y += np.sin(a); pts.append((x, y))
            pts = np.array(pts, np.float32)
            pad = int(w0 + px(1)); ymin = pts[:, 1].min() - pad
            Hh, Ww = int(pts[:, 1].max() - ymin + pad) + 1, int(pts[:, 0].max() + pad) + 1
            c = np.zeros((Hh * 4, Ww * 4), np.uint8)
            for i in range(len(pts) - 1):
                t = max(1, int(round(4 * w0 * (1 - i / len(pts)) ** 1.3)))
                p0 = (int(pts[i, 0] * 4), int((pts[i, 1] - ymin) * 4)); p1 = (int(pts[i + 1, 0] * 4), int((pts[i + 1, 1] - ymin) * 4))
                cv2.line(c, p0, p1, 255, t, cv2.LINE_AA)
            m = cv2.resize(c.astype(np.float32) / 255, (Ww, Hh), interpolation=cv2.INTER_AREA)
        else:                                                         # edge cut: asymmetric V-shaped tear with ragged sides
            b = px(rng.uniform(3, 30)); d = b * rng.uniform(0.5, 1.4)
            tip_y = b * rng.uniform(0.25, 0.75) + rng.normal(0, 0.15 * b)          # tip offset -> slanted tear
            tip = np.array([d, tip_y]); p_top = np.array([0.0, 0.0]); p_bot = np.array([0.0, b])
            def torn(p0, p1):
                n = max(3, int(np.hypot(*(p1 - p0)) / px(0.4))); pts = []
                nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / (np.hypot(*(p1 - p0)) + 1e-6)
                off = np.cumsum(rng.normal(0, px(0.12), n)); off -= np.linspace(off[0], off[-1], n)   # wanders, ends pinned
                for i, t in enumerate(np.linspace(0, 1, n, endpoint=False)):
                    pts.append(p0 + t * (p1 - p0) + nrm * off[i] + rng.normal(0, px(0.05), 2))
                return pts
            poly = np.array(torn(p_top, tip) + torn(tip, p_bot) + [p_bot], np.float32)
            poly[:, 0] = np.maximum(poly[:, 0], 0)
            ymin = min(0.0, poly[:, 1].min()); poly[:, 1] -= ymin
            Hh, Ww = int(poly[:, 1].max()) + 3, int(poly[:, 0].max()) + 4
            c = np.zeros((Hh * 4, Ww * 4), np.uint8)
            cv2.fillPoly(c, [np.round(poly * 4).astype(np.int32)], 255, cv2.LINE_AA)
            m = cv2.resize(c.astype(np.float32) / 255, (Ww, Hh), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (m.shape[1], max(2, int(round(m.shape[0] * md)))), interpolation=cv2.INTER_LINEAR)   # MD stretch
        return np.clip(m, 0, 1).astype(np.float32)

    def render_edge_frame(self, paper, edge, fi, anns, md_stretch=1.8):
        """Apply the web edge to frame fi: dark space beyond the edge and in edge openings.
        Clips existing defect labels to the web and adds labels for edge cracks / cuts."""
        if edge is None:
            return paper, anns
        from layer4b_new_defects import fbm
        H, W = paper.shape; top = fi * H
        ex = edge["edge_x"][top:top + H]
        xs = np.arange(W, dtype=np.float32)[None, :]
        dist = (xs - ex[:, None]) if edge["side"] == "left" else ((W - 1 - xs) - ex[:, None])   # >0 on the web
        web = np.clip(dist / 1.2 + 0.5, 0, 1)
        new_anns = []
        for op in edge["openings"]:
            m = op["m"]; ph, pw = m.shape
            y0 = int(op["yc"] - ph / 2) - top
            if y0 >= H or y0 + ph <= 0: continue
            ys0, ye = max(0, y0), min(H, y0 + ph)
            full = np.zeros((H, W), np.float32)
            for r in range(ys0, ye):                                  # attach the patch to the edge, row by row
                e_ = ex[r]; row = m[r - y0]
                if edge["side"] == "left":
                    c0 = int(round(e_)) - 1; c1 = min(W, c0 + pw)
                    if c1 > max(0, c0): full[r, max(0, c0):c1] = row[max(0, -c0):c1 - c0]
                else:
                    c1 = W - int(round(e_)) + 1; c0 = max(0, c1 - pw)
                    if c1 > c0: full[r, c0:c1] = row[::-1][pw - (c1 - c0):]
            full *= (dist > -1)
            web = web * (1 - full)
            mk = full > 0.5
            if mk.sum() < 6: continue
            yy, xx = np.nonzero(mk)
            bx = (int(xx.min()), int(yy.min()), int(np.ptp(xx) + 1), int(np.ptp(yy) + 1))
            size_mm = max(bx[2], bx[3] / md_stretch) * self.cfg["mm_per_px"]
            if size_mm < self.cfg["min_size_mm"]: continue
            new_anns.append(dict(cls=op["cls"], cls_id=CLASSES.index(op["cls"]), bbox=bx, src_ann=-1, mask=mk, peak=1.0,
                                 tier="small" if size_mm < 5 else "medium" if size_mm < 20 else "large",
                                 size_mm=round(float(size_mm), 1), gid=-1, group=-1, kind="edge",
                                 cut=bool(y0 < 0 or y0 + ph > H)))
        bg_level = edge["bg"] if self.cfg["lighting"] != "back" else 3.0           # transmission: direct backlight, fully saturated
        bg = bg_level * (1 + 0.06 * fbm(H, W, np.random.default_rng([edge["bg_seed"], fi]), 400, 3))
        rim = edge["rim"] * np.exp(-np.clip(dist, 0, None) / edge["rim_w"]) * web       # edge catches light / shadow
        out = (paper + rim) * web + bg * (1 - web)
        kept = []
        for an in anns:                                               # defects only exist on the web
            mk = an["mask"] & (web > 0.5)
            if mk.sum() < 6: continue
            yy, xx = np.nonzero(mk)
            bx = (int(xx.min()), int(yy.min()), int(np.ptp(xx) + 1), int(np.ptp(yy) + 1))
            if max(bx[2], bx[3] / md_stretch) * self.cfg["mm_per_px"] < self.cfg["min_size_mm"] and an["cls"] not in MD_LINES: continue
            an = dict(an, mask=mk, bbox=bx); kept.append(an)
        return out.astype(np.float32), kept + new_anns
