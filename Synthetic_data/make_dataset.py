#!/usr/bin/env python3
"""
Synthetic WIS dataset: line-scan paper frames with defects transplanted from your samples.

Per frame (all built on the same paper and the same random draws):
    layer 1  formation (Basler-calibrated)
    layer 2  fibres, specks, faint spots
    layer 4  defects from the sample library (texture-preserving, soft edges)
    camera   four versions: shading / prnu / dsnu / combined
Outputs:
    <out>/images/<version>/<split>/frame_000001.png     (version = shading, prnu, dsnu, combined)
    <out>/labels/<split>/frame_000001.txt                YOLO boxes, shared by all four versions
    <out>/masks/<split>/frame_000001.png                 pixel = class id + 1 (0 = no defect)
    <out>/annotations/instances_<split>.json             COCO boxes
    <out>/frames.csv                                     per-frame settings and defects
    <out>/dataset.yaml, classes.txt
Usage:
    python make_dataset.py --n 200 --out wis_synth --workers 8
    python make_dataset.py --n 50 --prnu-mult 3 --dsnu-mult 3      # exaggerated camera stripes
    python make_dataset.py --n 200 --class-strength "Impression line=1.5" bladeline=1.5 "Coating mark=1.5"
    python make_dataset.py --n 200 --defect-strength 1.3 --class-weight pinhole=2
"""
import argparse, csv, json, os, sys
from multiprocessing import Pool
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VERSIONS = ["shading", "prnu", "dsnu", "combined"]
_M = {}
_DEFECT_CFG = {}
_LOOKALIKE_CFG = {}


def _parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.rsplit("=", 1); out[k.strip()] = float(v)
    return out


def _init(defect_cfg=None, lookalike_cfg=None):
    _DEFECT_CFG.update(defect_cfg or {})
    _LOOKALIKE_CFG.update(lookalike_cfg or {})
    sys.path.insert(0, HERE)
    from layer1_formation import FormationModel
    from layer2_fibres import FibreSpeckLayer
    from layer3_marks import LookalikeLayer
    from layer4_defects import DefectLayer
    from layer_camera import CameraModel
    _M.update(F=FormationModel(os.path.join(HERE, "layer1_formation_model.npz")), L2=FibreSpeckLayer(),
              L3=LookalikeLayer(**_LOOKALIKE_CFG),
              D=DefectLayer(os.path.join(HERE, "defect_library.npy"), **_DEFECT_CFG), C=CameraModel(os.path.join(HERE, "camera_model.npz")))


def _visibility(a, imgs, anns):
    """Step 7: score every label in the final (combined) image; drop invisible ones, flag difficult ones."""
    from layer_visibility import defect_cnr
    if not anns:
        return anns, 0
    img = imgs["combined"]
    union = np.zeros(anns[0]["mask"].shape, bool)
    for an in anns: union |= an["mask"]
    kept, dropped = [], 0
    for an in anns:
        v = defect_cnr(img, an["mask"], union & ~an["mask"], an["cls"])
        v = 99.0 if not np.isfinite(v) else v                      # no clean ring (crowded): keep, cannot judge
        if v < a.vis_drop:
            dropped += 1; continue
        kept.append(dict(an, visibility=round(float(v), 2), difficult=bool(v < a.vis_difficult)))
    return kept, dropped


def _write_frame(a, split, name, imgs, anns, extra):
    anns, n_dropped = _visibility(a, imgs, anns)
    for v in VERSIONS:
        cv2.imwrite(os.path.join(a.out, "images", v, split, name + ".png"), imgs[v])
    H, W = a.height, a.width
    mask = np.zeros((H, W), np.uint8)
    with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
        for an in anns:
            if a.drop_difficult and an["difficult"]:
                continue
            x, y, w, h = an["bbox"]
            f.write(f"{an['cls_id']} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
            mask[an["mask"]] = an["cls_id"] + 1
    cv2.imwrite(os.path.join(a.out, "masks", split, name + ".png"), mask)
    row = dict(file=name + ".png", split=split, **extra, n_defects=len(anns),
               n_difficult=sum(an["difficult"] for an in anns), n_dropped_invisible=n_dropped,
               defects=";".join(f'{an["cls"]} ({an["tier"]}, {an["size_mm"]} mm)' for an in anns),
               defect_flags=";".join(("edge|" if an.get("kind") == "edge" else "") + ("cut" if an.get("cut") else "") + ("|" if an.get("cut") and an.get("kind") in ("repeat", "cluster", "line") else "")
                                     + (f'{an["kind"]}#{an["group"]}' if an.get("kind") in ("repeat", "cluster") else ("line" if an.get("kind") == "line" else ""))
                                     for an in anns))
    return row, [dict(cls_id=an["cls_id"], bbox=an["bbox"], area=int(an["mask"].sum()),
                      visibility=an["visibility"], difficult=an["difficult"]) for an in anns], name


def make_frame(job):
    """Independent single frame (--sequence-length 1)."""
    idx, split, a = job
    rng = np.random.default_rng([a.seed, idx])
    paper = 1 + _M["F"].generate(a.height, a.width, rng)[0]
    paper, p2, _ = _M["L2"].apply(paper, rng)
    paper, looks = _M["L3"].apply(paper, _M["L3"].plan(1, a.height, a.width, rng, p2["md_stretch"]), 0, rng, p2["md_stretch"])
    paper, anns = _M["D"].apply(paper, rng, md_stretch=p2["md_stretch"])
    mult = {"shading": a.shading_mult, "prnu": a.prnu_mult, "dsnu": a.dsnu_mult}
    imgs, pc, _ = _M["C"].apply_variants(paper, rng, mult=mult)
    extra = dict(sequence=idx, frame_in_sequence=0, lighting=a.lighting, web_edge="none", lookalikes=";".join(looks), md_stretch=round(p2["md_stretch"], 3), paper_cleanliness=round(p2["density"], 3),
                 shading_strength=round(pc["shading_strength"], 3), prnu_std=round(pc["prnu_std"], 5), dsnu_std=round(pc["dsnu_std"], 3),
                 irregular_light=pc["irregular_light"])
    return [_write_frame(a, split, f"frame_{idx:06d}", imgs, anns, extra)]


def make_sequence(job):
    """A continuous stretch of web: consecutive frames sharing one defect plan, one camera
    (fixed shading / PRNU / DSNU) and one machine speed (MD stretch); noise changes per frame."""
    sq, split, a = job
    rng = np.random.default_rng([a.seed, sq])
    lighting = a.lighting if a.lighting != "mixed" else ("back" if sq in a.back_sequences else "front")
    _M["D"].cfg["lighting"] = lighting
    fr_ = rng.random()                                                          # drawn in both modes (same defect plan)
    fkind = "basler" if lighting == "front" else ("calm" if fr_ < 0.6 else "cloudy")   # transmission: cloudy formation
    p2 = _M["L2"].sample_params(rng)
    md = p2["md_stretch"]
    items, _ = _M["D"].plan_web(a.sequence_length, a.height, a.width, rng, md_stretch=md)
    edge = _M["D"].plan_edge(a.sequence_length, a.height, a.width, rng, md_stretch=md,
                             edge_prob=1.0 if sq in a.edge_sequences else 0.0)          # assigned exactly, not by chance
    marks = _M["L3"].plan(a.sequence_length, a.height, a.width, rng, md_stretch=md)
    cam = _M["C"].sample_variant_params(rng)
    pattern_seed = int(rng.integers(1 << 30))
    mult = {"shading": a.shading_mult, "prnu": a.prnu_mult, "dsnu": a.dsnu_mult}
    out = []
    for fi in range(a.sequence_length):
        frng = np.random.default_rng([a.seed, sq, fi + 1])
        paper = 1 + _M["F"].generate(a.height, a.width, frng, params=_M["F"].sample_params(frng, kind=fkind))[0]
        paper, _, _ = _M["L2"].apply(paper, frng, params=p2)
        paper, looks = _M["L3"].apply(paper, marks, fi, frng, md_stretch=md)             # unlabelled look-alikes
        paper, anns = _M["D"].render_web_frame(paper, items, fi, md_stretch=md)
        paper, anns = _M["D"].render_edge_frame(paper, edge, fi, anns, md_stretch=md)
        imgs, pc, _ = _M["C"].apply_variants(paper, frng, params=cam, mult=mult, pattern_seed=pattern_seed)
        extra = dict(sequence=sq, frame_in_sequence=fi, lighting=lighting, web_edge=(f'{edge["side"]} ({edge["kind"]})' if edge else "none"),
                     lookalikes=";".join(looks), md_stretch=round(md, 3), paper_cleanliness=round(p2["density"], 3),
                     shading_strength=round(pc["shading_strength"], 3), prnu_std=round(pc["prnu_std"], 5), dsnu_std=round(pc["dsnu_std"], 3),
                     irregular_light=pc["irregular_light"])
        out.append(_write_frame(a, split, f"seq{sq:05d}_f{fi:02d}", imgs, anns, extra))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of frames (rounded up to whole sequences)")
    ap.add_argument("--out", default="wis_synth")
    ap.add_argument("--sequence-length", type=int, default=8,
                    help="consecutive frames per continuous stretch of web (1 = independent frames)")
    ap.add_argument("--width", type=int, default=8192); ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--shading-mult", type=float, default=1.0); ap.add_argument("--prnu-mult", type=float, default=1.0)
    ap.add_argument("--dsnu-mult", type=float, default=1.0)
    ap.add_argument("--defect-strength", type=float, default=1.0, help="multiplier on all defect strengths")
    ap.add_argument("--class-strength", nargs="*", help='per-class strength, e.g. "Impression line=1.5" bladeline=2')
    ap.add_argument("--class-weight", nargs="*", help='per-class sampling weight, e.g. pinhole=2 "Coating mark=0.5"')
    ap.add_argument("--defect-free", type=float, default=0.10, help="share of frames without defects")
    ap.add_argument("--edge-prob", type=float, default=0.25, help="share of stretches seen by an outermost camera (web edge visible)")
    ap.add_argument("--vis-drop", type=float, default=1.0, help="drop labels whose visibility score (CNR) is below this (invisible)")
    ap.add_argument("--vis-difficult", type=float, default=2.0, help="flag labels below this score as 'difficult' (kept)")
    ap.add_argument("--drop-difficult", action="store_true", help="also leave 'difficult' labels out of the YOLO files")
    ap.add_argument("--no-lookalikes", action="store_true", help="disable the unlabelled look-alikes (wire / couch / seam marks, cloudy patches)")
    ap.add_argument("--lighting", choices=["front", "back", "mixed"], default="front",   # front = your Basler setup
                    help="front = reflected light (your Basler camera); back = transmitted light (backlit: formation visible, "
                         "holes bright, surface defects faint); mixed = each stretch randomly front- or backlit")
    ap.add_argument("--back-share", type=float, default=0.5, help="share of backlit stretches in --lighting mixed")
    ap.add_argument("--defects-per-frame", type=int, nargs=2, default=[8, 14])
    ap.add_argument("--size-small", type=float, nargs=2, default=[1, 5], help="mm, longest side")
    ap.add_argument("--size-medium", type=float, nargs=2, default=[5, 20])
    ap.add_argument("--size-large", type=float, nargs=2, default=[20, 80])
    a = ap.parse_args()
    dcfg = dict(strength_mult=a.defect_strength, class_strength=_parse_kv(a.class_strength),
                class_weights=_parse_kv(a.class_weight), defect_free=a.defect_free, lighting=("front" if a.lighting == "mixed" else a.lighting), defects_per_frame=tuple(a.defects_per_frame),
                size_tiers={"small": tuple(a.size_small), "medium": tuple(a.size_medium), "large": tuple(a.size_large)})
    sys.path.insert(0, HERE)
    from layer4_defects import CLASSES
    splits = ["train", "val", "test"]
    for s in splits:
        for v in VERSIONS: os.makedirs(os.path.join(a.out, "images", v, s), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", s), exist_ok=True); os.makedirs(os.path.join(a.out, "masks", s), exist_ok=True)
    os.makedirs(os.path.join(a.out, "annotations"), exist_ok=True)
    n_jobs = int(np.ceil(a.n / a.sequence_length))                  # splits are made per sequence: no leakage
    fr = np.array(a.split) / sum(a.split); order = np.random.default_rng(a.seed).permutation(n_jobs)
    ntr, nva = int(round(fr[0] * n_jobs)), int(round(fr[1] * n_jobs))
    split_of = {int(i): ("train" if r < ntr else "val" if r < ntr + nva else "test") for r, i in enumerate(order)}
    # web-edge stretches: exactly round(edge_prob x stretches), at least one when edge_prob > 0
    n_edge = int(round(a.edge_prob * n_jobs)) if a.sequence_length > 1 else 0
    if a.sequence_length > 1 and a.edge_prob > 0: n_edge = max(1, n_edge)
    a.edge_sequences = set(int(i) for i in np.random.default_rng([a.seed, 99]).permutation(n_jobs)[:n_edge])
    n_back = max(1, int(round(a.back_share * n_jobs))) if a.lighting == "mixed" else 0      # backlit stretches, assigned exactly
    a.back_sequences = set(int(i) for i in np.random.default_rng([a.seed, 77]).permutation(n_jobs)[:n_back])
    fn = make_sequence if a.sequence_length > 1 else make_frame
    lcfg = dict(wire_prob=0, couch_prob=0, seam_prob=0, cloudy_prob=0) if a.no_lookalikes else {}
    with Pool(a.workers, initializer=_init, initargs=(dcfg, lcfg)) as pool:
        res = [fr_ for lst in pool.imap(fn, [(i, split_of[i], a) for i in range(n_jobs)]) for fr_ in lst]
    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0][0].keys())); w.writeheader(); w.writerows([r[0] for r in res])
    for s in splits:
        images, anns, aid = [], [], 1
        for i, (row, al, name) in enumerate(res):
            if row["split"] != s: continue
            images.append(dict(id=i + 1, file_name=name + ".png", width=a.width, height=a.height))
            for an in al:
                anns.append(dict(id=aid, image_id=i + 1, category_id=an["cls_id"] + 1, bbox=list(an["bbox"]), area=an["area"], iscrowd=0,
                                 attributes=dict(visibility=an["visibility"], difficult=an["difficult"]))); aid += 1
        json.dump(dict(images=images, annotations=anns, categories=[dict(id=i + 1, name=c) for i, c in enumerate(CLASSES)]),
                  open(os.path.join(a.out, "annotations", f"instances_{s}.json"), "w"))
    open(os.path.join(a.out, "classes.txt"), "w").write("\n".join(CLASSES) + "\n")
    with open(os.path.join(a.out, "dataset.yaml"), "w") as f:
        f.write(f"# one YOLO dataset per camera version: set 'images' to images/<version>\npath: {os.path.abspath(a.out)}\n"
                "train: images/combined/train\nval: images/combined/val\ntest: images/combined/test\nnames:\n" +
                "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"Done: {a.n} frames x {len(VERSIONS)} versions -> {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
