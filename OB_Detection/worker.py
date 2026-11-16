import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms

TILE, OVERLAP = 1024, 256


def tile_starts(length):
    stride = TILE - OVERLAP
    starts = list(range(0, max(length - TILE, 0) + 1, stride))
    if starts[-1] + TILE < length:
        starts.append(length - TILE)
    return starts


def read_labels(txt, w, h):
    """YOLO txt -> [cls, x1, y1, x2, y2] in pixels."""
    if not txt.exists() or txt.stat().st_size == 0:
        return np.zeros((0, 5))
    c, xc, yc, bw, bh = np.loadtxt(txt, ndmin=2).T
    return np.stack([c, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h], 1)


def write_labels(txt, boxes, w, h):
    lines = [f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
             for c, x1, y1, x2, y2 in boxes]
    txt.write_text("\n".join(lines))


def clip_to_tile(boxes, x0, y0, min_visible=0.3):
    c, x1, y1, x2, y2 = boxes.T
    cx1, cx2 = np.clip(x1 - x0, 0, TILE), np.clip(x2 - x0, 0, TILE)
    cy1, cy2 = np.clip(y1 - y0, 0, TILE), np.clip(y2 - y0, 0, TILE)
    visible = (cx2 - cx1) * (cy2 - cy1) / ((x2 - x1) * (y2 - y1))
    keep = (visible >= min_visible) | (cx2 - cx1 >= TILE / 2) | (cy2 - cy1 >= TILE / 2)
    keep &= (cx2 > cx1) & (cy2 > cy1)
    return np.stack([c, cx1, cy1, cx2, cy2], 1)[keep]


def prepare_dataset(images_dir, labels_dir, out_dir, val_seqs):
    """Cut full frames into tiles, split train/val by sequence, write data.yaml (one class: defect)."""
    out_dir = Path(out_dir)
    for img_path in sorted(Path(images_dir).rglob("*.png")):
        seq = int(re.search(r"seq(\d+)", img_path.stem).group(1))
        split = "val" if seq in val_seqs else "train"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        boxes = read_labels(next(Path(labels_dir).rglob(f"{img_path.stem}.txt"), Path("none")), w, h)
        boxes[:, 0] = 0  # detection only: every defect is class 0
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for y0 in tile_starts(h):
            for x0 in tile_starts(w):
                name = f"{img_path.stem}_{x0}_{y0}"
                cv2.imwrite(str(out_dir / "images" / split / f"{name}.png"), img[y0:y0 + TILE, x0:x0 + TILE])
                write_labels(out_dir / "labels" / split / f"{name}.txt", clip_to_tile(boxes, x0, y0), TILE, TILE)
    yaml = out_dir / "data.yaml"
    yaml.write_text(f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames: [defect]\n")
    return yaml


def predict_full(model, img, conf=0.25, iou=0.5):
    """Tiled inference on a full frame -> boxes (x1,y1,x2,y2), scores."""
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    coords = [(x0, y0) for y0 in tile_starts(img.shape[0]) for x0 in tile_starts(img.shape[1])]
    results = model.predict([img[y0:y0 + TILE, x0:x0 + TILE] for x0, y0 in coords],
                            imgsz=TILE, conf=conf, verbose=False)
    boxes = torch.cat([r.boxes.xyxy.cpu() + torch.tensor([x0, y0, x0, y0]) for (x0, y0), r in zip(coords, results)])
    scores = torch.cat([r.boxes.conf.cpu() for r in results])
    keep = nms(boxes, scores, iou)
    return boxes[keep].numpy(), scores[keep].numpy()
