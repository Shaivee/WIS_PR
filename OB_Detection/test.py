import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from worker import predict_full, read_labels

WEIGHTS = "runs/wis-2/weights/best.pt"
IMAGES = Path("/home/shaivee/PycharmProjects/WIS_RuleClassification/Synthetic_data/wis_synth/images/combined/train")
LABELS = Path("/home/shaivee/PycharmProjects/WIS_RuleClassification/Synthetic_data/wis_synth/labels/train")
PATTERN = "seq00005_*.png"  # the held-out validation sequence
OUT = Path("runs/wis_test")
CONF = 0.20  # threshold for the JSON and the drawn boxes

model = YOLO(WEIGHTS)
OUT.mkdir(parents=True, exist_ok=True)
output, gt, preds = [], {"images": [], "annotations": [], "categories": [{"id": 0}]}, []

images = sorted(IMAGES.glob(PATTERN))
assert images, f"no images matching '{PATTERN}' in {IMAGES.resolve()} -- check IMAGES / PATTERN"

for img_id, img_path in enumerate(images, 1):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    boxes, scores = predict_full(model, img, conf=0.001)
    real = read_labels(LABELS / f"{img_path.stem}.txt", w, h)[:, 1:]

    # JSON output
    output.append({"image": img_path.name, "width": w, "height": h,
                   "detections": [{"bbox": [round(float(v), 1) for v in b], "score": round(float(s), 3)}
                                  for b, s in zip(boxes, scores) if s >= CONF]})

    # top: real annotations (green), bottom: predictions (red)
    top, bottom = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in real:
        cv2.rectangle(top, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        if s >= CONF:
            cv2.rectangle(bottom, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(bottom, f"{s:.2f}", (int(x1), max(int(y1) - 5, 20)), 0, 0.8, (0, 0, 255), 2)
    cv2.putText(top, f"REAL ({len(real)})", (20, 50), 0, 1.5, (0, 255, 0), 3)
    cv2.putText(bottom, f"PREDICTED ({int((scores >= CONF).sum())})", (20, 50), 0, 1.5, (0, 0, 255), 3)
    cv2.imwrite(str(OUT / f"{img_path.stem}.jpg"), np.vstack([top, np.full((20, w, 3), 255, np.uint8), bottom]))

    # for mAP
    gt["images"].append({"id": img_id, "width": w, "height": h})
    for x1, y1, x2, y2 in real:
        gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": img_id, "category_id": 0,
                                  "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    preds += [{"image_id": img_id, "category_id": 0, "score": float(s),
               "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]} for (x1, y1, x2, y2), s in zip(boxes, scores)]

(OUT / "predictions.json").write_text(json.dumps(output, indent=2))

print(f"{len(images)} images, {len(gt['annotations'])} real defects, {len(preds)} predictions -> {OUT}")
assert preds, "the model made no predictions -- check WEIGHTS points to a trained best.pt"

coco_gt = COCO()
coco_gt.dataset = gt
coco_gt.createIndex()
evaluator = COCOeval(coco_gt, coco_gt.loadRes(preds), "bbox")
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()