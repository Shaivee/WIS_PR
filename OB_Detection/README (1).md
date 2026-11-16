# WIS Defect Detection (YOLO)

Detects defects on synthetic paper-web (WIS) line-scan images using YOLO.

- **Task:** detection only. All 33 defect types are trained as one class, `defect`. Classification can be added later.
- **Input:** whole grayscale images (8192 × 1024).
- **Output:** a JSON file with bounding boxes and scores, plus comparison images (real annotations vs. predictions).
- **Hardware:** runs on CPU. No GPU is needed.

---

## Files

| File | Purpose | Run it? |
|---|---|---|
| `worker.py` | Shared functions: tiling, label conversion, whole-image prediction | No (imported by the other two) |
| `train.py` | Builds the tiled dataset and trains the model | Yes, step 1 |
| `test.py` | Predicts on whole images, writes JSON and comparison images, prints mAP | Yes, step 2 |
| `requirements.txt` | Python packages | Used once at setup |

---

## Setup

```bash
pip install -r requirements.txt
```

Expected dataset layout (YOLO txt labels, one `.txt` per image with the same name):

```
wis_synth/
├── images/combined/{train,val,test}/seqXXXXX_fYY.png
└── labels/{train,val,test}/seqXXXXX_fYY.txt
```

Set the paths at the top of `train.py` and `test.py`. Absolute paths are safest.

---

## How to run

```bash
python train.py     # 1. tiles the data and trains -> runs/wis/weights/best.pt
python test.py      # 2. predicts on whole images  -> runs/wis_test/
```

Delete `wis_tiles/` before running `train.py` again, especially after changing `VAL_SEQS` or the tile settings. Otherwise old tiles stay in the dataset.

---

## How it works

**Training on tiles.** Defects can be as small as 3 px, so images are never shrunk. Each 8192 × 1024 image is cut into overlapping 1024 × 1024 tiles at full resolution, and the boxes are clipped to each tile.

**Validation split by sequence.** Frames from the same sequence share lighting and paper settings. A whole sequence is held out for validation, so validation images are truly unseen.

**Testing on whole images.** YOLO is fully convolutional, so a model trained on 1024 tiles can predict on the full 8192 × 1024 image in a single pass, with defects at the same pixel scale as in training.

---

## Parameters

### Tiling (`worker.py`): affects training data

| Parameter | Default | Role |
|---|---|---|
| `TILE` | `1024` | Tile size in pixels. Must equal `imgsz` in `train.py`. Smaller tiles train faster but cut more large defects. |
| `OVERLAP` | `256` | Overlap between neighbouring tiles. Defects up to this width always appear whole in at least one tile. Increase it if large defects get cut too often. |
| `min_visible` (in `clip_to_tile`) | `0.3` | A defect cut by a tile edge is kept only if at least 30% of its area is inside the tile (or the visible part spans half the tile). Lower = more partial defects labelled. |

### Training (`train.py`)

**Data and split**

| Parameter | Default | Role |
|---|---|---|
| `DATA_ROOT`, `IMAGES`, `LABELS` | `wis_synth/...` | Where the full images and YOLO labels are. |
| `TILES` | `wis_tiles` | Where the tiled dataset is written. |
| `VAL_SEQS` | `{5}` | Sequence numbers held out for validation, e.g. `{5}` or `{3, 5}`. Keep `PATTERN` in `test.py` in sync. |

**Model and schedule**

| Parameter | Default | Role |
|---|---|---|
| model | `yolo26n.pt` | Model size: `n` < `s` < `m` < `l` < `x`. Bigger is more accurate but slower. `n` is chosen for CPU. |
| `epochs` | `60` | Passes over the training data. More epochs usually help, but take longer on CPU. |
| `patience` | `20` | Stops early if validation mAP hasn't improved for this many epochs. |
| `imgsz` | `1024` | Training input size. Keep it equal to `TILE`. Lower values shrink the image, and tiny defects disappear. |
| `batch` | `4` | Tiles per step. Lower it if you run out of memory; raise it if you have spare RAM. |
| `device` | `"cpu"` | `"cpu"`, `"0"` for the first GPU, `"mps"` for Apple silicon. |
| `cache` | `"ram"` | Keeps tiles in memory for faster epochs. Use `"disk"` with less than ~8 GB RAM, or `False` to disable. |

**Augmentation** (random changes to training tiles so the model generalises)

| Parameter | Default | Role |
|---|---|---|
| `hsv_h`, `hsv_s` | `0.0` | Hue and saturation changes, turned off because the images are grayscale. |
| `hsv_v` | `0.35` | Random brightness change (±35%). Simulates lighting and shading differences. |
| `flipud` | `0.5` | Vertical flip probability. (Horizontal flip `fliplr` is 0.5 by default.) |
| `degrees` | `0.0` | Rotation, turned off because line-scan images are axis-aligned. |
| `scale` | `0.3` | Random zoom (±30%). Kept mild, because strong zoom-out erases tiny defects. |
| `mosaic` | `1.0` (default) | Combines 4 tiles into one training image. Strong help with small datasets. |

All other [Ultralytics training settings](https://docs.ultralytics.com/modes/train/#train-settings) can be added to `model.train(...)`.

### Testing (`test.py`): affects results

| Parameter | Default | Role |
|---|---|---|
| `WEIGHTS` | `runs/wis/weights/best.pt` | The trained model to use. |
| `IMAGES`, `LABELS` | `wis_synth/.../train` | Folder of whole images to predict on, and their labels (used only for comparison and mAP). |
| `PATTERN` | `seq00005_*.png` | Which images to process. `*.png` = all images in `IMAGES`. |
| `OUT` | `runs/wis_test` | Output folder. |
| `CONF` | `0.25` | **Main tuning knob.** Minimum score for a box to go into the JSON and be drawn. Lower = more defects found but more false alarms; higher = fewer false alarms but more misses. |

Inside `predict_full` (`worker.py`):

| Parameter | Default | Role |
|---|---|---|
| `imgsz=[h, w]` | full image size | The image is processed at its native resolution, in one pass. |
| `max_det` | `1000` | Maximum boxes per image. |
| `conf` | `0.001` when called from `test.py` | Very low on purpose: mAP needs all candidate boxes. `CONF` filters them afterwards for the JSON and images. |

---

## Outputs

### Training (`runs/wis/`)

- `weights/best.pt`: the best model (by validation mAP). Use this one.
- `weights/last.pt`: the model from the last epoch.
- `results.png`, `results.csv`: loss and mAP per epoch.
- Validation plots (PR curve, sample predictions).

### Testing (`runs/wis_test/`)

**`predictions.json`.** Boxes are `[x1, y1, x2, y2]` in pixels of the original image:

```json
[
  {
    "image": "seq00005_f00.png",
    "width": 8192,
    "height": 1024,
    "detections": [
      {"bbox": [6049.8, 203.5, 6149.3, 506.0], "score": 0.833}
    ]
  }
]
```

**`<image>.jpg`.** One per image:
- **Top:** original image with **real annotations** (green) and their count.
- **Bottom:** same image with **predicted boxes** (red), with scores and their count.

**Terminal.** COCO metrics on whole images. The key lines:
- `AP @ IoU=0.50`: mAP50, how well predicted boxes match real ones (0 to 1, higher is better).
- `AR @ IoU=0.50:0.95 maxDets=100`: recall, the share of real defects found.

---

## Predicting on new images

No retraining needed. In `test.py`, set `IMAGES` to the new folder and `PATTERN = "*.png"`, then run `python test.py`. If the images have no labels, the green half stays empty and the mAP is meaningless; use the JSON and the red boxes.

---

## Known limitations and next steps

- **Faint defects are missed most.** Recall drops sharply for defects with low `visibility` (see the COCO annotations).
- **Small dataset.** About 30 labelled training frames. Generating more synthetic sequences (varied lighting, paper, low-visibility defects) is the biggest improvement available.
- **CPU training is slow.** A free cloud GPU (Google Colab, Kaggle) is much faster; set `device="0"` there.
- **Possible upgrades:** background flattening and CLAHE contrast enhancement before training and testing; a larger model (`yolo26s`, `yolo26m`); `yolo26n-p2.yaml` for very small defects; adding classification (33 classes) later.
