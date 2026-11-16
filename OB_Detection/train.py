from pathlib import Path

from ultralytics import YOLO

from worker import prepare_dataset

DATA_ROOT = Path("/home/shaivee/PycharmProjects/WIS_RuleClassification/Synthetic_data/wis_synth")
IMAGES = DATA_ROOT / "images/combined"
LABELS = DATA_ROOT / "labels"
TILES = Path("wis_tiles")
VAL_SEQS = {5}  # whole sequences held out for validation

data_yaml = prepare_dataset(IMAGES, LABELS, TILES, VAL_SEQS)

model = YOLO("yolo26s.pt")
model.train(
    data=str(data_yaml), epochs=150, imgsz=1024, batch=8, patience=10,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.35,  # grayscale images
    flipud=0.5, degrees=0.0, scale=0.3,  # small defects, axis-aligned line-scan
    project=str(Path("runs").resolve()), name="wis",
)
