from pathlib import Path

from blob_mask_classifier import MaskBlobClassifier

HERE = Path(__file__).parent
BLOB_PATH = "/home/shaivee/PycharmProjects/WIS/DATA/defects_shubham/defect_crops/grey-spot/1_png.rf.b9351c5e9cff00f1272392034ac548ac_9.png"    # change to your blob image
MASK_PATH = "/home/shaivee/PycharmProjects/WIS/DATA/defects_shubham/masks/grey-spot/1_png.rf.b9351c5e9cff00f1272392034ac548ac_9_mask.png"   # change to its mask

print(MaskBlobClassifier().classify(BLOB_PATH, MASK_PATH))
