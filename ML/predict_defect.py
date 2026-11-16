"""
ISRA Blob-Ready Defect Classifier — Inference Script
======================================================
Trained on ISRA CSV data using ONLY features computable from a plain,
uncalibrated blob image (no physical size, no camera metadata).

Model: Random Forest, 300 trees
Features (in this exact order): aspect_ratio, PERC_BRIGHT, PERC_DARK, A_to_roi_ratio
Test accuracy: 70.8% (unthresholded), 82.5% at recommended 0.5 confidence threshold
Trained on: ISRA CSV package, files 1238-1800, 18-group consolidated taxonomy

USAGE
-----
    from predict_defect import BlobDefectClassifier

    clf = BlobDefectClassifier("blob_model_v1.pkl")

    # From raw blob measurements:
    result = clf.predict_from_blob(
        blob_width_px=42,
        blob_height_px=118,
        blob_pixel_count=3210,
        blob_mean_intensity=180,
        background_mean_intensity=110,
    )
    print(result)
    # {'group': 'Streak/Scratch (family)', 'confidence': 0.71, 'accept': True}
"""

import pickle
import numpy as np


class BlobDefectClassifier:
    def __init__(self, model_path="blob_model_v1.pkl", confidence_threshold=None):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.classes = bundle["classes"]
        self.test_accuracy = bundle["test_accuracy"]
        self.threshold = confidence_threshold or bundle["recommended_confidence_threshold"]
        self.threshold_table = bundle["threshold_table"]

    # ------------------------------------------------------------------
    # Step-by-step feature extraction from raw blob measurements
    # ------------------------------------------------------------------
    @staticmethod
    def compute_features(
        blob_width_px: float,
        blob_height_px: float,
        blob_pixel_count: int,
        blob_mean_intensity: float,
        background_mean_intensity: float,
        intensity_std: float = None,
    ) -> dict:
        """
        Turns raw measurements from a segmented blob image into the 4
        model-ready features. This mirrors the ISRA CSV columns:
            aspect_ratio    ~ SizeMD / SizeCD
            PERC_BRIGHT     ~ proportion of blob brighter than background
            PERC_DARK       ~ proportion of blob darker than background
            A_to_roi_ratio  ~ solidity (blob area / bounding box area)

        blob_width_px / blob_height_px : bounding box dimensions, in pixels
        blob_pixel_count               : number of pixels in the segmented blob mask
        blob_mean_intensity            : average pixel brightness of the blob
        background_mean_intensity      : average pixel brightness of the surrounding
                                          material (NOT the blob) — required to compute
                                          the bright/dark proportions
        intensity_std                  : optional. If you have per-pixel intensities
                                          for the blob, a more faithful PERC_BRIGHT/DARK
                                          can be computed directly (see
                                          `compute_features_from_pixels` below).
                                          If omitted, a simple threshold rule is used.
        """
        if blob_width_px <= 0 or blob_height_px <= 0:
            raise ValueError("blob_width_px and blob_height_px must be positive")

        aspect_ratio = blob_height_px / blob_width_px
        roi_area = blob_width_px * blob_height_px
        a_to_roi_ratio = blob_pixel_count / roi_area

        # Simple bright/dark proportion from mean-intensity comparison.
        # This is a coarse approximation of ISRA's per-pixel PERC_BRIGHT/PERC_DARK.
        # For higher fidelity, use compute_features_from_pixels() instead.
        delta = blob_mean_intensity - background_mean_intensity
        # Normalize delta into a 0-1 "how bright/dark" proportion using a soft scale.
        # 50 intensity levels (out of 255) is treated as a strong, unambiguous signal.
        scale = 50.0
        if delta > 0:
            perc_bright = min(1.0, delta / scale)
            perc_dark = 0.0
        elif delta < 0:
            perc_dark = min(1.0, -delta / scale)
            perc_bright = 0.0
        else:
            perc_bright = perc_dark = 0.0

        return {
            "aspect_ratio": aspect_ratio,
            "PERC_BRIGHT": perc_bright,
            "PERC_DARK": perc_dark,
            "A_to_roi_ratio": a_to_roi_ratio,
        }

    @staticmethod
    def compute_features_from_pixels(
        blob_width_px: float,
        blob_height_px: float,
        blob_pixel_values: np.ndarray,
        background_pixel_values: np.ndarray,
    ) -> dict:
        """
        Higher-fidelity version: pass in the actual pixel intensity arrays
        for the blob and for a sampled patch of background, and this computes
        PERC_BRIGHT / PERC_DARK the same way ISRA's own pipeline conceptually
        does — the proportion of blob pixels that fall outside the background's
        normal brightness range.

        blob_pixel_values       : 1D array of intensities for every pixel in the blob mask
        background_pixel_values : 1D array of intensities for a background sample
        """
        bg_mean = background_pixel_values.mean()
        bg_std = background_pixel_values.std() + 1e-6

        bright_mask = blob_pixel_values > (bg_mean + bg_std)
        dark_mask = blob_pixel_values < (bg_mean - bg_std)

        perc_bright = bright_mask.mean()
        perc_dark = dark_mask.mean()

        roi_area = blob_width_px * blob_height_px
        a_to_roi_ratio = len(blob_pixel_values) / roi_area
        aspect_ratio = blob_height_px / blob_width_px

        return {
            "aspect_ratio": aspect_ratio,
            "PERC_BRIGHT": perc_bright,
            "PERC_DARK": perc_dark,
            "A_to_roi_ratio": a_to_roi_ratio,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: dict) -> dict:
        """Predict from an already-computed feature dict."""
        import pandas as pd
        x = pd.DataFrame([[features[c] for c in self.feature_names]], columns=self.feature_names)
        proba = self.model.predict_proba(x)[0]
        top_idx = np.argmax(proba)
        confidence = proba[top_idx]
        group = self.model.classes_[top_idx]

        # Full ranked list, useful for manual review queues
        ranked = sorted(zip(self.model.classes_, proba), key=lambda t: -t[1])

        return {
            "group": group,
            "confidence": round(float(confidence), 4),
            "accept": bool(confidence >= self.threshold),
            "threshold_used": self.threshold,
            "top_3": [(g, round(float(p), 4)) for g, p in ranked[:3]],
        }

    def predict_from_blob(self, blob_width_px, blob_height_px, blob_pixel_count,
                           blob_mean_intensity, background_mean_intensity):
        """Convenience: raw measurements straight to prediction."""
        features = self.compute_features(
            blob_width_px, blob_height_px, blob_pixel_count,
            blob_mean_intensity, background_mean_intensity,
        )
        result = self.predict(features)
        result["features_used"] = features
        return result

    def show_threshold_table(self):
        print(f"{'threshold':>10} {'coverage':>10} {'accuracy':>10}")
        for t, cov, acc in self.threshold_table:
            print(f"{t:>10.1f} {cov:>10.1%} {acc:>10.1%}")


if __name__ == "__main__":
    clf = BlobDefectClassifier("blob_model_v1.pkl")

    print("Model loaded.")
    print("Classes:", clf.classes)
    print(f"Overall test accuracy (no threshold): {clf.test_accuracy:.1%}\n")

    print("Confidence threshold / coverage tradeoff:")
    clf.show_threshold_table()

    print("\n--- Example prediction ---")
    example = clf.predict_from_blob(
        blob_width_px=12,
        blob_height_px=42,
        blob_pixel_count=310,
        blob_mean_intensity=95,
        background_mean_intensity=140,
    )
    print(example)
