"""
main.py  —  Pipeline đo body shape
=====================================================
Bước 1: pose_estimator.py  → keypoints.json
Bước 2: bg_remover.py      → mask từ ảnh đã remove bg
Bước 3: body_measurements  → tỉ lệ shoulder/waist/hip
Bước 4: classify           → body shape
Bước 5: visualize          → output_measurements.png
"""

from pose_estimator    import detect_pose, draw_pose_overlay
from bg_remover        import remove_background, alpha_to_binary_mask
from body_measurements import load_keypoints, estimate_measurements
from bodyshape_classifier import classify
from visualize         import draw_measurements

import json
import config

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH   = config.IMAGE_PATH
MODEL_PATH   = config.MODEL_PATH
KP_JSON      = config.KP_JSON
BG_REMOVED   = config.BG_REMOVED
OUTPUT_IMAGE = config.OUTPUT_IMAGE


def main():
    # 1. Detect pose
    print("[1/5] Detecting pose...")
    keypoints, result, mp_image = detect_pose(IMAGE_PATH, MODEL_PATH)
    with open(KP_JSON, "w") as f:
        json.dump(keypoints, f, indent=2)
    draw_pose_overlay(mp_image, result, "output_pose.png")

    # 2. Remove background
    print("[2/5] Removing background...")
    bgra = remove_background(IMAGE_PATH, output_path=BG_REMOVED)
    mask = alpha_to_binary_mask(bgra)

    # 3. Estimate measurements
    print("[3/5] Estimating measurements...")
    kp = load_keypoints(KP_JSON)
    measurements, lines = estimate_measurements(kp, mask)

    # 4. Classify body shape
    print("[4/5] Classifying body shape...")
    shape = classify(
        measurements["shoulder_px"],
        measurements["waist_px"],
        measurements["hip_px"],
    )

    # 5. Visualize
    print("[5/5] Drawing measurements...")
    draw_measurements(BG_REMOVED, measurements, lines, OUTPUT_IMAGE, kp=kp, debug_arms=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 40)
    print(f"  shoulder  ratio : {measurements['shoulder_ratio']:.3f}  ({measurements['shoulder_px']:.0f} px)")
    print(f"  Waist ratio : {measurements['waist_ratio']:.3f}  ({measurements['waist_px']:.0f} px)")
    print(f"  Hip   ratio : {measurements['hip_ratio']:.3f}  ({measurements['hip_px']:.0f} px)")
    print(f"  Body shape  : {shape}")
    print("=" * 40)
    print(f"\nOutput → {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()