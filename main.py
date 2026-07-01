"""
main.py
"""

from pose_estimator       import detect_pose, draw_pose_overlay
from bg_remover           import remove_background, alpha_to_binary_mask
from body_measurements    import load_keypoints, estimate_measurements
from bodyshape_classifier import classify
from visualize            import draw_measurements

import json
import config

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
        shoulder_px  = measurements["shoulder_px"],
        waist_px     = measurements["waist_px"],
        hip_px       = measurements["hip_px"],
        high_hip_px  = measurements["high_hip_px"],
    )

    # 5. Visualize
    print("[5/5] Drawing measurements...")
    draw_measurements(BG_REMOVED, measurements, lines, OUTPUT_IMAGE, kp=kp, debug_arms=True)


    print()
    print("=" * 45)
    print(f"  Shoulder  : {measurements['shoulder_px']:.0f} px  (ratio {measurements['shoulder_ratio']:.3f})")
    print(f"  Waist     : {measurements['waist_px']:.0f} px  (ratio {measurements['waist_ratio']:.3f})")
    print(f"  High Hip  : {measurements['high_hip_px']:.0f} px  (ratio {measurements['high_hip_ratio']:.3f})")
    print(f"  Hip       : {measurements['hip_px']:.0f} px  (ratio {measurements['hip_ratio']:.3f})")
    print(f"  Body Shape: {shape}")
    print("=" * 45)
    print(f"\nOutput → {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()