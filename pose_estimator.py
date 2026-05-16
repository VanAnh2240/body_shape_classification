"""
pose_estimator.py
Detect pose landmarks và lưu keypoints JSON.
Không dùng segmentation mask.
"""

import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import config

mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
mp_pose    = mp.solutions.pose


def detect_pose(image_path: str, model_path: str) -> tuple[list, int, int]:
    """Chạy PoseLandmarker, trả về (keypoints_list, height, width)."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,   # không cần mask nữa
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    image  = mp.Image.create_from_file(image_path)
    result = detector.detect(image)

    if not result.pose_landmarks:
        raise RuntimeError("Không phát hiện người trong ảnh.")

    h, w = image.height, image.width
    keypoints = [
        {
            "id":         i,
            "x_pixel":    int(lm.x * w),
            "y_pixel":    int(lm.y * h),
            "visibility": float(lm.visibility),
        }
        for i, lm in enumerate(result.pose_landmarks[0])
    ]
    return keypoints, result, image


def draw_pose_overlay(image: mp.Image, result, output_path: str):
    """Vẽ skeleton lên ảnh gốc và lưu."""
    annotated = cv2.cvtColor(image.numpy_view().copy(), cv2.COLOR_RGB2BGR)

    for pose_landmarks in result.pose_landmarks:
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for lm in pose_landmarks:
            lm_pb = landmark_list.landmark.add()
            lm_pb.x, lm_pb.y, lm_pb.z = lm.x, lm.y, lm.z
            lm_pb.visibility = lm.visibility

        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=landmark_list,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
        )

    cv2.imwrite(output_path, annotated)


if __name__ == "__main__":
    keypoints, result, image = detect_pose(config.IMAGE_PATH, config.MODEL_PATH)

    with open(config.KP_JSON, "w") as f:
        json.dump(keypoints, f, indent=2)

    draw_pose_overlay(image, result, config.OUTPUT_IMAGE)

    print(f"keypoints → {config.KP_JSON}")
    print(f"overlay   → {config.OUTPUT_IMAGE}")