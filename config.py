# file config.py
IMAGE_PATH   = "test6.jpg"
MODEL_PATH   = "pose_landmarker.task"
KP_JSON      = "keypoints.json"
BG_REMOVED   = "bg_removed.png"
OUTPUT_IMAGE = "output_measurements.png"


# ── Keypoint indices (MediaPipe Pose) ───────────────────────────────────────
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24

WAIST_T          = 0.6
ROW_HALF_BAND    = 8
ARM_PAD_PX       = 10  
ARM_HORIZONTAL_THRESHOLD = 0.3  # nếu tay dang ngang quá thì không đo vòng tay
BODY_CENTER_SEARCH_HALF = 30

TORSO_CORRIDOR_MARGIN = 8