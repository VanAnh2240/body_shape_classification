# body_measurements.py
"""
body_measurements.py
"""

import json
import numpy as np
import config

LEFT_SHOULDER   = config.LEFT_SHOULDER
RIGHT_SHOULDER  = config.RIGHT_SHOULDER
LEFT_ELBOW      = config.LEFT_ELBOW
RIGHT_ELBOW     = config.RIGHT_ELBOW
LEFT_WRIST      = config.LEFT_WRIST
RIGHT_WRIST     = config.RIGHT_WRIST
LEFT_HIP        = config.LEFT_HIP
RIGHT_HIP       = config.RIGHT_HIP

WAIST_T         = config.WAIST_T   # 0.6 vai và hông
HIGH_HIP_T      = 0.80             # high hip ≈ 80% giữa shoulder và hip

ARM_PAD_PX      = config.ARM_PAD_PX
TRACE_STEP          = 0.5
MAX_GAP_PX          = 4
ARM_EXCLUSION_DIST  = 18


def load_keypoints(path: str = "keypoints.json") -> dict:
    with open(path) as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def _kp_xy(kp: dict, idx: int) -> np.ndarray:
    p = kp[idx]
    return np.array([p["x_pixel"], p["y_pixel"]], dtype=float)

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _point_in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    H, W = mask.shape
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or xi >= W or yi < 0 or yi >= H:
        return False
    return bool(mask[yi, xi])


def _dist_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-9:
        return float(np.linalg.norm(p - a))
    t = np.clip(float(np.dot(p - a, ab)) / ab_len2, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _point_near_arm(
    px: float, py: float,
    arm_pts: list,
    threshold: float = ARM_EXCLUSION_DIST,
) -> bool:
    if not arm_pts:
        return False
    p = np.array([px, py])
    for i in range(len(arm_pts) - 1):
        a = arm_pts[i]
        b = arm_pts[i + 1]
        if min(a[1], b[1]) - threshold > py or max(a[1], b[1]) + threshold < py:
            continue
        if _dist_point_to_segment(p, a, b) < threshold:
            return True
    return False

def _trace_to_edge(
    mask: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    left_arm_pts: list,
    right_arm_pts: list,
    max_gap: int = MAX_GAP_PX,
) -> np.ndarray:
    H, W = mask.shape
    d = _unit(direction)

    last_body_pt = start.copy()
    gap_count    = 0.0
    t            = 0.0

    while True:
        t += TRACE_STEP
        cx = start[0] + d[0] * t
        cy = start[1] + d[1] * t

        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            break

        in_mask = _point_in_mask(mask, cx, cy)
        in_arm  = (_point_near_arm(cx, cy, left_arm_pts) or
                   _point_near_arm(cx, cy, right_arm_pts))

        if in_mask:
            gap_count = 0
            if not in_arm:
                last_body_pt = np.array([cx, cy])
        else:
            if not in_arm:
                gap_count += TRACE_STEP
                if gap_count > max_gap:
                    break
            else:
                gap_count += TRACE_STEP * 0.3
                if gap_count > max_gap * 3:
                    break

    return last_body_pt


def _measure_oriented_line(
    mask: np.ndarray,
    kp_left: np.ndarray,
    kp_right: np.ndarray,
    left_arm_pts: list,
    right_arm_pts: list,
    label: str = "",
) -> dict:
    direction = _unit(kp_right - kp_left)
    mid       = (kp_left + kp_right) / 2.0

    p_right = _trace_to_edge(mask, mid,  direction, left_arm_pts, right_arm_pts)
    p_left  = _trace_to_edge(mask, mid, -direction, left_arm_pts, right_arm_pts)

    width_px = float(np.linalg.norm(p_right - p_left))

    return {
        "p0":        p_left,
        "p1":        p_right,
        "width_px":  width_px,
        "midpoint":  (p_left + p_right) / 2.0,
        "angle_deg": float(np.degrees(np.arctan2(direction[1], direction[0]))),
    }

def _is_arm_horizontal(arm_pts: list) -> bool:
    if len(arm_pts) < 2:
        return False
    shoulder, wrist = arm_pts[0], arm_pts[-1]
    dx = abs(wrist[0] - shoulder[0])
    dy = abs(wrist[1] - shoulder[1])
    if dx < 1:
        return False
    return (dy / dx) < config.ARM_HORIZONTAL_THRESHOLD


def estimate_measurements(kp: dict, mask: np.ndarray) -> tuple[dict, dict]:
    ls = _kp_xy(kp, LEFT_SHOULDER)
    rs = _kp_xy(kp, RIGHT_SHOULDER)
    lh = _kp_xy(kp, LEFT_HIP)
    rh = _kp_xy(kp, RIGHT_HIP)
    le = _kp_xy(kp, LEFT_ELBOW)
    re = _kp_xy(kp, RIGHT_ELBOW)
    lw = _kp_xy(kp, LEFT_WRIST)
    rw = _kp_xy(kp, RIGHT_WRIST)

    left_arm_pts  = [ls, le, lw]
    right_arm_pts = [rs, re, rw]

    left_arm_horizontal  = _is_arm_horizontal(left_arm_pts)
    right_arm_horizontal = _is_arm_horizontal(right_arm_pts)

    eff_left_arm  = left_arm_pts  if not left_arm_horizontal else []
    eff_right_arm = right_arm_pts if not right_arm_horizontal else []

    lw_kp  = ls + WAIST_T    * (lh - ls)   # waist trái   (t=0.60)
    rw_kp  = rs + WAIST_T    * (rh - rs)   # waist phải
    lhh_kp = ls + HIGH_HIP_T * (lh - ls)   # high_hip trái (t=0.80)
    rhh_kp = rs + HIGH_HIP_T * (rh - rs)   # high_hip phải

    shoulder_info  = _measure_oriented_line(mask, ls,     rs,     eff_left_arm, eff_right_arm, "shoulder")
    waist_info     = _measure_oriented_line(mask, lw_kp,  rw_kp,  eff_left_arm, eff_right_arm, "waist")
    high_hip_info  = _measure_oriented_line(mask, lhh_kp, rhh_kp, eff_left_arm, eff_right_arm, "high_hip")
    hip_info       = _measure_oriented_line(mask, lh,     rh,     eff_left_arm, eff_right_arm, "hip")

    ref = shoulder_info["width_px"] if shoulder_info["width_px"] > 0 else 1.0
    measurements = {
        "shoulder_px":    round(shoulder_info["width_px"],  1),
        "waist_px":       round(waist_info["width_px"],     1),
        "high_hip_px":    round(high_hip_info["width_px"],  1),
        "hip_px":         round(hip_info["width_px"],       1),

        "shoulder_ratio": round(shoulder_info["width_px"] / ref, 3),
        "waist_ratio":    round(waist_info["width_px"]    / ref, 3),
        "high_hip_ratio": round(high_hip_info["width_px"] / ref, 3),
        "hip_ratio":      round(hip_info["width_px"]      / ref, 3),

        "shoulder_angle": round(shoulder_info["angle_deg"],  1),
        "waist_angle":    round(waist_info["angle_deg"],     1),
        "high_hip_angle": round(high_hip_info["angle_deg"],  1),
        "hip_angle":      round(hip_info["angle_deg"],       1),
    }

    lines = {
        "shoulder": shoulder_info,
        "waist":    waist_info,
        "high_hip": high_hip_info,
        "hip":      hip_info,
    }
    return measurements, lines
