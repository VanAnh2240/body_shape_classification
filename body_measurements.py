# body_measurements.py
"""
Cải tiến v2: hỗ trợ người nghiêng / xoay.

Chiến lược cốt lõi:
  - Không scan hàng ngang (horizontal row) cố định.
  - Kẻ đường thẳng qua 2 keypoint (LS→RS, LH→RH, LWAIST→RWAIST ước tính)
    rồi EXTEND ra 2 phía cho đến khi thoát khỏi foreground mask.
  - Đường này có thể nghiêng tùy ý → đúng với mọi pose nghiêng/xoay.
  - Vẫn giữ arm-exclusion: với mỗi điểm trên đường scan, kiểm tra xem
    pixel đó có nằm trong vùng skeleton tay không → bỏ qua.

Thuật toán extend_line_to_mask_edge():
  1. Bắt đầu từ midpoint của 2 keypoint.
  2. Đi theo hướng perpendicular (vuông góc với đường nối 2 kp)?
     → Không. Đi theo hướng CỦA đường nối 2 kp (extend trái/phải).
  3. Dừng khi pixel mask = 0 (ngoài thân).
  4. Nếu đang trong vùng tay (arm zone) → skip, tiếp tục để tìm biên thân.
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

WAIST_T                 = config.WAIST_T
ARM_PAD_PX              = config.ARM_PAD_PX

# ── Tuning params ─────────────────────────────────────────────────────────
# Số pixel step khi trace dọc theo đường (sub-pixel precision)
TRACE_STEP          = 0.5      # px
# Khi trace ra ngoài biên mask, cho phép gap nhỏ để vượt qua nhiễu
MAX_GAP_PX          = 4        # px — nếu gap > này thì dừng
# Arm exclusion: nếu điểm trace nằm trong cylinder tay (khoảng cách < threshold)
ARM_EXCLUSION_DIST  = 18       # px — khoảng cách vuông góc từ skeleton tay
# Waist: nội suy giữa shoulder và hip
# (WAIST_T đã có trong config, thường = 0.45)


def load_keypoints(path: str = "keypoints.json") -> dict:
    with open(path) as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def _kp_xy(kp: dict, idx: int) -> np.ndarray:
    p = kp[idx]
    return np.array([p["x_pixel"], p["y_pixel"]], dtype=float)


# ── Geometry helpers ───────────────────────────────────────────────────────

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
    """Khoảng cách từ p đến đoạn thẳng a-b."""
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
    """True nếu điểm (px,py) nằm trong cylinder bao quanh skeleton tay."""
    if not arm_pts:
        return False
    p = np.array([px, py])
    for i in range(len(arm_pts) - 1):
        a = arm_pts[i]
        b = arm_pts[i + 1]
        # Chỉ check đoạn nằm trong vùng Y của điểm ±threshold
        if min(a[1], b[1]) - threshold > py or max(a[1], b[1]) + threshold < py:
            continue
        d = _dist_point_to_segment(p, a, b)
        if d < threshold:
            return True
    return False


# ── Core: trace đường qua 2 keypoint đến biên mask ────────────────────────

def _trace_to_edge(
    mask: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,        # unit vector, đi ra phía cần tìm biên
    left_arm_pts: list,
    right_arm_pts: list,
    max_gap: int = MAX_GAP_PX,
) -> np.ndarray:
    """
    Bắt đầu từ `start`, bước theo `direction` (TRACE_STEP px mỗi lần).
    Trả về điểm cuối cùng còn nằm trong mask (sau khi bỏ qua vùng tay).

    Logic arm exclusion:
      - Nếu điểm hiện tại nằm trong vùng tay → KHÔNG coi là biên thân,
        nhưng vẫn tiếp tục đi (để tìm gap phía sau tay).
      - Khi gặp gap (pixel=0) nằm ngoài vùng tay → dừng, trả về điểm
        foreground cuối.
    """
    H, W = mask.shape
    d = _unit(direction)

    last_body_pt   = start.copy()   # điểm foreground cuối KHÔNG phải tay
    gap_count      = 0.0
    t              = 0.0

    while True:
        t += TRACE_STEP
        cx = start[0] + d[0] * t
        cy = start[1] + d[1] * t

        # Ra ngoài ảnh
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            break

        in_mask = _point_in_mask(mask, cx, cy)
        in_arm  = _point_near_arm(cx, cy, left_arm_pts) or \
                  _point_near_arm(cx, cy, right_arm_pts)

        if in_mask:
            gap_count = 0
            if not in_arm:
                last_body_pt = np.array([cx, cy])
        else:
            if not in_arm:
                gap_count += TRACE_STEP
                if gap_count > max_gap:
                    break
            # Nếu trong arm zone mà gặp gap → coi đây là vùng giữa tay và thân,
            # tiếp tục trace nhưng đếm gap riêng (nhỏ hơn)
            else:
                gap_count += TRACE_STEP * 0.3  # arm gap ít bị phạt hơn
                if gap_count > max_gap * 3:    # nhưng không vô hạn
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
    """
    Đo chiều rộng dọc theo đường qua kp_left và kp_right.

    Hướng của đường: direction = normalize(kp_right - kp_left)
    Midpoint: giữa 2 keypoint (là điểm bắt đầu trace)
    Trace trái: ngược direction
    Trace phải: theo direction
    """
    direction = _unit(kp_right - kp_left)
    mid       = (kp_left + kp_right) / 2.0

    p_right = _trace_to_edge(mask, mid,  direction,      left_arm_pts, right_arm_pts)
    p_left  = _trace_to_edge(mask, mid, -direction,      left_arm_pts, right_arm_pts)

    width_px = float(np.linalg.norm(p_right - p_left))

    return {
        "p0":       p_left,
        "p1":       p_right,
        "width_px": width_px,
        "midpoint": (p_left + p_right) / 2.0,
        "angle_deg": float(np.degrees(np.arctan2(direction[1], direction[0]))),
    }


# ── Arm geometry helpers ───────────────────────────────────────────────────

def _is_arm_horizontal(arm_pts: list) -> bool:
    if len(arm_pts) < 2:
        return False
    shoulder, wrist = arm_pts[0], arm_pts[-1]
    dx = abs(wrist[0] - shoulder[0])
    dy = abs(wrist[1] - shoulder[1])
    if dx < 1:
        return False
    return (dy / dx) < config.ARM_HORIZONTAL_THRESHOLD


# ── Public API ─────────────────────────────────────────────────────────────

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

    # ── Waist keypoints: nội suy giữa shoulder và hip ─────────────────────
    lw_kp = ls + WAIST_T * (lh - ls)   # waist trái
    rw_kp = rs + WAIST_T * (rh - rs)   # waist phải

    # ── Đo 3 vùng theo đường nghiêng qua keypoints ────────────────────────
    shoulder_info = _measure_oriented_line(
        mask, ls, rs,
        eff_left_arm, eff_right_arm,
        label="shoulder",
    )
    waist_info = _measure_oriented_line(
        mask, lw_kp, rw_kp,
        eff_left_arm, eff_right_arm,
        label="waist",
    )
    hip_info = _measure_oriented_line(
        mask, lh, rh,
        eff_left_arm, eff_right_arm,
        label="hip",
    )

    # ── Ratios ────────────────────────────────────────────────────────────
    ref = shoulder_info["width_px"] if shoulder_info["width_px"] > 0 else 1.0
    measurements = {
        "shoulder_ratio": round(shoulder_info["width_px"] / ref, 3),
        "waist_ratio":    round(waist_info["width_px"]    / ref, 3),
        "hip_ratio":      round(hip_info["width_px"]      / ref, 3),
        "shoulder_px":    round(shoulder_info["width_px"], 1),
        "waist_px":       round(waist_info["width_px"],    1),
        "hip_px":         round(hip_info["width_px"],      1),
        "shoulder_angle": round(shoulder_info["angle_deg"], 1),
        "waist_angle":    round(waist_info["angle_deg"],    1),
        "hip_angle":      round(hip_info["angle_deg"],      1),
    }

    lines = {"shoulder": shoulder_info, "waist": waist_info, "hip": hip_info}
    return measurements, lines


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bg_remover import remove_background, alpha_to_binary_mask

    bgra = remove_background("test.png")
    mask = alpha_to_binary_mask(bgra)
    kp   = load_keypoints("keypoints.json")

    measurements, lines = estimate_measurements(kp, mask)

    print("=" * 40)
    for k, v in measurements.items():
        print(f"  {k:<20}: {v}")
    print("=" * 40)