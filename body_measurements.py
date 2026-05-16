# body_measurements.py

import json
import numpy as np
import config

LEFT_SHOULDER  = config.LEFT_SHOULDER
RIGHT_SHOULDER = config.RIGHT_SHOULDER
LEFT_ELBOW     = config.LEFT_ELBOW
RIGHT_ELBOW    = config.RIGHT_ELBOW
LEFT_WRIST     = config.LEFT_WRIST
RIGHT_WRIST    = config.RIGHT_WRIST
LEFT_HIP       = config.LEFT_HIP
RIGHT_HIP      = config.RIGHT_HIP

WAIST_T             = config.WAIST_T
ROW_HALF_BAND       = config.ROW_HALF_BAND
ARM_PAD_PX          = config.ARM_PAD_PX
BODY_CENTER_SEARCH_HALF = config.BODY_CENTER_SEARCH_HALF

# Margin thêm vào torso corridor (pixel) để không clamp quá chặt
TORSO_CORRIDOR_MARGIN = 8


def load_keypoints(path: str = "keypoints.json") -> dict:
    with open(path) as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def _kp_xy(kp: dict, idx: int) -> np.ndarray:
    p = kp[idx]
    return np.array([p["x_pixel"], p["y_pixel"]], dtype=float)


def _arm_x_range_at_y(
    arm_pts: list,
    target_y: float,
    band: int = ROW_HALF_BAND,
    pad: int = ARM_PAD_PX,
) -> tuple | None:
    """Trả về (x_min, x_max) của đoạn tay tại target_y, hoặc None."""
    if not arm_pts:
        return None
    xs = []
    for i in range(len(arm_pts) - 1):
        a, b = arm_pts[i], arm_pts[i + 1]
        y0, y1 = min(a[1], b[1]), max(a[1], b[1])
        if y1 + band < target_y - band or y0 - band > target_y + band:
            continue
        if abs(b[1] - a[1]) < 1e-3:
            xs.extend([a[0], b[0]])
        else:
            t = np.clip((target_y - a[1]) / (b[1] - a[1]), 0.0, 1.0)
            xs.append(a[0] + t * (b[0] - a[0]))
    if not xs:
        return None
    return (min(xs) - pad, max(xs) + pad)


def _is_arm_horizontal(arm_pts: list) -> bool:
    """True nếu tay đang dang ngang (góc < threshold so với ngang)."""
    if not arm_pts:
        return False
    shoulder, wrist = arm_pts[0], arm_pts[-1]
    dx = abs(wrist[0] - shoulder[0])
    dy = abs(wrist[1] - shoulder[1])
    if dx < 1:
        return False
    return (dy / dx) < config.ARM_HORIZONTAL_THRESHOLD


def _body_center_x(mask: np.ndarray, center_y: int) -> float:
    H, W = mask.shape
    y0 = max(0, center_y - ROW_HALF_BAND)
    y1 = min(H - 1, center_y + ROW_HALF_BAND)
    col_hits = mask[y0:y1 + 1, :].any(axis=0)
    cols = np.where(col_hits)[0]
    if cols.size == 0:
        return W / 2.0
    return float((cols[0] + cols[-1]) / 2.0)


def _arm_overlaps_mask_outside_torso(
    mask: np.ndarray,
    arm_range: tuple,          # (x_min, x_max) của tay tại row này
    torso_left: float,
    torso_right: float,
    center_y: int,
    side: str,                 # "left" hoặc "right"
) -> bool:
    """
    Kiểm tra tay có pixel mask nằm NGOÀI torso corridor không.
    Nếu có → tay thực sự lộ ra ngoài thân → cần loại trừ.
    Nếu không → tay đang nằm hoàn toàn trước thân → không loại trừ.
    """
    arm_xmin, arm_xmax = arm_range
    H, W = mask.shape
    y0 = max(0, center_y - ROW_HALF_BAND)
    y1 = min(H - 1, center_y + ROW_HALF_BAND)

    if side == "left":
        # Kiểm tra pixel tay bên trái corridor
        x_start = max(0, int(arm_xmin))
        x_end   = min(W - 1, int(torso_left))
        if x_start >= x_end:
            return False
        region = mask[y0:y1 + 1, x_start:x_end]
    else:
        # Kiểm tra pixel tay bên phải corridor
        x_start = max(0, int(torso_right))
        x_end   = min(W - 1, int(arm_xmax))
        if x_start >= x_end:
            return False
        region = mask[y0:y1 + 1, x_start:x_end]

    return bool(region.any())


def _scan_border(
    col_hits: np.ndarray,
    cx: int,
    W: int,
    arm_range: tuple | None,
    torso_inner: float,   # shoulder_x hoặc hip_x phía trong
    torso_outer: float,   # hard boundary: arm chỉ block nếu ngoài đây
    side: str,            # "left" hoặc "right"
    arm_is_outside: bool, # kết quả từ _arm_overlaps_mask_outside_torso
) -> int:
    """
    Scan từ center (cx) ra biên, trả về column cuối cùng thuộc thân.

    Logic:
      - Luôn đi qua pixel trong [torso_outer_left, torso_outer_right]
      - Nếu gặp arm_range VÀ arm thực sự lộ ngoài torso → dừng
      - Nếu arm nằm hoàn toàn trong torso → bỏ qua arm_range, tiếp tục scan mask
    """
    col = cx
    if side == "left":
        rng = range(cx, -1, -1)
        def inside_torso(x): return x >= torso_outer
    else:
        rng = range(cx, W)
        def inside_torso(x): return x <= torso_outer

    for x in rng:
        if not col_hits[x]:
            break
        if arm_range is not None and arm_is_outside:
            arm_xmin, arm_xmax = arm_range
            if arm_xmin <= x <= arm_xmax:
                if inside_torso(x):
                    # Trong torso corridor → vẫn tính là thân
                    col = x
                    continue
                else:
                    # Ngoài torso corridor và trong arm range → dừng
                    break
        col = x
    return col


def _measure_row(
    mask: np.ndarray,
    center_y: int,
    body_cx: float,
    left_arm_pts: list,
    right_arm_pts: list,
    torso_left_x: float,    # biên trái của torso corridor
    torso_right_x: float,   # biên phải của torso corridor
) -> dict:
    """
    Đo width của thân tại một row, loại trừ tay chỉ khi tay lộ ra ngoài torso.
    """
    H, W = mask.shape
    y0 = max(0, center_y - ROW_HALF_BAND)
    y1 = min(H - 1, center_y + ROW_HALF_BAND)
    col_hits = mask[y0:y1 + 1, :].any(axis=0)
    cx = int(round(body_cx))

    left_arm_range  = _arm_x_range_at_y(left_arm_pts,  center_y)
    right_arm_range = _arm_x_range_at_y(right_arm_pts, center_y)

    # Kiểm tra tay có lộ ra ngoài torso corridor không
    left_arm_outside = False
    if left_arm_range is not None:
        left_arm_outside = _arm_overlaps_mask_outside_torso(
            mask, left_arm_range,
            torso_left_x, torso_right_x,
            center_y, "left"
        )

    right_arm_outside = False
    if right_arm_range is not None:
        right_arm_outside = _arm_overlaps_mask_outside_torso(
            mask, right_arm_range,
            torso_left_x, torso_right_x,
            center_y, "right"
        )

    left_col = _scan_border(
        col_hits, cx, W,
        left_arm_range, torso_left_x, torso_left_x,
        "left", left_arm_outside
    )
    right_col = _scan_border(
        col_hits, cx, W,
        right_arm_range, torso_right_x, torso_right_x,
        "right", right_arm_outside
    )

    p0  = np.array([left_col,  center_y], dtype=float)
    p1  = np.array([right_col, center_y], dtype=float)
    return {
        "p0":       p0,
        "p1":       p1,
        "width_px": float(right_col - left_col),
        "midpoint": (p0 + p1) / 2.0,
    }


def _measure_shoulder(
    mask: np.ndarray,
    shoulder_y: int,
    ls: np.ndarray, rs: np.ndarray,
    left_arm_pts: list, right_arm_pts: list,
    torso_left_x: float, torso_right_x: float,
) -> dict:
    """
    Đo shoulder:
    - Nếu tay lộ ngoài torso tại shoulder_y → clamp về shoulder keypoint
    - Nếu tay không lộ (tay ôm thân hoặc không có mặt) → lấy mask
    """
    H, W = mask.shape
    y0 = max(0, shoulder_y - ROW_HALF_BAND)
    y1 = min(H - 1, shoulder_y + ROW_HALF_BAND)
    col_hits = mask[y0:y1 + 1, :].any(axis=0)

    body_cx = (ls[0] + rs[0]) / 2.0
    cx = int(round(body_cx))

    left_arm_range  = _arm_x_range_at_y(left_arm_pts,  shoulder_y)
    right_arm_range = _arm_x_range_at_y(right_arm_pts, shoulder_y)

    left_arm_outside = False
    if left_arm_range is not None:
        left_arm_outside = _arm_overlaps_mask_outside_torso(
            mask, left_arm_range,
            torso_left_x, torso_right_x,
            shoulder_y, "left"
        )

    right_arm_outside = False
    if right_arm_range is not None:
        right_arm_outside = _arm_overlaps_mask_outside_torso(
            mask, right_arm_range,
            torso_left_x, torso_right_x,
            shoulder_y, "right"
        )

    # Biên trái
    if left_arm_outside:
        # Tay lộ ra ngoài → clamp về shoulder keypoint (không để mask kéo ra)
        final_left = float(min(ls[0], rs[0]))
    else:
        # Tay không lộ hoặc không có → scan mask tự do
        final_left = float(cx)
        for x in range(cx, -1, -1):
            if not col_hits[x]:
                break
            final_left = float(x)

    # Biên phải
    if right_arm_outside:
        final_right = float(max(ls[0], rs[0]))
    else:
        final_right = float(cx)
        for x in range(cx, W):
            if not col_hits[x]:
                break
            final_right = float(x)

    p0  = np.array([final_left,  shoulder_y], dtype=float)
    p1  = np.array([final_right, shoulder_y], dtype=float)
    return {
        "p0":       p0,
        "p1":       p1,
        "width_px": float(final_right - final_left),
        "midpoint": (p0 + p1) / 2.0,
    }


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

    shoulder_y  = int(round((ls[1] + rs[1]) / 2.0))
    hip_y   = int(round((lh[1] + rh[1]) / 2.0))
    waist_y = int(round(shoulder_y + WAIST_T * (hip_y - shoulder_y)))

    # ── Torso corridor ────────────────────────────────────────────────────────
    # Vùng X chắc chắn thuộc thân: từ keypoint trong cùng ± margin
    # Dùng min/max của shoulder + hip để bao phủ toàn thân
    torso_left_x  = min(ls[0], lh[0], rs[0], rh[0]) - TORSO_CORRIDOR_MARGIN
    torso_right_x = max(ls[0], lh[0], rs[0], rh[0]) + TORSO_CORRIDOR_MARGIN

    # Tay ngang → loại trừ hoàn toàn khỏi arm_pts khi scan
    eff_left_arm  = left_arm_pts  if not left_arm_horizontal  else []
    eff_right_arm = right_arm_pts if not right_arm_horizontal else []

    # ── shoulder ──────────────────────────────────────────────────────────────────
    shoulder_info = _measure_shoulder(
        mask, shoulder_y, ls, rs,
        eff_left_arm, eff_right_arm,
        torso_left_x, torso_right_x,
    )

    # ── Waist & Hip ───────────────────────────────────────────────────────────
    waist_cx = _body_center_x(mask, waist_y)
    hip_cx   = _body_center_x(mask, hip_y)

    waist_info = _measure_row(
        mask, waist_y, waist_cx,
        eff_left_arm, eff_right_arm,
        torso_left_x, torso_right_x,
    )
    hip_info = _measure_row(
        mask, hip_y, hip_cx,
        eff_left_arm, eff_right_arm,
        torso_left_x, torso_right_x,
    )

    # ── Ratios ────────────────────────────────────────────────────────────────
    ref = shoulder_info["width_px"] if shoulder_info["width_px"] > 0 else 1.0
    measurements = {
        "shoulder_ratio":  round(shoulder_info["width_px"]  / ref, 3),
        "waist_ratio": round(waist_info["width_px"] / ref, 3),
        "hip_ratio":   round(hip_info["width_px"]   / ref, 3),
        "shoulder_px":     round(shoulder_info["width_px"],  1),
        "waist_px":    round(waist_info["width_px"], 1),
        "hip_px":      round(hip_info["width_px"],   1),
    }

    lines = {"shoulder": shoulder_info, "waist": waist_info, "hip": hip_info}
    return measurements, lines


if __name__ == "__main__":
    from bg_remover import remove_background, alpha_to_binary_mask

    bgra = remove_background("test.png")
    mask = alpha_to_binary_mask(bgra)
    kp   = load_keypoints("keypoints.json")

    measurements, lines = estimate_measurements(kp, mask)

    print("=" * 40)
    for k, v in measurements.items():
        print(f"  {k:<14}: {v}")
    print("=" * 40)