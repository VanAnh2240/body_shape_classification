# body_measurements.py
"""
Cải tiến: xử lý các trường hợp
  1. Tay dang ngang hoàn toàn (T-pose) → shoulder bị kéo dài theo tay
  2. Tay dính vào hông/eo → biên mask bị mở rộng theo tay
  3. Tay cúp xuống sát thân → không bị loại sai

Chiến lược:
  - Không dùng keypoint làm mốc độ dài.
  - Dùng "contour narrowing" (tìm điểm lõm trên profile ngang) để phát hiện
    chỗ tay tách khỏi thân, sau đó cắt tại điểm lõm đó.
  - Kết hợp arm skeleton để ưu tiên loại trừ vùng tay đúng phía.
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
ROW_HALF_BAND           = config.ROW_HALF_BAND
ARM_PAD_PX              = config.ARM_PAD_PX
BODY_CENTER_SEARCH_HALF = config.BODY_CENTER_SEARCH_HALF
TORSO_CORRIDOR_MARGIN   = 8

# ── Concavity detection ────────────────────────────────────────────────────
# Khi scan profile ngang, nếu có đoạn lõm (gap hoặc narrow neck) giữa thân
# và tay thì đó là điểm cắt tự nhiên.  Nếu không có gap, dùng gradient âm
# mạnh để phát hiện "neck" chỗ tay gắn vào thân.
CONCAVITY_SEARCH_ROWS   = 30   # scan ±N row để tìm điểm lõm
MIN_CONCAVITY_DROP      = 6    # số pixel drop để coi là điểm lõm đáng kể
GAP_THRESHOLD           = 2    # pixel gap (0-pixel trong mask) = đứt hẳn


def load_keypoints(path: str = "keypoints.json") -> dict:
    with open(path) as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def _kp_xy(kp: dict, idx: int) -> np.ndarray:
    p = kp[idx]
    return np.array([p["x_pixel"], p["y_pixel"]], dtype=float)


# ── Arm geometry helpers ───────────────────────────────────────────────────

def _is_arm_horizontal(arm_pts: list) -> bool:
    """True nếu tay dang ngang (góc so với ngang < threshold)."""
    if len(arm_pts) < 2:
        return False
    shoulder, wrist = arm_pts[0], arm_pts[-1]
    dx = abs(wrist[0] - shoulder[0])
    dy = abs(wrist[1] - shoulder[1])
    if dx < 1:
        return False
    return (dy / dx) < config.ARM_HORIZONTAL_THRESHOLD


def _arm_x_range_at_y(
    arm_pts: list,
    target_y: float,
    band: int = ROW_HALF_BAND,
    pad: int = ARM_PAD_PX,
) -> tuple | None:
    """Trả về (x_min, x_max) của skeleton tay tại target_y, hoặc None."""
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


# ── Profile analysis ───────────────────────────────────────────────────────

def _row_profile(mask: np.ndarray, center_y: int, half_band: int = ROW_HALF_BAND):
    """Trả về mảng bool 1-D: cột nào có foreground pixel trong band."""
    H = mask.shape[0]
    y0 = max(0, center_y - half_band)
    y1 = min(H - 1, center_y + half_band)
    return mask[y0:y1 + 1, :].any(axis=0)


def _find_gap_or_concavity(
    col_hits: np.ndarray,
    start_x: int,
    direction: int,       # -1 = trái, +1 = phải
    arm_xmin: float,
    arm_xmax: float,
) -> int | None:
    """
    Scan từ start_x ra phía direction.
    Trong vùng arm (arm_xmin..arm_xmax):
      - Nếu gặp gap (0-pixel ≥ GAP_THRESHOLD) → trả về x ngay trước gap
      - Nếu không có gap → trả về None (không cắt)
    Ngoài vùng arm → không cắt.
    """
    W = len(col_hits)
    gap_count = 0
    last_body = start_x

    for x in range(start_x, -1 if direction < 0 else W, direction):
        in_arm_zone = arm_xmin <= x <= arm_xmax

        if col_hits[x]:
            gap_count = 0
            if not in_arm_zone:
                # Ngoài vùng arm mà vẫn có mask → đây là thân, ghi nhận
                last_body = x
        else:
            gap_count += 1
            if in_arm_zone and gap_count >= GAP_THRESHOLD:
                # Gap trong vùng tay → điểm cắt là x ngay trước gap
                return last_body

    return None


def _find_body_edge_with_concavity(
    mask: np.ndarray,
    center_y: int,
    cx: int,
    side: str,             # "left" | "right"
    arm_pts: list,         # skeleton tay phía đó
    search_rows: int = CONCAVITY_SEARCH_ROWS,
) -> int:
    """
    Tìm biên thân thực tế tại center_y, xử lý tay dính vào thân.

    Thuật toán:
    1. Lấy profile tại center_y.
    2. Nếu arm skeleton không đi qua center_y → scan thẳng, lấy biên mask.
    3. Nếu arm skeleton đi qua center_y:
       a. Tìm gap trong vùng arm → cắt tại gap.
       b. Nếu không có gap → scan ±search_rows row, chọn row có width
          NHỎ NHẤT trong vùng ngoài torso_x, rồi dùng đó làm ước tính
          ("narrowest cross-section" = eo/hông thực, không phải tay).
          Sau đó project ngược lại center_y.
    """
    H, W = mask.shape
    direction = -1 if side == "left" else 1

    col_hits = _row_profile(mask, center_y)
    arm_range = _arm_x_range_at_y(arm_pts, center_y) if arm_pts else None

    # Không có tay tại row này → lấy biên mask bình thường
    if arm_range is None:
        return _simple_edge(col_hits, cx, direction, W)

    arm_xmin, arm_xmax = arm_range

    # Thử tìm gap trong vùng tay
    gap_cut = _find_gap_or_concavity(col_hits, cx, direction, arm_xmin, arm_xmax)
    if gap_cut is not None:
        return gap_cut

    # Không có gap → dùng "narrowest row" trong vùng xung quanh
    # Scan nhiều row, tìm row có biên ngoài nhỏ nhất (= chỉ thân, không có tay)
    best_edge = _simple_edge(col_hits, cx, direction, W)
    best_edge_in_torso = True  # mặc định dùng biên hiện tại

    narrow_edges = []
    for dy in range(-search_rows, search_rows + 1):
        ry = center_y + dy
        if ry < 0 or ry >= H:
            continue
        rc = _row_profile(mask, ry, half_band=ROW_HALF_BAND)
        r_arm_range = _arm_x_range_at_y(arm_pts, ry) if arm_pts else None

        if r_arm_range is None:
            # Row này không có tay → edge này là "pure body"
            e = _simple_edge(rc, cx, direction, W)
            narrow_edges.append((abs(e - cx), e, ry))
        else:
            # Row này có tay → thử tìm gap
            rxmin, rxmax = r_arm_range
            gap = _find_gap_or_concavity(rc, cx, direction, rxmin, rxmax)
            if gap is not None:
                narrow_edges.append((abs(gap - cx), gap, ry))

    if narrow_edges:
        # Chọn edge ngắn nhất (thân hẹp nhất = không có tay)
        narrow_edges.sort(key=lambda t: t[0])
        best_half_width = narrow_edges[0][0]

        # Áp dụng: dùng half_width này để xác định biên tại center_y
        # Scan từ cx theo direction, lấy pixel thứ best_half_width
        count = 0
        for x in range(cx, -1 if direction < 0 else W, direction):
            if col_hits[x]:
                count += 1
                if count >= best_half_width:
                    return x
            # Nếu gặp gap nhỏ trong vùng arm thì dừng
            elif arm_xmin <= x <= arm_xmax:
                break

    return _simple_edge(col_hits, cx, direction, W)


def _simple_edge(col_hits: np.ndarray, cx: int, direction: int, W: int) -> int:
    """Scan từ cx ra direction, trả về column cuối cùng có foreground."""
    edge = cx
    for x in range(cx, -1 if direction < 0 else W, direction):
        if not col_hits[x]:
            break
        edge = x
    return edge


# ── Torso corridor ─────────────────────────────────────────────────────────

def _body_center_x(mask: np.ndarray, center_y: int) -> float:
    H, W = mask.shape
    y0 = max(0, center_y - ROW_HALF_BAND)
    y1 = min(H - 1, center_y + ROW_HALF_BAND)
    col_hits = mask[y0:y1 + 1, :].any(axis=0)
    cols = np.where(col_hits)[0]
    if cols.size == 0:
        return W / 2.0
    return float((cols[0] + cols[-1]) / 2.0)


# ── Core measurement functions ─────────────────────────────────────────────

def _measure_row(
    mask: np.ndarray,
    center_y: int,
    body_cx: float,
    left_arm_pts: list,
    right_arm_pts: list,
) -> dict:
    """Đo width tại một row, dùng concavity để loại tay."""
    cx = int(round(body_cx))

    left_col = _find_body_edge_with_concavity(
        mask, center_y, cx, "left", left_arm_pts
    )
    right_col = _find_body_edge_with_concavity(
        mask, center_y, cx, "right", right_arm_pts
    )

    p0 = np.array([left_col,  center_y], dtype=float)
    p1 = np.array([right_col, center_y], dtype=float)
    return {
        "p0":       p0,
        "p1":       p1,
        "width_px": float(right_col - left_col),
        "midpoint": (p0 + p1) / 2.0,
    }


def _measure_shoulder(
    mask: np.ndarray,
    shoulder_y: int,
    ls: np.ndarray,
    rs: np.ndarray,
    left_arm_pts: list,
    right_arm_pts: list,
) -> dict:
    """
    Đo shoulder width.
    Với tay dang ngang: arm_pts = [] → scan mask thẳng nhưng bị giới hạn
    bởi concavity của profile (chỗ tay gắn vào vai tạo lõm).
    """
    H, W = mask.shape
    body_cx = (ls[0] + rs[0]) / 2.0
    cx = int(round(body_cx))

    left_col = _find_body_edge_with_concavity(
        mask, shoulder_y, cx, "left", left_arm_pts,
        search_rows=CONCAVITY_SEARCH_ROWS,
    )
    right_col = _find_body_edge_with_concavity(
        mask, shoulder_y, cx, "right", right_arm_pts,
        search_rows=CONCAVITY_SEARCH_ROWS,
    )

    p0 = np.array([left_col,  shoulder_y], dtype=float)
    p1 = np.array([right_col, shoulder_y], dtype=float)
    return {
        "p0":       p0,
        "p1":       p1,
        "width_px": float(right_col - left_col),
        "midpoint": (p0 + p1) / 2.0,
    }


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

    # Tay ngang → truyền [] để concavity search hoạt động trên profile thuần
    # (không có skeleton tay làm reference, nhưng profile sẽ tự nhiên hẹp hơn
    # vì chúng ta tìm narrowest cross-section)
    eff_left_arm  = left_arm_pts  if not left_arm_horizontal else []
    eff_right_arm = right_arm_pts if not right_arm_horizontal else []

    shoulder_y = int(round((ls[1] + rs[1]) / 2.0))
    hip_y      = int(round((lh[1] + rh[1]) / 2.0))
    waist_y    = int(round(shoulder_y + WAIST_T * (hip_y - shoulder_y)))

    # ── Shoulder ──────────────────────────────────────────────────────────────
    shoulder_info = _measure_shoulder(
        mask, shoulder_y, ls, rs,
        eff_left_arm, eff_right_arm,
    )

    # ── Waist & Hip ───────────────────────────────────────────────────────────
    waist_cx = _body_center_x(mask, waist_y)
    hip_cx   = _body_center_x(mask, hip_y)

    waist_info = _measure_row(
        mask, waist_y, waist_cx,
        eff_left_arm, eff_right_arm,
    )
    hip_info = _measure_row(
        mask, hip_y, hip_cx,
        eff_left_arm, eff_right_arm,
    )

    # ── Ratios ────────────────────────────────────────────────────────────────
    ref = shoulder_info["width_px"] if shoulder_info["width_px"] > 0 else 1.0
    measurements = {
        "shoulder_ratio": round(shoulder_info["width_px"] / ref, 3),
        "waist_ratio":    round(waist_info["width_px"]    / ref, 3),
        "hip_ratio":      round(hip_info["width_px"]      / ref, 3),
        "shoulder_px":    round(shoulder_info["width_px"], 1),
        "waist_px":       round(waist_info["width_px"],    1),
        "hip_px":         round(hip_info["width_px"],      1),
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
        print(f"  {k:<16}: {v}")
    print("=" * 40)