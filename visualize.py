"""
visualize.py
Tạo 2 file output:
  - output_measurements.png  : ảnh clean (chỉ 3 đường đo + label nhỏ gọn)
  - output_debug.png         : ảnh debug đầy đủ (scan band, arm zone,
                               skeleton, label chi tiết, chú thích màu)
"""

import cv2
import numpy as np

COLORS = {
    "shoulder":  (0, 165, 255),   # cam
    "waist": (0, 220, 80),    # xanh lá
    "hip":   (80, 80, 255),   # xanh dương
}

ARM_KP_COLOR      = (255, 0, 255)   # magenta
ARM_ZONE_COLOR    = (200, 0, 200)   # tím đậm
SCAN_BAND_COLOR   = (200, 200, 200) # xám nhạt
LABEL_BG_ALPHA    = 0.55
LINE_THICKNESS    = 3
CIRCLE_RADIUS     = 6
FONT              = cv2.FONT_HERSHEY_SIMPLEX
ROW_HALF_BAND     = 8


# ── Utilities ─────────────────────────────────────────────────────────────────

def _load(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _label_bg(img: np.ndarray, x: int, y: int, text: str,
               color: tuple, font_scale: float = 0.5, thickness: int = 1):
    """Vẽ text với background mờ để đọc dễ trên mọi nền."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    pad = 4
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, LABEL_BG_ALPHA, img, 1 - LABEL_BG_ALPHA, 0, img)
    cv2.putText(img, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def _draw_measure_line(img: np.ndarray, info: dict, color: tuple,
                        label: str, thickness: int = LINE_THICKNESS,
                        show_band: bool = False):
    """Vẽ đường đo + endpoints + midpoint + label."""
    p0  = tuple(info["p0"].astype(int))
    p1  = tuple(info["p1"].astype(int))
    mid = tuple(info["midpoint"].astype(int))
    cy  = p0[1]

    # Scan band (debug only)
    if show_band:
        H, W = img.shape[:2]
        overlay = img.copy()
        y0 = max(0, cy - ROW_HALF_BAND)
        y1 = min(H - 1, cy + ROW_HALF_BAND)
        cv2.rectangle(overlay, (0, y0), (W, y1), SCAN_BAND_COLOR, -1)
        cv2.addWeighted(overlay, 0.07, img, 0.93, 0, img)

    # Đường chính
    cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)

    # Endpoint markers — hình tròn rỗng + nhỏ bên trong
    for pt in (p0, p1):
        cv2.circle(img, pt, CIRCLE_RADIUS + 2, color, 1, cv2.LINE_AA)
        cv2.circle(img, pt, CIRCLE_RADIUS,     color, -1)

    # Midpoint
    cv2.circle(img, mid, 4, color, -1)

    # Label tại mid
    _label_bg(img, mid[0] - 2, cy - 12, label, color, font_scale=0.5)


def _draw_arm_skeleton(img: np.ndarray, kp: dict):
    """Vẽ skeleton tay (shoulder→elbow→wrist) cả 2 bên."""
    pairs = [(11, 13), (13, 15), (12, 14), (14, 16)]
    for a, b in pairs:
        if a not in kp or b not in kp:
            continue
        pa = (int(kp[a]["x_pixel"]), int(kp[a]["y_pixel"]))
        pb = (int(kp[b]["x_pixel"]), int(kp[b]["y_pixel"]))
        cv2.line(img, pa, pb, ARM_KP_COLOR, 2, cv2.LINE_AA)
    for idx in (11, 12, 13, 14, 15, 16):
        if idx not in kp:
            continue
        pt = (int(kp[idx]["x_pixel"]), int(kp[idx]["y_pixel"]))
        cv2.circle(img, pt, 5, ARM_KP_COLOR, -1)


def _draw_arm_exclusion_zones(img: np.ndarray, kp: dict,
                               waist_y: int, hip_y: int,
                               arm_pad: int = 10):
    """
    Vẽ hộp mờ tại vùng tay bị loại trừ ở dải waist và hip.
    Dựa trên interpolation giống body_measurements.py.
    """
    def interp_x(pts, target_y, pad, band=ROW_HALF_BAND):
        xs = []
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
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
        return (int(min(xs) - pad), int(max(xs) + pad))

    def kp_pt(idx):
        if idx not in kp:
            return None
        return np.array([kp[idx]["x_pixel"], kp[idx]["y_pixel"]], dtype=float)

    ls, le, lw = kp_pt(11), kp_pt(13), kp_pt(15)
    rs, re, rw = kp_pt(12), kp_pt(14), kp_pt(16)

    overlay = img.copy()
    for target_y in (waist_y, hip_y):
        band_h = ROW_HALF_BAND
        y0 = max(0, target_y - band_h)
        y1 = min(img.shape[0] - 1, target_y + band_h)

        if all(p is not None for p in (ls, le, lw)):
            rng = interp_x([ls, le, lw], target_y, arm_pad)
            if rng:
                cv2.rectangle(overlay, (rng[0], y0), (rng[1], y1),
                              ARM_ZONE_COLOR, -1)

        if all(p is not None for p in (rs, re, rw)):
            rng = interp_x([rs, re, rw], target_y, arm_pad)
            if rng:
                cv2.rectangle(overlay, (rng[0], y0), (rng[1], y1),
                              ARM_ZONE_COLOR, -1)

    cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)


def _draw_legend(img: np.ndarray, measurements: dict):
    """
    Vẽ bảng chú thích ở góc trên-phải với px và ratio từng vùng.
    """
    H, W = img.shape[:2]
    lines = [
        ("shoulder",  COLORS["shoulder"],  measurements["shoulder_px"],  measurements["shoulder_ratio"]),
        ("WAIST", COLORS["waist"], measurements["waist_px"], measurements["waist_ratio"]),
        ("HIP",   COLORS["hip"],   measurements["hip_px"],   measurements["hip_ratio"]),
    ]

    box_w, box_h = 210, 18 * len(lines) + 32
    x0 = W - box_w - 16
    y0 = 16

    # Background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (180, 180, 180), 1)

    # Header
    cv2.putText(img, "MEASUREMENTS", (x0 + 10, y0 + 16),
                FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    for i, (name, color, px, ratio) in enumerate(lines):
        ty = y0 + 36 + i * 18
        # Swatch
        cv2.rectangle(img, (x0 + 10, ty - 10), (x0 + 22, ty + 2), color, -1)
        # Text
        text = f"{name:<5}  {px:>6.0f} px   {ratio:.3f}"
        cv2.putText(img, text, (x0 + 30, ty),
                    FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_tick_marks(img: np.ndarray, info: dict, color: tuple, tick_h: int = 8):
    """Vẽ dấu tick nhỏ ở 2 đầu đường đo (như thước đo)."""
    p0 = info["p0"].astype(int)
    p1 = info["p1"].astype(int)
    for pt in (p0, p1):
        top = (pt[0], pt[1] - tick_h)
        bot = (pt[0], pt[1] + tick_h)
        cv2.line(img, top, bot, color, 2, cv2.LINE_AA)


# ── Public API ────────────────────────────────────────────────────────────────

def draw_measurements(
    image_path: str,
    measurements: dict,
    lines: dict,
    output_path: str = "output_measurements.png",
    kp: dict | None = None,
    debug_arms: bool = False,   # giữ tham số cũ để không break main.py
    save_debug: bool = True,    # tự động tạo thêm output_debug.png
):
    """
    Parameters
    ----------
    image_path   : ảnh đã remove background
    measurements : dict shoulder_px / waist_px / hip_px / *_ratio
    lines        : dict shoulder / waist / hip info từ estimate_measurements()
    output_path  : file ảnh clean
    kp           : keypoints dict (cần cho debug)
    debug_arms   : (deprecated, vẫn giữ để tương thích)
    save_debug   : nếu True → lưu thêm output_debug.png
    """
    # ── CLEAN OUTPUT ──────────────────────────────────────────────────────────
    img_clean = _load(image_path)

    for name, info in lines.items():
        color = COLORS[name]
        px    = measurements[f"{name}_px"]
        ratio = measurements[f"{name}_ratio"]
        label = f"{name[0].upper()}"
        _draw_measure_line(img_clean, info, color, label)
        _draw_tick_marks(img_clean, info, color)

    # _draw_legend(img_clean, measurements)
    cv2.imwrite(output_path, img_clean)
    print(f"Clean  → {output_path}")

    # ── DEBUG OUTPUT ──────────────────────────────────────────────────────────
    if not save_debug:
        return

    debug_path = output_path.replace(".png", "_debug.png")
    img_dbg = _load(image_path)

    waist_y = int(lines["waist"]["p0"][1])
    hip_y   = int(lines["hip"]["p0"][1])

    # 1. Arm exclusion zones
    if kp is not None:
        _draw_arm_exclusion_zones(img_dbg, kp, waist_y, hip_y)

    # 2. Scan bands + đường đo với label chi tiết
    for name, info in lines.items():
        color = COLORS[name]
        px    = measurements[f"{name}_px"]
        ratio = measurements[f"{name}_ratio"]
        label = f"{name.upper()}  {px:.0f}px  ratio={ratio:.3f}"
        _draw_measure_line(img_dbg, info, color, label,
                           thickness=LINE_THICKNESS + 1, show_band=True)
        _draw_tick_marks(img_dbg, info, color, tick_h=12)

    # 3. Arm skeleton
    if kp is not None:
        _draw_arm_skeleton(img_dbg, kp)

    # 4. Hip keypoints (anchor chống bị tay kéo)
    if kp is not None:
        for idx, label_txt in [(23, "lhip"), (24, "rhip")]:
            if idx not in kp:
                continue
            pt = (int(kp[idx]["x_pixel"]), int(kp[idx]["y_pixel"]))
            cv2.circle(img_dbg, pt, 7, (0, 255, 255), 2, cv2.LINE_AA)
            _label_bg(img_dbg, pt[0] + 8, pt[1] + 4, label_txt,
                      (0, 255, 255), font_scale=0.38)

    # 5. Shoulder keypoints
    if kp is not None:
        for idx, label_txt in [(11, "lsho"), (12, "rsho")]:
            if idx not in kp:
                continue
            pt = (int(kp[idx]["x_pixel"]), int(kp[idx]["y_pixel"]))
            cv2.circle(img_dbg, pt, 7, (255, 200, 0), 2, cv2.LINE_AA)
            _label_bg(img_dbg, pt[0] + 8, pt[1] + 4, label_txt,
                      (255, 200, 0), font_scale=0.38)

    # 6. Legend
    _draw_legend(img_dbg, measurements)

    # 7. Chú thích màu ở góc dưới
    _draw_color_key(img_dbg)

    cv2.imwrite(debug_path, img_dbg)
    print(f"Debug  → {debug_path}")


def _draw_color_key(img: np.ndarray):
    """Vẽ bảng màu nhỏ ở góc dưới-trái."""
    H, W = img.shape[:2]
    items = [
        (COLORS["shoulder"],   "shoulder line"),
        (COLORS["waist"],  "Waist line"),
        (COLORS["hip"],    "Hip line"),
        (ARM_KP_COLOR,     "Arm skeleton (magenta)"),
        (ARM_ZONE_COLOR,   "Arm exclusion zone"),
        ((0, 255, 255),    "Hip keypoints (anchor)"),
        ((255, 200, 0),    "Shoulder keypoints"),
        (SCAN_BAND_COLOR,  "Scan band ±8px"),
    ]

    line_h = 16
    box_h  = line_h * len(items) + 20
    x0, y0 = 12, H - box_h - 12

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + 230, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, (color, text) in enumerate(items):
        ty = y0 + 14 + i * line_h
        cv2.rectangle(img, (x0 + 8, ty - 9), (x0 + 20, ty + 1), color, -1)
        cv2.putText(img, text, (x0 + 28, ty),
                    FONT, 0.38, (210, 210, 210), 1, cv2.LINE_AA)