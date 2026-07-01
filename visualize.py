"""
visualize.py
"""

import cv2
import numpy as np

COLORS = {
    "shoulder": (0, 165, 255),  
    "waist":    (0, 220, 80),    
    "high_hip": (180, 60, 220), 
    "hip":      (80, 80, 255),  
}

ARM_KP_COLOR    = (255, 0, 255)
ARM_ZONE_COLOR  = (200, 0, 200)
LABEL_BG_ALPHA  = 0.55
LINE_THICKNESS  = 3
CIRCLE_RADIUS   = 6
FONT            = cv2.FONT_HERSHEY_SIMPLEX


# ── Utilities ─────────────────────────────────────────────────────────────

def _load(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _label_bg(img, x, y, text, color, font_scale=0.5, thickness=1):
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    pad = 4
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, LABEL_BG_ALPHA, img, 1 - LABEL_BG_ALPHA, 0, img)
    cv2.putText(img, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def _draw_measure_line(img, info: dict, color: tuple, label: str,
                       thickness: int = LINE_THICKNESS, show_band: bool = False):
    p0  = tuple(info["p0"].astype(int))
    p1  = tuple(info["p1"].astype(int))
    mid = tuple(info["midpoint"].astype(int))

    if show_band:
        _draw_oriented_band(img, info, band_px=8)

    cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)

    for pt in (p0, p1):
        cv2.circle(img, pt, CIRCLE_RADIUS + 2, color, 1, cv2.LINE_AA)
        cv2.circle(img, pt, CIRCLE_RADIUS,     color, -1)

    cv2.circle(img, mid, 4, color, -1)

    lx = mid[0] - 2
    ly = mid[1] - 14
    _label_bg(img, lx, ly, label, color, font_scale=0.5)


def _draw_oriented_band(img, info: dict, band_px: int = 8, alpha: float = 0.07):
    p0  = info["p0"]
    p1  = info["p1"]
    d   = p1 - p0
    n   = np.linalg.norm(d)
    if n < 1e-6:
        return
    perp = np.array([-d[1], d[0]]) / n * band_px 

    pts = np.array([
        p0 + perp, p1 + perp,
        p1 - perp, p0 - perp,
    ], dtype=np.int32)

    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (200, 200, 200))
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_tick_marks(img, info: dict, color: tuple, tick_h: int = 8):
    p0 = info["p0"]
    p1 = info["p1"]
    d  = p1 - p0
    n  = np.linalg.norm(d)
    if n < 1e-6:
        return
    perp = np.array([-d[1], d[0]]) / n * tick_h 

    for pt in (p0, p1):
        tp0 = tuple((pt + perp).astype(int))
        tp1 = tuple((pt - perp).astype(int))
        cv2.line(img, tp0, tp1, color, 2, cv2.LINE_AA)


def _draw_arm_skeleton(img, kp: dict):
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


def _draw_legend(img, measurements: dict):
    H, W = img.shape[:2]
    lines_data = [
        ("shoulder", COLORS["shoulder"], measurements["shoulder_px"], measurements["shoulder_ratio"],
         measurements.get("shoulder_angle", 0)),
        ("waist",    COLORS["waist"],    measurements["waist_px"],    measurements["waist_ratio"],
         measurements.get("waist_angle", 0)),
        ("high_hip", COLORS["high_hip"], measurements["high_hip_px"], measurements["high_hip_ratio"],
         measurements.get("high_hip_angle", 0)),
        ("hip",      COLORS["hip"],      measurements["hip_px"],      measurements["hip_ratio"],
         measurements.get("hip_angle", 0)),
    ]

    box_w, box_h = 260, 18 * len(lines_data) + 48
    x0 = W - box_w - 16
    y0 = 16

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (180, 180, 180), 1)

    cv2.putText(img, "MEASUREMENTS", (x0 + 10, y0 + 16),
                FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "(oriented line)", (x0 + 10, y0 + 30),
                FONT, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

    for i, (name, color, px, ratio, angle) in enumerate(lines_data):
        ty = y0 + 48 + i * 18
        cv2.rectangle(img, (x0 + 10, ty - 10), (x0 + 22, ty + 2), color, -1)
        text = f"{name:<8}  {px:>6.0f}px  {ratio:.3f}  {angle:+.1f}deg"
        cv2.putText(img, text, (x0 + 30, ty),
                    FONT, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_keypoint_anchors(img, kp: dict):
    anchors = {
        11: ("lsho", (255, 200, 0)),
        12: ("rsho", (255, 200, 0)),
        23: ("lhip", (0, 255, 255)),
        24: ("rhip", (0, 255, 255)),
    }
    for idx, (lbl, color) in anchors.items():
        if idx not in kp:
            continue
        pt = (int(kp[idx]["x_pixel"]), int(kp[idx]["y_pixel"]))
        cv2.circle(img, pt, 7, color, 2, cv2.LINE_AA)
        cv2.circle(img, pt, 3, color, -1)
        _label_bg(img, pt[0] + 8, pt[1] + 4, lbl, color, font_scale=0.38)


def _draw_color_key(img):
    H, W = img.shape[:2]
    items = [
        (COLORS["shoulder"],   "Shoulder line"),
        (COLORS["waist"],      "Waist line"),
        (COLORS["high_hip"],   "High Hip line"),  
        (COLORS["hip"],        "Hip line"),
        (ARM_KP_COLOR,         "Arm skeleton"),
        ((255, 200, 0),        "Shoulder keypoints"),
        ((0, 255, 255),        "Hip keypoints"),
    ]
    line_h = 16
    box_h  = line_h * len(items) + 20
    x0, y0 = 12, H - box_h - 12

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + 250, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, (color, text) in enumerate(items):
        ty = y0 + 14 + i * line_h
        cv2.rectangle(img, (x0 + 8, ty - 9), (x0 + 20, ty + 1), color, -1)
        cv2.putText(img, text, (x0 + 28, ty),
                    FONT, 0.38, (210, 210, 210), 1, cv2.LINE_AA)

_LINE_ORDER = ["shoulder", "waist", "high_hip", "hip"]


def draw_measurements(
    image_path: str,
    measurements: dict,
    lines: dict,
    output_path: str = "output_measurements.png",
    kp: dict | None = None,
    debug_arms: bool = False, 
    save_debug: bool = True,
):
    img_clean = _load(image_path)

    for name in _LINE_ORDER:
        if name not in lines:
            continue
        info  = lines[name]
        color = COLORS[name]
        label = f"{name[0].upper()}" if name != "high_hip" else "HH"
        _draw_measure_line(img_clean, info, color, label)
        _draw_tick_marks(img_clean, info, color)

    cv2.imwrite(output_path, img_clean)
    print(f"Clean  → {output_path}")

    if not save_debug:
        return

    debug_path = output_path.replace(".png", "_debug.png")
    img_dbg = _load(image_path)

    for name in _LINE_ORDER:
        if name not in lines:
            continue
        info  = lines[name]
        color = COLORS[name]
        px    = measurements[f"{name}_px"]
        ratio = measurements[f"{name}_ratio"]
        angle = measurements.get(f"{name}_angle", 0)
        label = f"{name.upper()}  {px:.0f}px  r={ratio:.3f}  {angle:+.1f}deg"
        _draw_measure_line(img_dbg, info, color, label,
                           thickness=LINE_THICKNESS + 1, show_band=True)
        _draw_tick_marks(img_dbg, info, color, tick_h=12)

    if kp is not None:
        _draw_arm_skeleton(img_dbg, kp)

    if kp is not None:
        _draw_keypoint_anchors(img_dbg, kp)

    _draw_legend(img_dbg, measurements)

    _draw_color_key(img_dbg)

    cv2.imwrite(debug_path, img_dbg)
    print(f"Debug  → {debug_path}")