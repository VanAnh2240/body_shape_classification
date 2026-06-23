"""
bodyshape_classifier.py
Phân loại 5 body shape theo FFIT (Simmons et al. 2004 / Yim Lee et al. 2007),
chuyển đổi ngưỡng inches → pixel-ratio so với shoulder_px.

Quy đổi (tham chiếu shoulder width ≈ 17 inches trung bình):
  ±1"   → ±0.06 * S
  3.6"  → 0.21  * S
  9"    → 0.53  * S   (bust/shoulder-to-waist)
  10"   → 0.59  * S   (hip-to-waist)
  7"    → 0.41  * S   (hip-to-waist ngưỡng Spoon)

Thứ tự ưu tiên (đúng theo FFIT):
  1. Hourglass
  2. Spoon
  3. Inverted Triangle
  4. Triangle
  5. Rectangle  (mặc định)

Inputs (pixel width):
  shoulder_px  — chiều rộng đường vai
  waist_px     — chiều rộng đường eo
  high_hip_px  — chiều rộng đường high-hip (t=0.80 giữa vai và hông)
  hip_px       — chiều rộng đường hông
"""

# ── Ngưỡng pixel-ratio (nhân với shoulder_px) ─────────────────────────────
# Có thể chỉnh fine-tune tại đây nếu cần.

_R_BUST_HIP_SAME   = 0.06   # |shoulder - hip| < ngưỡng → coi bằng nhau (FFIT: 1")
_R_HIP_BUST_SMALL  = 0.21   # hip - shoulder  < ngưỡng  (FFIT: 3.6")
_R_WAIST_DEF_S     = 0.53   # shoulder - waist >= ngưỡng → định nghĩa eo rõ (FFIT: 9")
_R_WAIST_DEF_H     = 0.59   # hip      - waist >= ngưỡng → định nghĩa eo rõ (FFIT: 10")
_R_WAIST_SPOON     = 0.41   # hip      - waist >= ngưỡng → Spoon              (FFIT: 7")
_R_HIP_BUST_LARGE  = 0.21   # hip - shoulder  >= ngưỡng → hip dominates      (FFIT: 3.6")
_HIGH_HIP_WAIST_R  = 1.193  # high_hip / waist > ngưỡng → Spoon (FFIT: giữ nguyên)


def classify(
    shoulder_px: float,
    waist_px: float,
    hip_px: float,
    high_hip_px: float | None = None,
) -> str:
    """
    Phân loại body shape theo FFIT (5 loại chính).

    Parameters
    ----------
    shoulder_px  : chiều rộng vai (pixel)
    waist_px     : chiều rộng eo  (pixel)
    hip_px       : chiều rộng hông (pixel)
    high_hip_px  : chiều rộng high-hip (pixel); nếu None sẽ ước tính.

    Returns
    -------
    str : "Hourglass" | "Spoon" | "Inverted Triangle" | "Triangle" | "Rectangle"
    """
    if shoulder_px < 1e-6 or hip_px < 1e-6 or waist_px < 1e-6:
        return "Unknown"

    S = shoulder_px   # alias ngắn

    # Ước tính high_hip nếu không có (trung bình waist và hip)
    hh = high_hip_px if (high_hip_px is not None and high_hip_px > 1e-6) \
         else (waist_px + hip_px) / 2.0

    # ── Các hiệu số chuẩn (đều dương nếu "lớn hơn") ──────────────────────
    sh_diff  = shoulder_px - hip_px          # shoulder - hip  (>0: vai rộng hơn)
    hs_diff  = hip_px - shoulder_px          # hip - shoulder  (>0: hông rộng hơn)
    sw_diff  = shoulder_px - waist_px        # shoulder - waist
    hw_diff  = hip_px      - waist_px        # hip - waist
    hh_waist = hh / waist_px                 # high_hip / waist ratio

    # ── 1. Hourglass ──────────────────────────────────────────────────────
    # FFIT: |bust-hips| < 1" AND (bust-waist ≥ 9" OR hips-waist ≥ 10")
    if (abs(sh_diff) < _R_BUST_HIP_SAME * S
            and (sw_diff >= _R_WAIST_DEF_S * S or hw_diff >= _R_WAIST_DEF_H * S)):
        return "Hourglass"

    # ── 2. Spoon ──────────────────────────────────────────────────────────
    # FFIT: hips-bust > 3.6" AND hips-waist ≥ 7" AND high_hip/waist > 1.193
    if (hs_diff > _R_HIP_BUST_LARGE * S
            and hw_diff >= _R_WAIST_SPOON * S
            and hh_waist > _HIGH_HIP_WAIST_R):
        return "Spoon"

    # ── 3. Inverted Triangle ──────────────────────────────────────────────
    # FFIT: bust-hips ≥ 3.6" AND hips-waist < 10"
    if sh_diff >= _R_HIP_BUST_LARGE * S and hw_diff < _R_WAIST_DEF_H * S:
        return "Inverted Triangle"

    # ── 4. Triangle ───────────────────────────────────────────────────────
    # FFIT: hips-bust ≥ 3.6" AND hips-waist < 9"   (tìm SAU Spoon và Bottom Hourglass)
    if hs_diff >= _R_HIP_BUST_LARGE * S and hw_diff < _R_WAIST_DEF_S * S:
        return "Triangle"

    # ── 5. Rectangle (mặc định) ───────────────────────────────────────────
    # FFIT: |bust-hips| < 3.6" AND bust-waist < 9" AND hips-waist < 10"
    return "Rectangle"


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Một số test case điển hình
    cases = [
        # (shoulder, waist, hip, high_hip, expected)
        (170, 100, 168, 150, "Hourglass"),          # vai≈hông, eo thắt mạnh
        (160,  95, 185, 175, "Spoon"),               # hông >> vai, high_hip lớn
        (180, 120, 145, 133, "Inverted Triangle"),   # vai >> hông, eo không thắt
        (145, 115, 175, 155, "Triangle"),            # hông >> vai, eo không thắt
        (160, 140, 162, 151, "Rectangle"),           # vai≈hông, eo không thắt
    ]

    print(f"{'shoulder':>10} {'waist':>8} {'hip':>8} {'high_hip':>10}  {'result':<20} {'expected'}")
    print("-" * 75)
    for s, w, h, hh, expected in cases:
        result = classify(s, w, h, hh)
        mark = "✓" if result == expected else "✗"
        print(f"{s:>10} {w:>8} {h:>8} {hh:>10}  {result:<20} {expected}  {mark}")