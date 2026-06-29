"""
bodyshape_classifier.py

Quy đổi:
  ±1"   → ±0.06 * S
  3.6"  → 0.21  * S
  9"    → 0.53  * S   (bust/shoulder-to-waist)
  10"   → 0.59  * S   (hip-to-waist)
  7"    → 0.41  * S   (hip-to-waist ngưỡng Spoon)
"""

_R_BUST_HIP_SAME   = 0.06   # |shoulder - hip| < ngưỡng → coi bằng nhau (FFIT: 1")
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
    if shoulder_px < 1e-6 or hip_px < 1e-6 or waist_px < 1e-6:
        return "Unknown"

    S = shoulder_px 

    hh = high_hip_px if (high_hip_px is not None and high_hip_px > 1e-6) \
         else (waist_px + hip_px) / 2.0

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