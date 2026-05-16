"""
bodyshape_classifier.py
Phân loại body shape từ tỉ lệ shoulder/waist/hip (pixel ratio).
"""


def classify(shoulder: float, waist: float, hip: float) -> str:
    """
    Parameters
    ----------
    shoulder, waist, hip : pixel width

    Returns
    -------
    str : "Hourglass" | "Spoon" | "Triangle" | "Inverted Triangle" | "Rectangle"
    """
    if shoulder < 1e-6 or hip < 1e-6:
        return "Unknown"

    bw = (shoulder  - waist) / shoulder   # độ thắt eo so với shoulder
    hw = (hip   - waist) / hip    # độ thắt eo so với hip
    bh = (shoulder  - hip)   / max(shoulder, hip)  # chênh lệch shoulder vs hip

    if abs(bh) <= 0.10 and bw >= 0.20 and hw >= 0.20:
        return "Hourglass"
    if hip > shoulder:
        return "Spoon" if hw >= 0.20 else "Triangle"
    if shoulder > hip:
        return "Inverted Triangle"
    return "Rectangle"