"""
bg_remover.py
"""

import cv2
import numpy as np

try:
    from rembg import remove as rembg_remove
    _REMBG_AVAILABLE = True
except ImportError:
    _REMBG_AVAILABLE = False


def _grabcut_remove_bg(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.05), int(h * 0.02), int(w * 0.90), int(h * 0.96))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = fg_mask
    return bgra

def remove_background(image_path: str, output_path: str | None = None) -> np.ndarray:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    if _REMBG_AVAILABLE:
        import io
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        buf_in  = io.BytesIO()
        pil_img.save(buf_in, format="PNG")
        buf_out = io.BytesIO(rembg_remove(buf_in.getvalue()))
        pil_out = Image.open(buf_out).convert("RGBA")
        bgra = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGBA2BGRA)
    else:
        print("[bg_remover] rembg không có sẵn.")
        bgra = _grabcut_remove_bg(bgr)

    if output_path:
        cv2.imwrite(output_path, bgra)

    return bgra


def alpha_to_binary_mask(bgra: np.ndarray, threshold: int = 127) -> np.ndarray:
    return (bgra[:, :, 3] > threshold).astype(np.uint8)


if __name__ == "__main__":
    bgra = remove_background("test.png", output_path="bg_removed.png")
    print(f"bg removed → bg_removed.png  shape={bgra.shape}")