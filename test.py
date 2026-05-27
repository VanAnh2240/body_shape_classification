"""
batch_test.py
=============
Chạy pipeline đo body shape cho tất cả ảnh trong thư mục test/*.jpg
(cũng hỗ trợ .png và .jpeg)

Cách dùng:
    python test.py                   # test/ mặc định
    python test.py --input test/     # chỉ định thư mục
    python test.py --input test/ --no-debug   # chỉ xuất ảnh clean

Output cho mỗi ảnh input.jpg:
    results/input/
        ├── keypoints.json
        ├── bg_removed.png
        ├── output_pose.png
        ├── output_measurements.png
        └── output_measurements_debug.png

Kết thúc: in bảng tổng hợp kết quả tất cả ảnh.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ── Thêm thư mục gốc vào sys.path (nếu chạy từ subfolder) ────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pose_estimator    import detect_pose, draw_pose_overlay
from bg_remover        import remove_background, alpha_to_binary_mask
from body_measurements import load_keypoints, estimate_measurements
from bodyshape_classifier import classify
from visualize         import draw_measurements
import config


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ── ANSI colors cho terminal ─────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"
    WHITE  = "\033[97m"


def _hr(char="─", width=60):
    return char * width


def process_image(
    image_path: Path,
    out_dir: Path,
    save_debug: bool = True,
) -> dict:
    """
    Chạy full pipeline cho 1 ảnh.
    Trả về dict kết quả (hoặc {"error": ...} nếu thất bại).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    kp_json      = out_dir / "keypoints.json"
    bg_removed   = out_dir / "bg_removed.png"
    output_image = out_dir / "output_measurements.png"

    t0 = time.time()

    # 1. Pose estimation
    keypoints, result, mp_image = detect_pose(str(image_path), config.MODEL_PATH)
    with open(kp_json, "w", encoding="utf-8") as f:
        json.dump(keypoints, f, indent=2)
    draw_pose_overlay(mp_image, result, str(out_dir / "output_pose.png"))

    # 2. Remove background
    bgra = remove_background(str(image_path), output_path=str(bg_removed))
    mask = alpha_to_binary_mask(bgra)

    # 3. Measure
    kp = load_keypoints(str(kp_json))
    measurements, lines = estimate_measurements(kp, mask)

    # 4. Classify
    shape = classify(
        measurements["shoulder_px"],
        measurements["waist_px"],
        measurements["hip_px"],
    )

    # 5. Visualize
    draw_measurements(
        str(bg_removed), measurements, lines,
        output_path=str(output_image),
        kp=kp,
        save_debug=save_debug,
    )

    elapsed = time.time() - t0

    return {
        "file":           image_path.name,
        "shape":          shape,
        "shoulder_px":    measurements["shoulder_px"],
        "waist_px":       measurements["waist_px"],
        "hip_px":         measurements["hip_px"],
        "shoulder_ratio": measurements["shoulder_ratio"],
        "waist_ratio":    measurements["waist_ratio"],
        "hip_ratio":      measurements["hip_ratio"],
        "shoulder_angle": measurements.get("shoulder_angle", 0),
        "waist_angle":    measurements.get("waist_angle", 0),
        "hip_angle":      measurements.get("hip_angle", 0),
        "elapsed_s":      round(elapsed, 2),
        "out_dir":        str(out_dir),
        "error":          None,
    }


def print_summary(results: list):
    """In bảng tổng hợp ra terminal."""
    ok      = [r for r in results if r["error"] is None]
    failed  = [r for r in results if r["error"] is not None]

    print(f"\n{C.BOLD}{_hr('═', 90)}{C.RESET}")
    print(f"{C.BOLD}{'TONG HOP KET QUA':^90}{C.RESET}")
    print(f"{C.BOLD}{_hr('═', 90)}{C.RESET}")

    if ok:
        # Header
        col = [18, 18, 8, 8, 8, 7, 7, 7, 7, 6]
        hdr = ["File", "Shape", "Sho(px)", "Wai(px)", "Hip(px)",
               "S/rat", "W/rat", "H/rat", "Angle°", "Time"]
        print(C.CYAN + C.BOLD +
              "  ".join(h.ljust(col[i]) for i, h in enumerate(hdr)) +
              C.RESET)
        print(C.GRAY + _hr("─", 90) + C.RESET)

        for r in ok:
            shape_color = {
                "Hourglass":         C.GREEN,
                "Spoon":             C.YELLOW,
                "Triangle":          C.YELLOW,
                "Inverted Triangle": C.CYAN,
                "Rectangle":         C.WHITE,
            }.get(r["shape"], C.WHITE)

            row = [
                r["file"][:17],
                r["shape"][:17],
                str(int(r["shoulder_px"])),
                str(int(r["waist_px"])),
                str(int(r["hip_px"])),
                f"{r['shoulder_ratio']:.3f}",
                f"{r['waist_ratio']:.3f}",
                f"{r['hip_ratio']:.3f}",
                f"{r['shoulder_angle']:+.1f}",
                f"{r['elapsed_s']}s",
            ]
            line = "  ".join(v.ljust(col[i]) for i, v in enumerate(row))
            print(shape_color + line + C.RESET)

    if failed:
        print(f"\n{C.RED}{C.BOLD}THAT BAI ({len(failed)} anh):{C.RESET}")
        for r in failed:
            print(f"  {C.RED}✗  {r['file']}: {r['error']}{C.RESET}")

    print(C.GRAY + _hr("─", 90) + C.RESET)
    print(f"{C.BOLD}  Tong: {len(results)} anh  |  "
          f"{C.GREEN}OK: {len(ok)}{C.RESET}{C.BOLD}  |  "
          f"{C.RED}Loi: {len(failed)}{C.RESET}")
    print(f"{C.BOLD}{_hr('═', 90)}{C.RESET}\n")


def save_summary_json(results: list, out_path: Path):
    """Lưu kết quả tổng hợp ra JSON."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"{C.GRAY}Summary JSON -> {out_path}{C.RESET}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch test body shape measurement pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        default="test",
        help="Thu muc chua anh test (default: test/)"
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Thu muc luu ket qua (default: results/)"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Khong xuat file *_debug.png (nhanh hon)"
    )
    parser.add_argument(
        "--ext",
        default=".jpg,.jpeg,.png,.webp",
        help="Danh sach duoi file, ngan cach bang dau phay"
    )
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    save_debug = not args.no_debug
    exts       = {e.strip().lower() for e in args.ext.split(",")}

    # Tìm ảnh
    if not input_dir.exists():
        print(f"{C.RED}Loi: thu muc '{input_dir}' khong ton tai.{C.RESET}")
        sys.exit(1)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )

    if not images:
        print(f"{C.YELLOW}Khong tim thay anh nao trong '{input_dir}' (ext: {exts}){C.RESET}")
        sys.exit(0)

    print(f"\n{C.BOLD}{_hr('═', 60)}{C.RESET}")
    print(f"{C.BOLD}  BATCH BODY SHAPE TEST{C.RESET}")
    print(f"  Input  : {input_dir.resolve()}")
    print(f"  Output : {output_dir.resolve()}")
    print(f"  Anh    : {len(images)} file")
    print(f"  Debug  : {'Tat' if not save_debug else 'Bat'}")
    print(f"{C.BOLD}{_hr('═', 60)}{C.RESET}\n")

    results = []

    for idx, img_path in enumerate(images, 1):
        stem    = img_path.stem
        out_dir = output_dir / stem

        prefix = f"[{idx:>2}/{len(images)}]  {img_path.name:<30}"
        print(f"{C.BOLD}{prefix}{C.RESET}", end="", flush=True)

        try:
            result = process_image(img_path, out_dir, save_debug=save_debug)
            shape_color = {
                "Hourglass":         C.GREEN,
                "Spoon":             C.YELLOW,
                "Triangle":          C.YELLOW,
                "Inverted Triangle": C.CYAN,
                "Rectangle":         C.WHITE,
                "Unknown":           C.RED,
            }.get(result["shape"], C.WHITE)

            print(
                f"{shape_color}{result['shape']:<22}{C.RESET}"
                f"{C.GRAY}({result['elapsed_s']}s){C.RESET}"
            )
            results.append(result)

        except Exception as e:
            print(f"{C.RED}THAT BAI{C.RESET}")
            print(f"  {C.RED}{e}{C.RESET}")
            if "--verbose" in sys.argv or "-v" in sys.argv:
                traceback.print_exc()
            results.append({
                "file":    img_path.name,
                "error":   str(e),
                "out_dir": str(out_dir),
            })

    # Tổng hợp
    print_summary(results)
    save_summary_json(results, output_dir / "summary.json")

    print(f"{C.BOLD}Output dir: {output_dir.resolve()}{C.RESET}\n")


if __name__ == "__main__":
    main()