# bodyshape/api_server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile, shutil, json, base64, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "service": "bodyshape"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    from pose_estimator    import detect_pose
    from bg_remover        import remove_background, alpha_to_binary_mask
    from body_measurements import load_keypoints, estimate_measurements
    from bodyshape_classifier import classify
    from visualize         import draw_measurements

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        img_path = tmp.name

    kp_json    = img_path + "_kp.json"
    bg_removed = img_path + "_bg.png"
    out_img    = img_path + "_result.png"

    keypoints, result, mp_image = detect_pose(img_path, config.MODEL_PATH)
    with open(kp_json, "w") as f:
        json.dump(keypoints, f)

    bgra = remove_background(img_path, output_path=bg_removed)
    mask = alpha_to_binary_mask(bgra)
    kp   = load_keypoints(kp_json)
    measurements, lines = estimate_measurements(kp, mask)
    shape = classify(
        measurements["shoulder_px"],
        measurements["waist_px"],
        measurements["hip_px"],
    )
    draw_measurements(bg_removed, measurements, lines, out_img, kp=kp)

    with open(out_img, "rb") as f:
        img_bytes = f.read()

    return {
        "body_shape": shape,
        "result_image_base64": base64.b64encode(img_bytes).decode()
    }