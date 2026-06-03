# api_bodyshape.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2, numpy as np, base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    data = await image.read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    
    # Gọi model của bạn ở đây
    # result = your_bodyshape_model.predict(bgr)
    
    # Ví dụ response — thay bằng output thực của model
    return JSONResponse({
        "body_shape": "RECTANGLE",        # HOURGLASS|PEAR|APPLE|RECTANGLE|INVERTED_TRIANGLE
        "measurements": {
            "shoulder": 84,
            "waist": 62,
            "hip": 90
        },
        "confidence": 0.87,
        "annotated_image_b64": ""         # base64 ảnh có annotation nếu có
    })