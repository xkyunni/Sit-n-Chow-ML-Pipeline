import time
import numpy as np
import cv2

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from config import YOLO_MODEL_PATH, POSTURE_MODEL_PATH
from app.inference.detection import DogDetector
from app.inference.classification import PostureClassifier
from app.inference.posture_pipeline import InferencePipeline

app = FastAPI(title="Sit-n-Chow ML Service")

detector = DogDetector(YOLO_MODEL_PATH)
classifier = PostureClassifier(POSTURE_MODEL_PATH)
pipeline = InferencePipeline(detector, classifier)

@app.get("/health")
def health():
    return {"status": "ok"}

def decode_image_bytes(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        frame = decode_image_bytes(image_bytes)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result = pipeline.run(frame)
        result["updatedAt"] = int(time.time() * 1000)

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
