import time
import numpy as np
import cv2

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from config import YOLO_MODEL_PATH, POSTURE_MODEL_PATH
from app.inference.detection import DogDetector
from app.inference.classification import PostureClassifier

app = FastAPI(title="Sit-n-Chow ML Service")

detector = DogDetector(YOLO_MODEL_PATH)
classifier = PostureClassifier(POSTURE_MODEL_PATH)


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

        detection = detector.detect(frame)

        if detection is None:
            result = {
                "dog_detected": False,
                "posture": "no_dog",
            }
            return JSONResponse(content=result)

        classification = classifier.predict(detection["crop"])

        result = {
            "dog_detected": True,
            "bbox": detection["bbox"],
            "det_confidence": detection["det_confidence"],
            "posture": classification["posture"],
            "confidence": classification["confidence"],
            "sit_probability": classification["sit_probability"],
            "updatedAt": int(time.time() * 1000),
        }

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")