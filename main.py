import time
import numpy as np
import cv2

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from config import YOLO_MODEL_PATH, POSTURE_MODEL_PATH, WEBSOCKET_BASE_URL
from app.inference.detection import DogDetector
from app.inference.classification import PostureClassifier
from app.inference.posture_pipeline import InferencePipeline
from app.services.frame_client import FrameClient
from app.services.session_runner import SessionRunner
from app.session.session_manager import SessionManager

app = FastAPI(title="Sit-n-Chow ML Service")

detector = DogDetector(YOLO_MODEL_PATH)
classifier = PostureClassifier(POSTURE_MODEL_PATH)
pipeline = InferencePipeline(detector, classifier)
session_manager = SessionManager(detector, classifier)
frame_client = FrameClient(WEBSOCKET_BASE_URL)
session_runner = SessionRunner(session_manager, frame_client)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/sessions")
def list_sessions():
    return session_manager.list_status()

@app.post("/sessions/{device_id}/start")
def start_session(device_id: str, success_seconds: float = 5.0, timeout_seconds: float = 180.0):
    return session_manager.start_session(
        device_id=device_id,
        success_seconds=success_seconds,
        timeout_seconds=timeout_seconds,
        reset=True,
    )

@app.get("/sessions/{device_id}")
def get_session(device_id: str):
    return session_manager.get_status(device_id)

@app.delete("/sessions/{device_id}")
def stop_session(device_id: str):
    return session_manager.stop_session(device_id)

@app.post("/sessions/{device_id}/process-latest")
def process_latest(device_id: str):
    return session_runner.process_latest_frame(device_id)

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

@app.post("/predict-device-frame")
async def predict_device_frame(
    device_id: str = Form(...),
    frame_ts: int | None = Form(default=None),
    auto_start: bool = Form(default=True),
    file: UploadFile = File(...),
):
    try:
        image_bytes = await file.read()
        frame = decode_image_bytes(image_bytes)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        if auto_start and not session_manager.has_session(device_id):
            session_manager.start_session(device_id)

        result = session_manager.process_frame(device_id, frame, frame_ts=frame_ts)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
