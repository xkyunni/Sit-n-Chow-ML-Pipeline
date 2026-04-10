import time
import threading
from fastapi import FastAPI

from app.config import (
    YOLO_MODEL_PATH,
    POSTURE_MODEL_PATH,
    POLL_INTERVAL_SEC,
    SIT_CONF_THRESHOLD,
    REQUIRED_CONSECUTIVE_SIT_FRAMES,
)
from app.services.frame_client import FrameClient
from app.services.firebase_client import FirebaseClient
from app.inference.detection import DogDetector
from app.inference.classification import PostureClassifier
from app.inference.posture_pipeline import InferencePipeline
from app.utils.device_state import DeviceStateTracker

app = FastAPI()

frame_client = FrameClient()
firebase_client = FirebaseClient()
detector = DogDetector(YOLO_MODEL_PATH)
classifier = PostureClassifier(POSTURE_MODEL_PATH)
pipeline = InferencePipeline(detector, classifier)
device_state = DeviceStateTracker()

# local check for Google Cloud Run
@app.get("/health")
def health():
    return {"status": "ok"}

# loop that gets frames, runs ML, and writes results
def inference_loop():
    while True:
        try:
            device_ids = firebase_client.get_registered_devices()

            for device_id in device_ids:
                frame, ts = frame_client.fetch_latest_frame(device_id)
                if frame is None:
                    continue

                result = pipeline.run(frame)
                result["updatedAt"] = int(time.time() * 1000)
                result["frameTs"] = ts

                firebase_client.write_ml_result(device_id, result)

                state = device_state.update(device_id, result["posture"])

                if (
                    result["posture"] == "sitting"
                    and result["confidence"] >= SIT_CONF_THRESHOLD
                    and state["sit_count"] >= REQUIRED_CONSECUTIVE_SIT_FRAMES
                    and not state["dispensed"]
                ):
                    firebase_client.write_dispense_command(device_id, grams=25)
                    device_state.mark_dispensed(device_id)

        except Exception as e:
            print("Inference loop error:", e)

        time.sleep(POLL_INTERVAL_SEC)

@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=inference_loop, daemon=True)
    thread.start()