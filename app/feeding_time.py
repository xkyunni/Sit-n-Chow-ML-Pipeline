import time

from config import YOLO_MODEL_PATH, POSTURE_MODEL_PATH
from app.services.frame_client import FrameClient
from app.services.firebase_client import FirebaseClient
from app.inference.detection import DogDetector
from app.inference.classification import PostureClassifier
from app.inference.posture_pipeline import InferencePipeline

POLL_INTERVAL_SEC = 0.2

class DeviceSession:
    def __init__(self):
        detector = DogDetector(YOLO_MODEL_PATH)
        classifier = PostureClassifier(POSTURE_MODEL_PATH)

        self.pipeline = InferencePipeline(
            detector=detector,
            classifier=classifier,
            success_seconds=5.0,
            timeout_seconds=180.0,
        )
        self.last_ts = None
        self.done = False

def main():
        frame_client = FrameClient()
        firebase = FirebaseClient()
        sessions = {}

        while True:
            try:
                device_ids = firebase.get_registered_devices()

                for device_id in device_ids:
                    if device_id not in sessions:
                        sessions[device_id] = DeviceSession()

                    session = sessions[device_id]

                    if session.done:
                        continue

                    frame, ts = frame_client.fetch_latest_frame(device_id)

                    if frame is None:
                        continue

                    if session.last_ts is not None and ts == session.last_ts:
                        continue

                    session.last_ts = ts

                    result = session.pipeline.run(frame)
                    print([
                        f"device: {device_id}",
                        f"posture: {result['posture']}",
                        f"sit_duration_sec: {result['sit_duration_sec']:.2f}",
                        f"session_duration_sec: {result['session_duration_sec']:.2f}",
                        f"event: {result['event']}"
                    ])
                    result["updatedAt"] = int(time.time() * 1000)

                    firebase.write_ml_result(device_id, result)

                    event = result.get("event")
                    if event and event["type"] in ("sit_success", "sit_timeout"):
                        print(f"[{device_id}] final outcome: {event['type']}")
                        session.done = True
                        return

                time.sleep(POLL_INTERVAL_SEC)

            except Exception as e:
                print("Orchestrator loop error:", e)
                time.sleep(1.0)

if __name__ == "__main__":
    main()