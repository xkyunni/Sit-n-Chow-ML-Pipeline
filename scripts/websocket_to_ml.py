import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import argparse

import cv2
import requests
import numpy as np

from app.inference.detection import DogDetector
from app.inference.classification import PostureClassifier
from app.inference.posture_pipeline import InferencePipeline

# get latest jpeg frame for 1 device from websocket
def fetch_frame(base_url, device_id):
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/view",
            params={"device": device_id},
            timeout=5
        )

        if resp.status_code != 200:
            return None

        img_arr = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return frame

    except Exception as e:
        print("Fetch error:", e)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--device-id", default=None)

    repo_root = Path(__file__).resolve().parent.parent
    app_root = repo_root / "app"
    default_yolo = app_root / "models" / "yolo" / "yolo11n.pt"
    default_classifier = app_root / "models" / "posture" / "posture_classifier.pth"

    parser.add_argument("--yolo", default=str(default_yolo))
    parser.add_argument("--classifier", default=str(default_classifier))
    parser.add_argument("--success-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)

    args = parser.parse_args()

    print("Loading models...")
    print("YOLO path:", args.yolo)
    print("Classifier path:", args.classifier)

    detector = DogDetector(args.yolo)
    classifier = PostureClassifier(args.classifier)
    pipeline = InferencePipeline(
        detector,
        classifier,
        success_seconds=args.success_seconds,
        timeout_seconds=args.timeout_seconds,
    )

    print("Starting ML test loop...")

    while True:
        frame = fetch_frame(args.base_url, args.device_id)

        if frame is None:
            print("No frame yet...")
            time.sleep(0.2)
            continue

        result = pipeline.run(frame)

        # visualizing bboxes
        if result["dog_detected"]:
            x1, y1, x2, y2 = result["bbox"]
            posture = result["posture"]
            conf = result["confidence"]
            sit_prob = result["sit_probability"]
            det_conf = result.get("det_confidence", 0.0)
            sit_elapsed = result.get("sit_duration_sec", 0.0)

            sitting_now = posture == "sitting"
            color = (0, 255, 0) if sitting_now else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text_y = max(20, y1 - 45)
            cv2.putText(frame, f"dog_conf: {det_conf:.2f}", (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"posture: {posture}", (x1, text_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"posture_conf: {conf:.2f}", (x1, text_y + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"sit_prob: {sit_prob:.2f}", (x1, text_y + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"sit_time: {sit_elapsed:.2f}s", (x1, text_y + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "No dog detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        session_elapsed = result.get("session_duration_sec", 0.0)
        cv2.putText(frame, f"session_time: {session_elapsed:.1f}s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # show result of event and stop script
        if result["event"] is not None:
            print(result["event"])

            if result["event"]["type"] == "sit_success":
                cv2.putText(frame, "SIT SUCCESS (5s)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            elif result["event"]["type"] == "sit_timeout":
                cv2.putText(frame, "TIMEOUT (3 min)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            cv2.imshow("Websocket ML Test", frame)
            cv2.waitKey(1000)
            break

        cv2.imshow("Websocket ML Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()