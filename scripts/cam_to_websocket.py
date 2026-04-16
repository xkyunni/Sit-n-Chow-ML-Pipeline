import os
import cv2
import json
import time
import argparse
import traceback
import websocket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ws-url",
        default=os.getenv("WS_URL", "ws://localhost:8080/ingest"),
        help="WebSocket ingest URL",
    )
    parser.add_argument(
        "--device-id",
        default=os.getenv("DEVICE_ID", "webcam-test-1"),
        help="Device ID to register with the websocket service",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality (0-100)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second to send",
    )
    args = parser.parse_args()

    ws_url = args.ws_url
    device_id = args.device_id
    frame_delay = 1.0 / args.fps if args.fps > 0 else 0.1

    print(f"Connecting to websocket: {ws_url}")
    ws = websocket.create_connection(ws_url, timeout=10)

    hello = {
        "type": "hello",
        "deviceId": device_id,
    }
    ws.send(json.dumps(hello))
    print(f"Registered deviceId: {device_id}")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera_index}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read webcam frame")
                time.sleep(0.1)
                continue

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
            )
            if not ok:
                print("Failed to encode frame as JPEG")
                time.sleep(0.1)
                continue

            ws.send_binary(encoded.tobytes())
            print(
                json.dumps(
                    {
                        "deviceId": device_id,
                        "bytes_sent": int(len(encoded)),
                        "timestamp_ms": int(time.time() * 1000),
                    },
                    indent=2,
                )
            )

            time.sleep(frame_delay)

    finally:
        cap.release()
        ws.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("cam_to_websocket.py failed:")
        traceback.print_exc()