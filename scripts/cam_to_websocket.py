import json
import time
import traceback

import cv2
import websocket

WS_URL = "ws://localhost:8080/ingest"
DEVICE_ID = "webcam-test-1"
FPS = 5
JPEG_QUALITY = 60
WIDTH = 640
HEIGHT = 480


def main():
    ws = None
    cap = None
    sent_count = 0

    try:
        print(f"Connecting to websocket: {WS_URL}")
        ws = websocket.create_connection(WS_URL, timeout=5)
        print("Websocket connected")

        ws.send(json.dumps({
            "type": "hello",
            "deviceId": DEVICE_ID
        }))
        print(f"Registered deviceId={DEVICE_ID}")

        print("Opening webcam...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("Could not open webcam with CAP_DSHOW, trying default backend...")
            cap.release()
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Force smaller capture size if supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        delay = 1.0 / FPS
        print(f"Streaming webcam as {DEVICE_ID} to {WS_URL}")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read webcam frame")
                time.sleep(0.1)
                continue

            # Resize again just to be sure
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                print("Failed to encode frame as JPEG")
                continue

            payload = jpg.tobytes()
            print(f"Sending frame {sent_count + 1}, bytes={len(payload)}")

            try:
                ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)
                sent_count += 1
                print(f"Sent frame {sent_count}")
            except Exception as e:
                print(f"Failed to send frame {sent_count + 1}: {e}")
                break

            time.sleep(delay)

    except KeyboardInterrupt:
        print("Stopping webcam sender")
    except Exception as e:
        print("cam_to_websocket.py failed:")
        print(e)
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()