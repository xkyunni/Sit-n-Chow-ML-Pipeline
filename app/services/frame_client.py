import requests
import numpy as np
import cv2

from app.config import WEBSOCKET_BASE_URL

class FrameClient:
    def __init__(self, base_url: str = WEBSOCKET_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def fetch_latest_frame(self, device_id: str):
        url = f"{self.base_url}/view"

        try:
            resp = requests.get(
                url,
                params={"device": device_id},
                timeout=5
            )

            if resp.status_code != 200:
                return None, None

            # Convert timestamp safely
            ts_raw = resp.headers.get("X-Timestamp", "0")
            try:
                ts = int(ts_raw)
            except (TypeError, ValueError):
                ts = 0

            # Decode JPEG → OpenCV image
            jpg_bytes = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)

            if frame is None:
                return None, None

            return frame, ts

        except Exception:
            return None, None