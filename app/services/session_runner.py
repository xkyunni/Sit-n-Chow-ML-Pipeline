import time
from typing import Any

from app.services.frame_client import FrameClient

class SessionRunner:
    def __init__(self, session_manager, frame_client: FrameClient):
        self.session_manager = session_manager
        self.frame_client = frame_client

    def process_latest_frame(self, device_id: str) -> dict[str, Any]:
        frame, ts = self.frame_client.fetch_latest_frame(device_id)
        if frame is None:
            return self.session_manager.get_status(device_id) | {
                "processed": False,
                "reason": "no_frame_available",
                "updatedAt": int(time.time() * 1000),
            }

        status = self.session_manager.process_frame(device_id, frame, frame_ts=ts)
        return status | {"processed": True, "updatedAt": int(time.time() * 1000)}
