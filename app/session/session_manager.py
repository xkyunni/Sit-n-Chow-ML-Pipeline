import time
from dataclasses import dataclass, field
from typing import Any

from app.inference.posture_pipeline import InferencePipeline

@dataclass
class ManagedSession:
    device_id: str
    pipeline: InferencePipeline
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    active: bool = True
    last_frame_ts: int | None = None
    last_result: dict[str, Any] | None = None


class SessionManager:
    def __init__(self, detector, classifier):
        self.detector = detector
        self.classifier = classifier
        self.sessions: dict[str, ManagedSession] = {}

    def start_session(
        self,
        device_id: str,
        success_seconds: float = 5.0,
        timeout_seconds: float = 180.0,
        reset: bool = True,
    ) -> dict[str, Any]:
        if device_id in self.sessions and not reset:
            return self.get_status(device_id)

        pipeline = InferencePipeline(
            detector=self.detector,
            classifier=self.classifier,
            success_seconds=success_seconds,
            timeout_seconds=timeout_seconds,
        )

        self.sessions[device_id] = ManagedSession(
            device_id=device_id,
            pipeline=pipeline,
        )
        return self.get_status(device_id)

    def has_session(self, device_id: str) -> bool:
        return device_id in self.sessions

    def stop_session(self, device_id: str) -> dict[str, Any]:
        session = self.sessions.get(device_id)
        if session is None:
            return {
                "deviceId": device_id,
                "active": False,
                "message": "No active session",
            }

        session.active = False
        status = self.get_status(device_id)
        del self.sessions[device_id]
        return status

    def process_frame(self, device_id: str, frame, frame_ts: int | None = None) -> dict[str, Any]:
        if device_id not in self.sessions:
            self.start_session(device_id)

        session = self.sessions[device_id]

        if frame_ts is not None and session.last_frame_ts is not None and frame_ts == session.last_frame_ts:
            return self.get_status(device_id) | {"skipped": True, "reason": "duplicate_frame"}

        result = session.pipeline.run(frame)
        session.updated_at_ms = int(time.time() * 1000)
        session.last_frame_ts = frame_ts
        session.last_result = {
            **result,
            "updatedAt": session.updated_at_ms,
            "deviceId": device_id,
        }

        event = result.get("event")
        if event is not None and event.get("type") in {"sit_success", "sit_timeout"}:
            session.active = False

        return self.get_status(device_id)

    def get_status(self, device_id: str) -> dict[str, Any]:
        session = self.sessions.get(device_id)
        if session is None:
            return {
                "deviceId": device_id,
                "active": False,
                "startedAt": None,
                "lastFrameId": None,
                "lastInferenceAt": None,
                "sitStartedAt": None,
                "lastResult": None,
                "finalEvent": None,
            }

        pipeline = session.pipeline
        final_event = pipeline.final_event

        return {
            "deviceId": device_id,
            "active": session.active,
            "startedAt": session.created_at_ms,
            "lastFrameId": session.last_frame_ts,
            "lastInferenceAt": session.updated_at_ms if session.last_result else None,
            "sitStartedAt": int(pipeline.sit_started_at * 1000) if pipeline.sit_started_at else None,
            "lastResult": session.last_result,
            "finalEvent": final_event,
            "timeoutSeconds": pipeline.timeout_seconds,
            "successSeconds": pipeline.success_seconds,
        }

    def list_status(self) -> dict[str, dict[str, Any]]:
        return {device_id: self.get_status(device_id) for device_id in self.sessions}
