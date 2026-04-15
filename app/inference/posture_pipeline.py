import time

class InferencePipeline:
    def __init__(
        self,
        detector,
        classifier,
        success_seconds: float = 5.0,
        timeout_seconds: float = 180.0,
    ):
        self.detector = detector
        self.classifier = classifier
        self.success_seconds = success_seconds
        self.timeout_seconds = timeout_seconds

        self.session_started_at = None
        self.sit_started_at = None
        self.finished = False
        self.final_event = None

    def reset(self):
        self.session_started_at = None
        self.sit_started_at = None
        self.finished = False
        self.final_event = None

    def run(self, frame):
        now = time.time()

        if self.session_started_at is None:
            self.session_started_at = now

        session_duration = now - self.session_started_at

        if self.finished:
            return {
                "dog_detected": False,
                "bbox": None,
                "det_confidence": 0.0,
                "posture": "finished",
                "confidence": 0.0,
                "sit_probability": 0.0,
                "sit_duration_sec": 0.0,
                "session_duration_sec": session_duration,
                "event": self.final_event,
            }

        detection = self.detector.detect(frame)

        if detection is None:
            self.sit_started_at = None

            event = None
            if session_duration >= self.timeout_seconds:
                event = {"type": "sit_timeout"}
                self.finished = True
                self.final_event = event

            return {
                "dog_detected": False,
                "bbox": None,
                "det_confidence": 0.0,
                "posture": "no_dog",
                "confidence": 0.0,
                "sit_probability": 0.0,
                "sit_duration_sec": 0.0,
                "session_duration_sec": session_duration,
                "event": event,
            }

        classification = self.classifier.predict(detection["crop"])
        posture = classification["posture"]

        if posture == "sitting":
            if self.sit_started_at is None:
                self.sit_started_at = now
            sit_duration = now - self.sit_started_at
        else:
            self.sit_started_at = None
            sit_duration = 0.0

        event = None

        if posture == "sitting" and sit_duration >= self.success_seconds:
            event = {"type": "sit_success"}
            self.finished = True
            self.final_event = event
        elif session_duration >= self.timeout_seconds:
            event = {"type": "sit_timeout"}
            self.finished = True
            self.final_event = event

        return {
            "dog_detected": True,
            "bbox": detection["bbox"],
            "det_confidence": detection["det_confidence"],
            "posture": posture,
            "confidence": classification["confidence"],
            "sit_probability": classification["sit_probability"],
            "sit_duration_sec": sit_duration,
            "session_duration_sec": session_duration,
            "event": event,
        }