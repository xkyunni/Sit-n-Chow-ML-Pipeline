import time

# combine detection and classification
class InferencePipeline:
    def __init__(self, detector, classifier, success_seconds=5.0, timeout_seconds=180.0):
        self.detector = detector
        self.classifier = classifier
        
        self.success_seconds = success_seconds
        self.timeout_seconds = timeout_seconds

        self.session_start = None
        self.sit_start_time = None
        self.sit_confidences = []
        self.session_active = False

    def start_session(self):
        self.session_start = time.time()
        self.sit_start_time = None
        self.sit_confidences = []
        self.session_active = True

    def reset_session(self):
        self.session_start = None
        self.sit_start_time = None
        self.sit_confidences = []
        self.session_active = False

    # run 1 frame and return latest state
    def run(self, frame):
        now = time.time()

        # autostart when frames begin
        if not self.session_active or self.session_start is None:
            self.start_session()

        detection = self.detector.detect(frame)

        session_elapsed = now - self.session_start
        event = None

        if detection is None:
            # if sitting is not continuous for 5s, timeout at 3min
            if self.session_active and session_elapsed >= self.timeout_seconds:
                event = {
                    "type": "sit_timeout",
                    "success": False,
                    "posture": "not_successful",
                    "session_duration_sec": round(session_elapsed, 2),
                    "avg_confidence": 0.0,
                    "triggered": False,
                    "timestamp": round(now, 3),
                }
                self.session_active = False

            self.sit_start_time = None
            self.sit_confidences = []

            return {
                "dog_detected": False,
                "bbox": None,
                "posture": "no_dog",
                "confidence": 0.0,
                "sit_probability": 0.0,
                "sit_duration_sec": 0.0,
                "session_duration_sec": round(session_elapsed, 2),
                "event": event,
            }

        # if dog is detected, classify cropped img
        classification = self.classifier.predict(detection["crop"])

        posture = classification["posture"]
        confidence = classification["confidence"]
        sit_probability = classification["sit_probability"]

        sitting_now = posture == "sitting"

        if sitting_now:
            # track sit confidence to get the avg
            self.sit_confidences.append(sit_probability)

            if self.sit_start_time is None:
                self.sit_start_time = now

            sit_elapsed = now - self.sit_start_time
        else:
            # any frame not sitting resets 5s timer
            self.sit_start_time = None
            self.sit_confidences = []
            sit_elapsed = 0.0

        if self.session_active:
            if sit_elapsed >= self.success_seconds:
                avg_conf = (
                    sum(self.sit_confidences) / len(self.sit_confidences)
                    if self.sit_confidences else 0.0
                )

                event = {
                    "type": "sit_success",
                    "success": True,
                    "posture": "sitting",
                    "sit_duration_sec": round(sit_elapsed, 2),
                    "avg_confidence": round(avg_conf, 4),
                    "frames_in_sit_window": len(self.sit_confidences),
                    "triggered": True,
                    "timestamp": round(now, 3),
                }

                self.session_active = False

            elif session_elapsed >= self.timeout_seconds:
                avg_conf = (
                    sum(self.sit_confidences) / len(self.sit_confidences)
                    if self.sit_confidences else 0.0
                )

                event = {
                    "type": "sit_timeout",
                    "success": False,
                    "posture": "not_successful",
                    "session_duration_sec": round(session_elapsed, 2),
                    "avg_confidence": round(avg_conf, 4),
                    "triggered": False,
                    "timestamp": round(now, 3),
                }

                self.session_active = False

        return {
            "dog_detected": True,
            "bbox": detection["bbox"],
            "det_confidence": detection["det_confidence"],
            "posture": posture,
            "confidence": confidence,
            "sit_probability": sit_probability,
            "sit_duration_sec": round(sit_elapsed, 2),
            "session_duration_sec": round(session_elapsed, 2),
            "event": event,
        }