class InferencePipeline:
    def __init__(self, detector, classifier):
        self.detector = detector
        self.classifier = classifier

    def run(self, frame):
        detection = self.detector.detect(frame)

        if detection is None:
            return {
                "dog_detected": False,
                "bbox": None,
                "det_confidence": 0.0,
                "posture": "no_dog",
                "confidence": 0.0,
                "sit_probability": 0.0,
            }

        classification = self.classifier.predict(detection["crop"])

        return {
            "dog_detected": True,
            "bbox": detection["bbox"],
            "det_confidence": detection["det_confidence"],
            "posture": classification["posture"],
            "confidence": classification["confidence"],
            "sit_probability": classification["sit_probability"],
        }