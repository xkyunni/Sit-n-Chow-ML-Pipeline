import os

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "app/models/yolo/yolo11n.pt")
POSTURE_MODEL_PATH = os.getenv("POSTURE_MODEL_PATH", "app/models/posture/posture_classifier.pth")