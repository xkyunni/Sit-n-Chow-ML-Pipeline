import os

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "app/models/yolo/yolo11n.pt")
POSTURE_MODEL_PATH = os.getenv("POSTURE_MODEL_PATH", "app/models/posture/posture_classifier.pth")

WEBSOCKET_BASE_URL = os.getenv("WEBSOCKET_BASE_URL", "http://localhost:8080")

FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL", "https://sit-n-chow-v1-default-rtdb.firebaseio.com")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")