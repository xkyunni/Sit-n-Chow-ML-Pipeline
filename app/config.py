import os

WEBSOCKET_BASE_URL = os.getenv("WEBSOCKET_BASE_URL", "https://sit-n-chow-ws-5jph4zpsja-uc.a.run.app")
FIREBASE_DB_URL = os.getenv("FIREBASE_DATABASE_URL", "")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT", "")

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "app/models/yolo/yolo11n.pt")
POSTURE_MODEL_PATH = os.getenv("POSTURE_MODEL_PATH", "app/models/posture/posture_classifier.pth")

POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "0.5"))
SIT_CONF_THRESHOLD = float(os.getenv("SIT_CONF_THRESHOLD", "0.85"))
REQUIRED_CONSECUTIVE_SIT_FRAMES = int(os.getenv("REQUIRED_CONSECUTIVE_SIT_FRAMES", "5"))