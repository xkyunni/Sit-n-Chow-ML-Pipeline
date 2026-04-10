import json
import time
import firebase_admin
from firebase_admin import credentials, db

from app.config import FIREBASE_DB_URL, FIREBASE_SERVICE_ACCOUNT_JSON

class FirebaseClient:
    def __init__(self):
        if not FIREBASE_DB_URL:
            raise ValueError("FIREBASE_DATABASE_URL is not set")

        if not FIREBASE_SERVICE_ACCOUNT_JSON:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT is not set")

        if not firebase_admin._apps:
            service_account_info = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(
                cred,
                {"databaseURL": FIREBASE_DB_URL},
            )

    def get_registered_devices(self):
        snap = db.reference("devices").get() or {}
        device_ids = []

        for device_id, info in snap.items():
            if isinstance(info, dict) and info.get("ownerUid"):
                device_ids.append(device_id)

        return device_ids

    def write_ml_result(self, device_id: str, result: dict):
        db.reference(f"devices/{device_id}/ml").set(result)

    def write_dispense_command(self, device_id: str, grams: int = 25):
        db.reference(f"commands/{device_id}/pending").set({
            "id": f"ml-{device_id}-{int(time.time() * 1000)}",
            "action": "dispense",
            "grams": grams,
            "by": "ml-service",
            "ts": int(time.time() * 1000),
            "source": "ml-service",
        })