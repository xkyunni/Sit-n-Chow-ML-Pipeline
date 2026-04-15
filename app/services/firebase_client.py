import json
import firebase_admin
from firebase_admin import credentials, db

from config import FIREBASE_DB_URL, FIREBASE_SERVICE_ACCOUNT_JSON

class FirebaseClient:
    def __init__(self):
        if not FIREBASE_DB_URL:
            raise ValueError("FIREBASE_DB_URL is not set")

        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase_key.json")
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