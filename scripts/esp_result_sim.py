import time
import firebase_admin
from firebase_admin import credentials, db

from config import FIREBASE_DB_URL, FIREBASE_SERVICE_ACCOUNT_JSON

DEVICE_ID = "webcam-test-1"

def main():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_key.json")
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

    last_handled = None
    print(f"Watching ML output for {DEVICE_ID}")

    while True:
        snap = db.reference(f"devices/{DEVICE_ID}/ml").get() or {}
        event = snap.get("event")
        updated_at = snap.get("updatedAt")

        if event and updated_at and updated_at != last_handled:
            event_type = event.get("type")

            if event_type == "sit_success":
                print("ESP SIM: DISPENSE FOOD")
                print("ESP SIM: STOP CAMERA")
                last_handled = updated_at

            elif event_type == "sit_timeout":
                print("ESP SIM: DISPENSE FOOD")
                print("ESP SIM: STOP CAMERA")
                last_handled = updated_at

        time.sleep(0.5)

if __name__ == "__main__":
    main()