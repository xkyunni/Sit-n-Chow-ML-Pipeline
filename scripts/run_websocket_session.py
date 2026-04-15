import argparse
import time
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-url", default="http://localhost:8080")
    parser.add_argument("--device-id", default="webcam-test-1")
    parser.add_argument("--success-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    args = parser.parse_args()

    base = args.ml_url.rstrip("/")

    print("Starting ML session...")
    start = requests.post(
        f"{base}/sessions/{args.device_id}/start",
        params={
            "success_seconds": args.success_seconds,
            "timeout_seconds": args.timeout_seconds,
        },
        timeout=15,
    )
    start.raise_for_status()
    print(start.json())

    while True:
        resp = requests.post(
            f"{base}/sessions/{args.device_id}/process-latest",
            timeout=30,
        )
        resp.raise_for_status()
        status = resp.json()
        last = status.get("lastResult") or {}
        event = status.get("finalEvent") or (last.get("event") if isinstance(last, dict) else None)
        print({
            "processed": status.get("processed"),
            "reason": status.get("reason"),
            "posture": last.get("posture"),
            "sit_duration_sec": last.get("sit_duration_sec"),
            "session_duration_sec": last.get("session_duration_sec"),
            "event": event,
        })

        if event is not None:
            print("Session finished:", event)
            break

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
