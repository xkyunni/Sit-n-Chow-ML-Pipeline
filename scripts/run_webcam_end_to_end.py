import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

def stream_subprocess_output(proc, name):
    # best-effort non-blocking display through inherited stdout/stderr
    return proc

def wait_for_health(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url.rstrip('/')}/health", timeout=3)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end webcam feeder test")
    parser.add_argument("--ml-url", default="http://localhost:8000")
    parser.add_argument("--device-id", default="webcam-test-1")
    parser.add_argument("--success-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--start-webcam", action="store_true", help="Also start cam_to_websocket.py")
    parser.add_argument("--websocket-url", default="ws://localhost:8080/ingest")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    webcam_proc = None

    try:
        if not wait_for_health(args.ml_url):
            raise RuntimeError(
                f"ML service is not reachable at {args.ml_url}. Start uvicorn first."
            )

        if args.start_webcam:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["WS_URL"] = args.websocket_url
            env["DEVICE_ID"] = args.device_id
            env["FPS"] = str(args.fps)
            env["JPEG_QUALITY"] = str(args.jpeg_quality)
            env["WIDTH"] = str(args.width)
            env["HEIGHT"] = str(args.height)

            cam_script = ROOT / "scripts" / "cam_to_websocket.py"
            print(f"Starting webcam sender: {cam_script}")
            webcam_proc = subprocess.Popen(
                [sys.executable, str(cam_script)],
                cwd=str(ROOT),
                env=env,
            )
            time.sleep(2.0)

        base = args.ml_url.rstrip("/")
        print("\nStarting ML session...")
        start = requests.post(
            f"{base}/sessions/{args.device_id}/start",
            params={
                "success_seconds": args.success_seconds,
                "timeout_seconds": args.timeout_seconds,
            },
            timeout=15,
        )
        start.raise_for_status()
        print("Session:", start.json())
        print("\nWatching live results...\n")

        while True:
            resp = requests.post(
                f"{base}/sessions/{args.device_id}/process-latest",
                timeout=30,
            )
            resp.raise_for_status()
            status = resp.json()
            last = status.get("lastResult") or {}
            event = status.get("finalEvent") or (last.get("event") if isinstance(last, dict) else None)

            posture = last.get("posture")
            confidence = last.get("confidence")
            sit_duration = last.get("sit_duration_sec")
            session_duration = last.get("session_duration_sec")
            processed = status.get("processed")
            reason = status.get("reason")

            print(
                f"processed={processed} | reason={reason} | posture={posture} | "
                f"confidence={confidence} | sit_duration_sec={sit_duration} | "
                f"session_duration_sec={session_duration} | event={event}"
            )

            if event is not None:
                print("\nFinal result:")
                print(event)
                break

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nRunner failed: {e}")
        sys.exit(1)
    finally:
        if webcam_proc is not None and webcam_proc.poll() is None:
            print("Stopping webcam sender...")
            if os.name == "nt":
                webcam_proc.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(1)
                if webcam_proc.poll() is None:
                    webcam_proc.terminate()
            else:
                webcam_proc.terminate()
            try:
                webcam_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                webcam_proc.kill()


if __name__ == "__main__":
    main()
