"""Driver Drowsiness Detection
Detects closed eyes and yawning with MediaPipe Face Mesh.
When the driver is drowsy/sleeping:
  - the console prints "ALARM RINGING"
  - the video window draws a RED box around the face and shows "DROWSY"

Run:  python drowsiness_detector.py        (press 'q' to quit)
"""
import os
import platform
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------- Settings -----------------------------
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
DROWSY_SECONDS = float(os.getenv("DROWSY_SECONDS", 2.0))   # eyes closed -> DROWSY
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", 4.0))     # eyes closed -> SLEEPING
YAWN_MAR = float(os.getenv("YAWN_MAR", 0.6))               # mouth ratio for a yawn
YAWNS_FOR_ALERT = int(os.getenv("YAWNS_FOR_ALERT", 3))     # yawns within 60 s
CALIBRATION_SECONDS = 3.0

# MediaPipe Face Mesh landmark indices: [outer corner, top1, top2, inner corner, bottom2, bottom1]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = {"top": 13, "bottom": 14, "left": 78, "right": 308}

RED = (0, 0, 255)
GREEN = (0, 200, 0)
YELLOW = (0, 200, 255)
GRAY = (128, 128, 128)


# ----------------------------- Helpers -----------------------------
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)


def mouth_aspect_ratio(lm):
    return dist(lm["top"], lm["bottom"]) / (dist(lm["left"], lm["right"]) + 1e-6)


def ring_alarm(level):
    """Print the alarm message to the console (and beep if the OS supports it)."""
    print(f"[{time.strftime('%H:%M:%S')}] *** ALARM RINGING *** Driver is {level}!", flush=True)

    def _beep():
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(2500 if level == "SLEEPING" else 1800, 400)
        else:
            print("\a", end="", flush=True)

    threading.Thread(target=_beep, daemon=True).start()


# ----------------------------- Main loop -----------------------------
def main():
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Try CAMERA_INDEX=1")

    start = time.time()
    calib_ears = []
    ear_threshold = 0.21            # replaced after calibration
    closed_since = None
    yawn_times = deque()
    yawning = False
    last_alarm = 0.0
    last_yawn_msg = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        status, color = "ALERT", GREEN
        box = None

        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            px = lambda i: (lms[i].x * w, lms[i].y * h)

            # Face bounding box from all landmarks
            xs = [p.x * w for p in lms]
            ys = [p.y * h for p in lms]
            pad = 15
            box = (int(max(min(xs) - pad, 0)), int(max(min(ys) - pad, 0)),
                   int(min(max(xs) + pad, w - 1)), int(min(max(ys) + pad, h - 1)))

            ear = (eye_aspect_ratio([px(i) for i in RIGHT_EYE]) +
                   eye_aspect_ratio([px(i) for i in LEFT_EYE])) / 2.0
            mar = mouth_aspect_ratio({k: px(v) for k, v in MOUTH.items()})

            # --- Calibration: learn this driver's open-eye EAR ---
            if now - start < CALIBRATION_SECONDS:
                calib_ears.append(ear)
                cv2.rectangle(frame, box[:2], box[2:], GREEN, 2)
                cv2.putText(frame, "Calibrating - keep eyes open...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
                cv2.imshow("Driver Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            if calib_ears:
                ear_threshold = float(np.clip(np.median(calib_ears) * 0.75, 0.15, 0.25))
                print(f"[calibration] EAR threshold = {ear_threshold:.3f}")
                calib_ears = []

            # --- Eye closure timing ---
            if ear < ear_threshold:
                closed_since = closed_since or now
            else:
                closed_since = None
            closed_for = (now - closed_since) if closed_since else 0.0

            # --- Yawn counting ---
            if mar > YAWN_MAR and not yawning:
                yawning = True
                yawn_times.append(now)
            elif mar <= YAWN_MAR:
                yawning = False
            while yawn_times and now - yawn_times[0] > 60:
                yawn_times.popleft()

            # --- Decide status ---
            if closed_for >= SLEEP_SECONDS:
                status, color = "SLEEPING", RED
            elif closed_for >= DROWSY_SECONDS:
                status, color = "DROWSY", RED
            elif len(yawn_times) >= YAWNS_FOR_ALERT:
                status, color = "YAWNING", YELLOW

            # --- Console alarm (repeats once per second while drowsy) ---
            if status in ("DROWSY", "SLEEPING"):
                if now - last_alarm > 1.0:
                    ring_alarm(status)
                    last_alarm = now
            elif status == "YAWNING" and now - last_yawn_msg > 10:
                print(f"[{time.strftime('%H:%M:%S')}] Notice: driver is yawning frequently.", flush=True)
                last_yawn_msg = now

            cv2.putText(frame, f"EAR: {ear:.2f} (thr {ear_threshold:.2f})", (20, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}  Yawns/min: {len(yawn_times)}", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            closed_since = None
            status, color = "NO FACE", GRAY

        # --- Draw box and label on the video window ---
        if box:
            thickness = 4 if status in ("DROWSY", "SLEEPING") else 2
            cv2.rectangle(frame, box[:2], box[2:], color, thickness)
            if status in ("DROWSY", "SLEEPING"):
                label = status
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                y0 = max(box[1] - th - 14, 0)
                cv2.rectangle(frame, (box[0], y0), (box[0] + tw + 12, y0 + th + 14), RED, -1)
                cv2.putText(frame, label, (box[0] + 6, y0 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        cv2.putText(frame, f"STATUS: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()