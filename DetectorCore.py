"""Detection core: takes a BGR frame, returns (annotated_frame, info dict)."""
import os
import platform
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
DROWSY_SECONDS = float(os.getenv("DROWSY_SECONDS", 2.0))
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", 4.0))
YAWN_MAR = float(os.getenv("YAWN_MAR", 0.6))
YAWNS_FOR_ALERT = int(os.getenv("YAWNS_FOR_ALERT", 3))
CALIBRATION_SECONDS = 3.0

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = {"top": 13, "bottom": 14, "left": 78, "right": 308}

RED, GREEN, YELLOW, GRAY = (0, 0, 255), (0, 200, 0), (0, 200, 255), (128, 128, 128)


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)


def mouth_aspect_ratio(lm):
    return dist(lm["top"], lm["bottom"]) / (dist(lm["left"], lm["right"]) + 1e-6)


def ring_alarm(level):
    print(f"[{time.strftime('%H:%M:%S')}] *** ALARM RINGING *** Driver is {level}!", flush=True)

    def _beep():
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(2500 if level == "SLEEPING" else 1800, 400)
        else:
            print("\a", end="", flush=True)

    threading.Thread(target=_beep, daemon=True).start()


class DrowsinessDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.start = time.time()
        self.calib_ears = []
        self.ear_threshold = 0.21
        self.calibrated = False
        self.closed_since = None
        self.yawn_times = deque()
        self.yawning = False
        self.last_alarm = 0.0
        self.last_yawn_msg = 0.0

    def process(self, frame):
        h, w = frame.shape[:2]
        now = time.time()
        info = {"status": "NO FACE", "ear": 0.0, "mar": 0.0, "yawns": 0, "closed_for": 0.0}
        status, color, box = "NO FACE", GRAY, None

        result = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            px = lambda i: (lms[i].x * w, lms[i].y * h)
            xs, ys = [p.x * w for p in lms], [p.y * h for p in lms]
            pad = 15
            box = (int(max(min(xs) - pad, 0)), int(max(min(ys) - pad, 0)),
                   int(min(max(xs) + pad, w - 1)), int(min(max(ys) + pad, h - 1)))

            ear = (eye_aspect_ratio([px(i) for i in RIGHT_EYE]) +
                   eye_aspect_ratio([px(i) for i in LEFT_EYE])) / 2.0
            mar = mouth_aspect_ratio({k: px(v) for k, v in MOUTH.items()})

            if not self.calibrated:
                if now - self.start < CALIBRATION_SECONDS:
                    self.calib_ears.append(ear)
                    status, color = "CALIBRATING", YELLOW
                    info.update(status=status, ear=float(ear), mar=float(mar))
                    return self._draw(frame, box, status, color), info
                if self.calib_ears:
                    self.ear_threshold = float(np.clip(np.median(self.calib_ears) * 0.75, 0.15, 0.25))
                    print(f"[calibration] EAR threshold = {self.ear_threshold:.3f}")
                self.calibrated = True

            if ear < self.ear_threshold:
                self.closed_since = self.closed_since or now
            else:
                self.closed_since = None
            closed_for = (now - self.closed_since) if self.closed_since else 0.0

            if mar > YAWN_MAR and not self.yawning:
                self.yawning = True
                self.yawn_times.append(now)
            elif mar <= YAWN_MAR:
                self.yawning = False
            while self.yawn_times and now - self.yawn_times[0] > 60:
                self.yawn_times.popleft()

            status, color = "ALERT", GREEN
            if closed_for >= SLEEP_SECONDS:
                status, color = "SLEEPING", RED
            elif closed_for >= DROWSY_SECONDS:
                status, color = "DROWSY", RED
            elif len(self.yawn_times) >= YAWNS_FOR_ALERT:
                status, color = "YAWNING", YELLOW

            if status in ("DROWSY", "SLEEPING"):
                if now - self.last_alarm > 1.0:
                    ring_alarm(status)
                    self.last_alarm = now
            elif status == "YAWNING" and now - self.last_yawn_msg > 10:
                print(f"[{time.strftime('%H:%M:%S')}] Notice: driver is yawning frequently.", flush=True)
                self.last_yawn_msg = now

            info.update(status=status, ear=float(ear), mar=float(mar),
                        yawns=len(self.yawn_times), closed_for=round(closed_for, 1))
        else:
            self.closed_since = None
            info["status"] = status

        return self._draw(frame, box, status, color), info

    @staticmethod
    def _draw(frame, box, status, color):
        if box:
            bad = status in ("DROWSY", "SLEEPING")
            cv2.rectangle(frame, box[:2], box[2:], color, 4 if bad else 2)
            if bad:
                (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                y0 = max(box[1] - th - 14, 0)
                cv2.rectangle(frame, (box[0], y0), (box[0] + tw + 12, y0 + th + 14), RED, -1)
                cv2.putText(frame, status, (box[0] + 6, y0 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return frame