import os
import threading
import numpy as np
import cv2
from scipy.spatial import distance as dist
from logger import log

def eye_aspect_ratio(eye: np.ndarray) -> float:
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def average_ear(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

def trigger_alarm(sound_path: str) -> None:
    def _play():
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception:
            try:
                import winsound
                from config import ALARM_FREQUENCY, ALARM_DURATION
                winsound.Beep(ALARM_FREQUENCY, ALARM_DURATION)
            except Exception:
                log.warning("Could not play alarm. Check pygame / winsound installation.")

    t = threading.Thread(target=_play, daemon=True)
    t.start()

def draw_text(frame: np.ndarray, text: str, pos: tuple,
              color=(0, 255, 0), scale: float = 0.65,
              thickness: int = 2) -> None:
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_eye_contour(frame: np.ndarray, eye_pts: np.ndarray,
                      color=(0, 255, 0)) -> None:
    hull = cv2.convexHull(eye_pts)
    cv2.drawContours(frame, [hull], -1, color, 1)

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def get_video_writer(path: str, fps: float, width: int,
                     height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    log.info(f"VideoWriter created → {path} ({width}×{height} @ {fps:.1f} fps)")
    return writer