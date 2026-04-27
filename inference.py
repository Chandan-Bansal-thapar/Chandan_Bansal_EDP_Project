import time
import cv2
import numpy as np

from preprocessing import preprocess_frame, extract_eye_landmarks
from utils import average_ear, draw_text, draw_eye_contour, trigger_alarm
from config import (EAR_THRESHOLD, EAR_CONSEC_FRAMES,
                    ALARM_SOUND_PATH, SHOW_LANDMARKS, FONT_SCALE)
from logger import log

def run_inference(frame: np.ndarray,
                  detector,
                  predictor,
                  state: dict) -> tuple:
    t_start = time.perf_counter()

    display = cv2.resize(frame, (640, 480))
    gray    = preprocess_frame(frame)

    faces = extract_eye_landmarks(gray, detector, predictor)

    ear             = None
    eye_closed      = False
    alarm_triggered = False

    if faces:
        face = faces[0]
        ear  = average_ear(face["left_eye"], face["right_eye"])

        eye_closed = ear < EAR_THRESHOLD

        if eye_closed:
            state["counter"] += 1
            log.debug(f"Eyes closed | EAR={ear:.4f} | counter={state['counter']}")
        else:
            if state["alarm_on"]:
                log.info("Eyes re-opened. Alarm reset.")
            state["counter"] = 0
            state["alarm_on"] = False

        if state["counter"] >= EAR_CONSEC_FRAMES:
            if not state["alarm_on"]:
                state["alarm_on"]     = True
                state["total_alarms"] += 1
                alarm_triggered        = True
                log.warning(
                    f"DROWSINESS ALERT! EAR={ear:.4f} "
                    f"closed for {state['counter']} frames. "
                    f"Alarm #{state['total_alarms']} triggered."
                )
                trigger_alarm(ALARM_SOUND_PATH)

        if SHOW_LANDMARKS:
            draw_eye_contour(display, face["left_eye"],  color=(0, 255, 0))
            draw_eye_contour(display, face["right_eye"], color=(0, 255, 0))

    t_end          = time.perf_counter()
    inference_ms   = (t_end - t_start) * 1000.0

    now            = time.perf_counter()
    elapsed        = now - state["prev_time"]
    fps            = 1.0 / elapsed if elapsed > 0 else 0.0
    state["prev_time"] = now

    _annotate_frame(display, ear, eye_closed, state, fps, inference_ms)

    metadata = {
        "ear"             : ear,
        "eye_closed"      : eye_closed,
        "alarm_triggered" : alarm_triggered,
        "fps"             : fps,
        "inference_ms"    : inference_ms,
        "faces_detected"  : len(faces),
    }

    return display, metadata

def _annotate_frame(frame: np.ndarray,
                    ear,
                    eye_closed: bool,
                    state: dict,
                    fps: float,
                    inference_ms: float) -> None:
    h, w = frame.shape[:2]

    if ear is not None:
        ear_text = f"EAR: {ear:.3f}"
        color    = (0, 255, 0) if not eye_closed else (0, 0, 255)
        draw_text(frame, ear_text, (10, 30), color=color, scale=FONT_SCALE)

        status = "Eyes: OPEN" if not eye_closed else "Eyes: CLOSED"
        draw_text(frame, status, (10, 60), color=color, scale=FONT_SCALE)

        draw_text(frame, f"Closed Frames: {state['counter']}/{EAR_CONSEC_FRAMES}",
                  (10, 90), color=(200, 200, 0), scale=FONT_SCALE)
    else:
        draw_text(frame, "No Face Detected", (10, 30),
                  color=(100, 100, 255), scale=FONT_SCALE)

    if state.get("alarm_on"):
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 180), -1)
        draw_text(frame, "!!! DROWSINESS ALERT — WAKE UP !!!",
                  (10, h - 20), color=(255, 255, 255), scale=0.75, thickness=2)

    fps_text = f"FPS: {fps:.1f}"
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)[0]
    draw_text(frame, fps_text, (w - fps_size[0] - 10, 30),
              color=(255, 255, 0), scale=FONT_SCALE)

    inf_text = f"Inf: {inference_ms:.1f} ms"
    inf_size = cv2.getTextSize(inf_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)[0]
    draw_text(frame, inf_text, (w - inf_size[0] - 10, 60),
              color=(255, 200, 0), scale=FONT_SCALE)

    draw_text(frame, f"Alarms: {state['total_alarms']}",
              (w - 120, 90), color=(0, 180, 255), scale=FONT_SCALE)

def init_state() -> dict:
    return {
        "counter"      : 0,
        "alarm_on"     : False,
        "total_alarms" : 0,
        "prev_time"    : time.perf_counter(),
    }