import cv2
import numpy as np
import dlib
from imutils import face_utils
from config import FRAME_WIDTH, FRAME_HEIGHT, MODEL_PATH
from logger import log

(L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def load_detectors():
    log.info("Loading dlib HOG face detector...")
    detector  = dlib.get_frontal_face_detector()
    log.info(f"Loading shape predictor from: {MODEL_PATH}")
    predictor = dlib.shape_predictor(MODEL_PATH)
    log.info("Detectors loaded successfully.")
    return detector, predictor

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(gray)

    denoised  = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return denoised

def extract_eye_landmarks(gray: np.ndarray, detector, predictor):
    rects   = detector(gray, 1)
    results = []

    for rect in rects:
        shape    = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)

        left_eye  = shape_np[L_START:L_END]
        right_eye = shape_np[R_START:R_END]

        results.append({
            "rect"      : rect,
            "left_eye"  : left_eye,
            "right_eye" : right_eye,
            "shape"     : shape_np,
        })

    return results

def prepare_input_source(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    cap.set(cv2.CAP_PROP_CONTRAST,   150)
    cap.set(cv2.CAP_PROP_GAIN,        50)

    log.info(f"Video source opened -> {source}")
    return cap