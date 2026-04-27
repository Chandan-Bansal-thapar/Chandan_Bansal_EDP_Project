import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH    = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs",   "eye_detection.log")
OUTPUT_VIDEO  = os.path.join(BASE_DIR, "output", "output_recording.avi")

INPUT_SOURCE  = 0

EAR_THRESHOLD     = 0.20
EAR_CONSEC_FRAMES = 3

ALARM_SOUND_PATH = os.path.join(BASE_DIR, "assets", "alarm.wav")
ALARM_FREQUENCY  = 2500
ALARM_DURATION   = 1000

FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
SHOW_LANDMARKS = True
FONT_SCALE     = 0.65

LOG_LEVEL = "DEBUG"