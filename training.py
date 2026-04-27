import os
import time
import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from preprocessing import load_detectors, preprocess_frame, extract_eye_landmarks
from utils import average_ear, ensure_dirs
from config import MODEL_PATH, BASE_DIR
from logger import log

CLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "ear_classifier.pkl")
SCALER_PATH     = os.path.join(BASE_DIR, "models", "ear_scaler.pkl")
GRAPH_DIR       = os.path.join(BASE_DIR, "output", "training_graphs")

def calibrate_ear(num_open_frames: int = 100,
                  num_closed_frames: int = 100) -> float:
    detector, predictor = load_detectors()
    cap = cv2.VideoCapture(0)
    open_ears, closed_ears = [], []

    for label, store, msg in [
        ("OPEN",   open_ears,   "Keep eyes OPEN  – press SPACE to start"),
        ("CLOSED", closed_ears, "Keep eyes CLOSED – press SPACE to start"),
    ]:
        target = num_open_frames if label == "OPEN" else num_closed_frames
        count  = 0
        log.info(msg)
        print(f"\n{msg}")
        print("Press SPACE to begin or Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray    = preprocess_frame(frame)
            faces   = extract_eye_landmarks(gray, detector, predictor)
            display = cv2.resize(frame, (640, 480))
            cv2.putText(display, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, f"Collected: {count}/{target}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cap.release(); cv2.destroyAllWindows()
                raise KeyboardInterrupt("Calibration aborted by user.")

            if key == ord(" ") or count > 0:
                for face in faces:
                    ear = average_ear(face["left_eye"], face["right_eye"])
                    store.append(ear)
                    count += 1

            if count >= target:
                break

    cap.release()
    cv2.destroyAllWindows()

    open_mean   = float(np.mean(open_ears))
    closed_mean = float(np.mean(closed_ears))
    threshold   = (open_mean + closed_mean) / 2.0

    log.info(f"Calibration → open mean EAR={open_mean:.4f}  "
             f"closed mean EAR={closed_mean:.4f}  threshold={threshold:.4f}")

    _save_calibration_graph(open_ears, closed_ears, threshold)
    return threshold

def train_classifier(X: np.ndarray, y: np.ndarray) -> dict:
    ensure_dirs(os.path.dirname(CLASSIFIER_PATH), GRAPH_DIR)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)

    log.info("Training SVM classifier …")
    train_losses, val_losses = [], []

    sizes = np.linspace(0.1, 1.0, 10)
    for sz in sizes:
        n = max(2, int(len(X_train) * sz))
        clf.fit(X_train[:n], y_train[:n])
        t_acc = accuracy_score(y_train[:n], clf.predict(X_train[:n]))
        v_acc = accuracy_score(y_test,      clf.predict(X_test))
        train_losses.append(1 - t_acc)
        val_losses.append(1 - v_acc)
        log.debug(f"  Subset {sz:.0%} → train_acc={t_acc:.3f}  val_acc={v_acc:.3f}")

    clf.fit(X_train, y_train)
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                      target_names=["Open", "Closed"])

    log.info(f"SVM Accuracy: {accuracy:.4f}")
    log.info(f"\n{report}")

    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"Classifier saved → {CLASSIFIER_PATH}")

    _save_training_graph(sizes, train_losses, val_losses)

    return {
        "classifier": clf,
        "scaler":     scaler,
        "accuracy":   accuracy,
        "report":     report,
    }

def load_classifier():
    with open(CLASSIFIER_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    log.info("Classifier and scaler loaded from disk.")
    return clf, scaler

def _save_training_graph(sizes, train_losses, val_losses) -> None:
    ensure_dirs(GRAPH_DIR)
    path = os.path.join(GRAPH_DIR, "svm_learning_curve.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes * 100, train_losses, "o-", label="Train Loss (1 - Acc)", color="#1f77b4")
    ax.plot(sizes * 100, val_losses,   "s--", label="Val Loss (1 - Acc)",   color="#ff7f0e")
    ax.set_xlabel("Training Subset (%)")
    ax.set_ylabel("Loss")
    ax.set_title("SVM Learning Curve — Eye State Classifier")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Training graph saved → {path}")

def _save_calibration_graph(open_ears, closed_ears, threshold) -> None:
    ensure_dirs(GRAPH_DIR)
    path = os.path.join(GRAPH_DIR, "ear_calibration.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(open_ears,   label="Open EAR",   color="green",   alpha=0.7)
    ax.plot(closed_ears, label="Closed EAR", color="red",     alpha=0.7)
    ax.axhline(threshold, color="blue", linestyle="--",
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Frame #")
    ax.set_ylabel("EAR Value")
    ax.set_title("EAR Calibration — Open vs Closed Eyes")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Calibration graph saved → {path}")

if __name__ == "__main__":
    np.random.seed(0)
    n = 200
    X_open   = np.random.normal(0.32, 0.02, (n, 1))
    X_closed = np.random.normal(0.18, 0.02, (n, 1))
    X = np.vstack([X_open, X_closed])
    y = np.array([0] * n + [1] * n)
    result = train_classifier(X, y)
    print(f"\nAccuracy: {result['accuracy']:.4f}")
    print(result["report"])