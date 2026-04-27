import os
import sys
import cv2

from config import INPUT_SOURCE, OUTPUT_VIDEO, FRAME_WIDTH, FRAME_HEIGHT
from preprocessing import load_detectors, prepare_input_source
from inference import run_inference, init_state
from utils import ensure_dirs, get_video_writer
from logger import log

def initialise() -> tuple:
    log.info("=== Eye Detection Based Alarm System — Starting ===")
    ensure_dirs(
        os.path.join(os.path.dirname(__file__), "logs"),
        os.path.join(os.path.dirname(__file__), "output"),
        os.path.join(os.path.dirname(__file__), "models"),
    )

    detector, predictor = load_detectors()
    cap = prepare_input_source(INPUT_SOURCE)
    return detector, predictor, cap

def main() -> None:
    try:
        detector, predictor, cap = initialise()
    except Exception as e:
        log.error(f"Initialisation failed: {e}")
        sys.exit(1)

    writer = get_video_writer(OUTPUT_VIDEO, fps=20.0,
                              width=FRAME_WIDTH, height=FRAME_HEIGHT)

    state = init_state()

    log.info("Entering main loop. Press 'q' to quit.")
    print("\n[INFO] System running. Press 'q' in the window to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame not received. Stream ended or camera disconnected.")
                break

            annotated_frame, meta = run_inference(frame, detector, predictor, state)

            cv2.imshow("Eye Detection Alarm System", annotated_frame)

            writer.write(annotated_frame)

            if state["counter"] % 30 == 0 and meta["ear"] is not None:
                log.info(
                    f"FPS={meta['fps']:.1f} | "
                    f"EAR={meta['ear']:.3f} | "
                    f"Closed={meta['eye_closed']} | "
                    f"InfTime={meta['inference_ms']:.1f}ms | "
                    f"Alarms={state['total_alarms']}"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("User pressed 'q'. Exiting.")
                break

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl-C).")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        log.info(f"Session ended. Total alarms triggered: {state['total_alarms']}")
        log.info(f"Output saved → {OUTPUT_VIDEO}")
        print(f"\n[INFO] Output saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()