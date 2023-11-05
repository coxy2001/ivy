from dotenv import load_dotenv

load_dotenv()

import cv2
import json
import os
import settings
import time

from datetime import datetime
from detectors import Detector
from detectors.yolo import DarknetYOLODetector
from detectors.yolov8 import UltralyticsYOLODetector
from pathlib import Path
from util.blob import Blob
from util.debugger import mouse_callback
from util.logger import init_logger, get_logger
from util.image import take_screenshot


init_logger()
logger = get_logger()


def get_detector():
    with open(settings.CLASSES_PATH, "r") as classes_file:
        classes = [line.strip() for line in classes_file.readlines()]
    with open(settings.CLASSES_OF_INTEREST_PATH, "r") as coi_file:
        classes_of_interest = [line.strip() for line in coi_file.readlines()]

    if settings.DETECTOR == "yolo":
        return DarknetYOLODetector(
            settings.YOLO_WEIGHTS_PATH,
            settings.YOLO_CONFIG_PATH,
            settings.CONFIDENCE_THRESHOLD,
            classes,
            classes_of_interest,
        )
    elif settings.DETECTOR == "yolov8":
        return UltralyticsYOLODetector(
            settings.YOLOV8_MODEL_PATH,
            settings.CONFIDENCE_THRESHOLD,
            classes_of_interest,
        )
    else:
        logger.error(
            "Invalid detector model, algorithm or API specified (options: yolo, yolov8)",
            extra={"meta": {"label": "INVALID_DETECTION_ALGORITHM"}},
        )
        return None


def main():
    """
    Load video and  heatmap and flowmap data
    """

    # Get detector
    detector = get_detector()
    if not detector:
        return

    VIDEO_INPUT_DIRECTORY = Path(settings.VIDEO_INPUT_DIRECTORY).resolve()
    VIDEO_OUTPUT_DIRECTORY = Path(settings.VIDEO_OUTPUT_DIRECTORY).resolve()
    DATA_OUTPUT_DIRECTORY = Path(settings.DATA_OUTPUT_DIRECTORY).resolve()

    while True:
        files = os.listdir(settings.VIDEO_INPUT_DIRECTORY)
        if files:
            file = Path(VIDEO_INPUT_DIRECTORY / files[0])
            result = process(str(file), detector)
            if result:
                with open(DATA_OUTPUT_DIRECTORY / (file.stem + ".json"), "w") as output:
                    output.write(json.dumps(result, indent=4))
                file.rename(VIDEO_OUTPUT_DIRECTORY / file.name)
            else:
                print("Some error occurred")

        time.sleep(1)


def process(video, detector: Detector):
    # Start capture
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        logger.error(
            "Invalid video source %s",
            video,
            extra={
                "meta": {"label": "INVALID_VIDEO_SOURCE"},
            },
        )
        return

    # Get first frame
    retval, frame = cap.read()
    f_height, f_width, _ = frame.shape

    # Render debug window
    if not settings.HEADLESS:
        cv2.namedWindow("Debug")
        cv2.setMouseCallback(
            "Debug", mouse_callback, {"frame_width": f_width, "frame_height": f_height}
        )
        cv2.imshow("Debug", cv2.resize(frame, settings.DEBUG_WINDOW_SIZE))

    # Debug render info
    hud_color = (0, 255, 0)

    # Process video
    start = datetime.now()
    is_paused = False
    frame_count = 0
    blobs: list[Blob] = []
    try:
        while retval:
            # Check key press
            if not settings.HEADLESS:
                k = cv2.waitKey(1) & 0xFF
                if k == ord("p"):
                    is_paused = not is_paused
                if k == ord("s") and frame is not None:
                    take_screenshot(frame)
                if k == ord("q"):
                    break

                if is_paused:
                    time.sleep(0.5)
                    continue

            if frame_count == 0:
                # Create blobs
                for box in detector.get_bounding_boxes(frame):
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, box.box)
                    blobs.append(Blob(box.box, box.type, box.confidence, tracker))
            else:
                # Track blobs
                for blob in blobs:
                    success, box = blob.tracker.update(frame)
                    if success:
                        blob.update(box)

            frame_count += 1

            logger.info(
                "Frame processed",
                extra={
                    "meta": {
                        "label": "FRAME_PROCESS",
                        "frames": frame_count,
                    },
                },
            )

            # Render blobs
            if not settings.HEADLESS:
                for blob in blobs:
                    (x, y, w, h) = [int(v) for v in blob.bounding_box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), hud_color, 2)
                    object_label = (
                        f"I: {blob.id[:4]} T: {blob.type} ({blob.type_confidence:.2f})"
                    )
                    cv2.putText(
                        frame,
                        object_label,
                        (x, y - 5),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        hud_color,
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("Debug", cv2.resize(frame, settings.DEBUG_WINDOW_SIZE))

            retval, frame = cap.read()
    finally:
        # End capture
        cap.release()

        # Logging
        total_time = datetime.now() - start
        seconds = total_time.total_seconds()
        logger.info(
            "Processing ended",
            extra={
                "meta": {
                    "label": "END_PROCESS",
                    "seconds": seconds,
                    "frames": frame_count,
                    "fps": frame_count / seconds,
                    "blobs": len(blobs),
                },
            },
        )

        # Save data
        vectors = []
        counts = {
            "quarter_1": 0,
            "quarter_2": 0,
            "quarter_3": 0,
            "quarter_4": 0,
        }
        for blob in blobs:
            vectors.append(f"{blob.position_first_detected},{blob.centroid}")
            x, y = blob.position_first_detected
            left = x < f_width / 2
            top = y < f_height / 2
            if left and top:
                counts["quarter_1"] += 1
            elif top:
                counts["quarter_2"] += 1
            elif left:
                counts["quarter_3"] += 1
            else:
                counts["quarter_4"] += 1

        # Render final result
        if not settings.HEADLESS:
            for blob in blobs:
                cv2.line(
                    frame, blob.position_first_detected, blob.centroid, hud_color, 2
                )
                cv2.circle(frame, blob.centroid, 5, hud_color, 5)
            cv2.imshow("Debug", cv2.resize(frame, settings.DEBUG_WINDOW_SIZE))
            cv2.waitKey()
            cv2.destroyAllWindows()

        return {"vectors": vectors, "counts": counts}


if __name__ == "__main__":
    main()
