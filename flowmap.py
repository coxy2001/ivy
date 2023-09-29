"""
VCS entry point.
"""

from dotenv import load_dotenv

load_dotenv()

import cv2
import time
import settings
import sys

from datetime import datetime
from detectors.yolo import DarknetYOLODetector
from detectors.yolov8 import UltralyticsYOLODetector
from util.blob import Blob
from util.debugger import mouse_callback
from util.logger import init_logger, get_logger
from util.image import take_screenshot


init_logger()
logger = get_logger()


def run():
    """
    Initialize object counter class and run counting loop.
    """

    cap = cv2.VideoCapture(settings.VIDEO)
    if not cap.isOpened():
        logger.error(
            "Invalid video source %s",
            settings.VIDEO,
            extra={
                "meta": {"label": "INVALID_VIDEO_SOURCE"},
            },
        )
        sys.exit()
    retval, frame = cap.read()
    f_height, f_width, _ = frame.shape

    with open(settings.CLASSES_PATH, "r") as classes_file:
        classes = [line.strip() for line in classes_file.readlines()]
    with open(settings.CLASSES_OF_INTEREST_PATH, "r") as coi_file:
        classes_of_interest = [line.strip() for line in coi_file.readlines()]

    if settings.DETECTOR == "yolo":
        detector = DarknetYOLODetector(
            settings.YOLO_WEIGHTS_PATH,
            settings.YOLO_CONFIG_PATH,
            settings.CONFIDENCE_THRESHOLD,
            classes,
            classes_of_interest,
        )
    elif settings.DETECTOR == "yolov8":
        detector = UltralyticsYOLODetector(
            settings.YOLOV8_MODEL_PATH,
            settings.CONFIDENCE_THRESHOLD,
            classes_of_interest,
        )
    else:
        logger.error(
            "Invalid detector model, algorithm or API specified (options: yolo, yolov8)",
            extra={"meta": {"label": "INVALID_DETECTION_ALGORITHM"}},
        )
        sys.exit()

    if not settings.HEADLESS:
        cv2.namedWindow("Debug")
        cv2.setMouseCallback(
            "Debug", mouse_callback, {"frame_width": f_width, "frame_height": f_height}
        )
        resized_frame = cv2.resize(frame, settings.DEBUG_WINDOW_SIZE)
        cv2.imshow("Debug", resized_frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    line_type = cv2.LINE_AA
    hud_color = (0, 255, 0)
    colors = {
        "car": (255, 0, 0),
        "truck": (0, 0, 255),
    }

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
                for box, _type, _confidence in zip(*detector.get_bounding_boxes(frame)):
                    _tracker = cv2.TrackerKCF_create()
                    _tracker.init(frame, tuple(box))
                    blobs.append(Blob(box, _type, _confidence, _tracker))
            else:
                # Track blobs
                for blob in blobs:
                    success, box = blob.tracker.update(frame)
                    if success:
                        blob.update(box)

            frame_count += 1
            if frame_count >= 50:
                total_time = datetime.now() - start
                seconds = total_time.seconds + (total_time.microseconds / 1_000_000)
                print(f"Seconds: {seconds}")
                print(f"FPS {frame_count / seconds}")
                break

            # Render blobs
            if not settings.HEADLESS:
                for blob in blobs:
                    (x, y, w, h) = [int(v) for v in blob.bounding_box]
                    color = colors.get(blob.type, hud_color)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    object_label = (
                        f"I: {blob.id[:4]} T: {blob.type} ({blob.type_confidence:.2f})"
                    )
                    cv2.putText(
                        frame, object_label, (x, y - 5), font, 1, color, 2, line_type
                    )
                resized_frame = cv2.resize(frame, settings.DEBUG_WINDOW_SIZE)
                cv2.imshow("Debug", resized_frame)

            retval, frame = cap.read()
    finally:
        # end capture, close window, close log file and video object if any
        cap.release()

        for blob in blobs:
            cv2.line(frame, blob.position_first_detected, blob.centroid, hud_color, 2)
            cv2.circle(frame, blob.centroid, 5, hud_color, 5)

        resized_frame = cv2.resize(frame, settings.DEBUG_WINDOW_SIZE)
        cv2.imshow("Debug", resized_frame)
        cv2.waitKey()

        if not settings.HEADLESS:
            cv2.destroyAllWindows()


# Create or update blobs
# matched_blob_ids = []
# for box, _type, _confidence in zip(*detector.get_bounding_boxes(frame)):
#     _tracker = cv2.TrackerKCF_create()
#     _tracker.init(frame, tuple(box))
#     for blob in blobs:
#         if blob.get_overlap(box) >= 0.6:
#             if blob.id not in matched_blob_ids:
#                 blob.num_consecutive_detection_failures = 0
#                 matched_blob_ids.append(blob.id)
#             blob.update(box, _type, _confidence, _tracker)
#             break
#     else:
#         blobs.append(Blob(box, _type, _confidence, _tracker))

# for blob in list(blobs):
#     if blob.id not in matched_blob_ids:
#         blob.num_consecutive_detection_failures += 1
#     if blob.num_consecutive_detection_failures > settings.MCDF:
#         blobs.remove(blob)


if __name__ == "__main__":
    run()
