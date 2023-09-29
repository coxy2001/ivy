"""
VCS entry point.
"""

from dotenv import load_dotenv

from util.blob import Blob

load_dotenv()

import settings
import sys
import cv2

from detectors.yolo import DarknetYOLODetector
from detectors.yolov8 import UltralyticsYOLODetector
from util.logger import init_logger
from util.logger import get_logger

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
    detection_interval = settings.DI
    mcdf = settings.MCDF
    mctf = settings.MCTF
    show_counts = settings.SHOW_COUNTS

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
        debug_window_size = settings.DEBUG_WINDOW_SIZE
        resized_frame = cv2.resize(frame, debug_window_size)
        cv2.imshow("Debug", resized_frame)
        cv2.waitKey(1)

    font = cv2.FONT_HERSHEY_DUPLEX
    line_type = cv2.LINE_AA
    hud_color = (0, 255, 0)
    colors = {
        "car": (255, 0, 0),
        "truck": (0, 0, 255),
    }

    blobs: list[Blob] = []
    for box, type, confidence in zip(*detector.get_bounding_boxes(frame)):
        blob = Blob(box, type, confidence, None)
        blobs.append(blob)
        (x, y, w, h) = [int(v) for v in blob.bounding_box]
        color = hud_color if blob.type is None else colors.get(blob.type, hud_color)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        object_label = f"T: {blob.type} ({blob.type_confidence:.2f})"
        cv2.putText(frame, object_label, (x, y - 5), font, 1, color, 2, line_type)

    quarter_1 = 0
    quarter_2 = 0
    quarter_3 = 0
    quarter_4 = 0
    for blob in blobs:
        x, y = blob.centroid
        if x < f_width / 2 and y < f_height / 2:
            quarter_1 += 1
        elif y < f_height / 2:
            quarter_2 += 1
        elif x < f_width / 2:
            quarter_3 += 1
        else:
            quarter_4 += 1

    print(f"Quarter 1: {quarter_1}")
    print(f"Quarter 2: {quarter_2}")
    print(f"Quarter 3: {quarter_3}")
    print(f"Quarter 4: {quarter_4}")

    # show counts
    # hud_color = (255, 0, 0)
    # offset = 1
    # for line, objects in self.counts.items():
    #     cv2.putText(frame, line, (10, 40 * offset), font, 1, hud_color, 2, line_type)
    #     for label, count in objects.items():
    #         offset += 1
    #         cv2.putText(frame, f"{label}: {count}", (10, 40 * offset), font, 1, hud_color, 2, line_type)
    #     offset += 2

    if not settings.HEADLESS:
        debug_window_size = settings.DEBUG_WINDOW_SIZE
        resized_frame = cv2.resize(frame, debug_window_size)
        cv2.imshow("Debug", resized_frame)
        cv2.waitKey()

    # end capture, close window, close log file and video object if any
    cap.release()
    if not settings.HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
