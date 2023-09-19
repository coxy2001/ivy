"""
Detectors entry point.
"""

# pylint: disable=import-outside-toplevel

import sys
from util.logger import get_logger


logger = get_logger()


def get_bounding_boxes(frame, model):
    """
    Run object detection algorithm and return a list of bounding boxes and other metadata.
    """
    if model == "yolo":
        from detectors.yolo import get_bounding_boxes as detector
    elif model == "yolov8":
        from detectors.yolov8 import get_bounding_boxes as detector
    else:
        logger.error(
            "Invalid detector model, algorithm or API specified (options: yolo, yolov8)",
            extra={"meta": {"label": "INVALID_DETECTION_ALGORITHM"}},
        )
        sys.exit()

    return detector(frame)
