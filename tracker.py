"""
Functions for keeping track of detected objects in a video.
"""

import cv2
import settings
import sys

from detectors import Detector
from util.blob import Blob
from util.image import get_base64_image
from util.logger import get_logger


logger = get_logger()


def _csrt_create(bounding_box, frame):
    """
    Create an OpenCV CSRT Tracker object.
    """
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bounding_box)
    return tracker


def _kcf_create(bounding_box, frame):
    """
    Create an OpenCV KCF Tracker object.
    """
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, bounding_box)
    return tracker


def get_tracker(algorithm, bounding_box, frame):
    """
    Fetch a tracker object based on the algorithm specified.
    """
    if algorithm == "csrt":
        return _csrt_create(bounding_box, frame)
    if algorithm == "kcf":
        return _kcf_create(bounding_box, frame)

    logger.error(
        "Invalid tracking algorithm specified (options: csrt, kcf)",
        extra={
            "meta": {"label": "INVALID_TRACKING_ALGORITHM"},
        },
    )
    sys.exit()


def _remove_stray_blobs(blobs: list[Blob], matched_blob_ids, mcdf):
    """
    Remove blobs that "hang" after a tracked object has left the frame.
    """
    for blob in list(blobs):
        if blob.id not in matched_blob_ids:
            blob.num_consecutive_detection_failures += 1
        if blob.num_consecutive_detection_failures > mcdf:
            blobs.remove(blob)
    return blobs


def add_new_blobs(
    detector: Detector, droi_frame, blobs: list[Blob], frame, tracker, mcdf
):
    """
    Add new blobs or updates existing ones.
    """
    matched_blob_ids = []
    for box in detector.get_bounding_boxes(droi_frame):
        _tracker = get_tracker(tracker, box.box, frame)

        for blob in blobs:
            if blob.get_overlap(box.box) >= 0.6:
                if blob.id not in matched_blob_ids:
                    blob.num_consecutive_detection_failures = 0
                    matched_blob_ids.append(blob.id)
                blob.update(box.box, box.type, box.confidence, _tracker)

                blob_update_log_meta = {
                    "label": "BLOB_UPDATE",
                    "object_id": blob.id,
                    "bounding_box": blob.bounding_box,
                    "type": blob.type,
                    "type_confidence": blob.type_confidence,
                }
                if settings.LOG_IMAGES:
                    blob_update_log_meta["image"] = get_base64_image(
                        blob.get_box_image(frame)
                    )
                logger.debug("Blob updated.", extra={"meta": blob_update_log_meta})
                break
        else:
            blob = Blob(box.box, box.type, box.confidence, _tracker)
            blobs.append(blob)

            blog_create_log_meta = {
                "label": "BLOB_CREATE",
                "object_id": blob.id,
                "bounding_box": blob.bounding_box,
                "type": blob.type,
                "type_confidence": blob.type_confidence,
            }
            if settings.LOG_IMAGES:
                blog_create_log_meta["image"] = get_base64_image(
                    blob.get_box_image(frame)
                )
            logger.debug("Blob created.", extra={"meta": blog_create_log_meta})

    blobs = _remove_stray_blobs(blobs, matched_blob_ids, mcdf)
    return blobs


def remove_duplicates(blobs: list[Blob]):
    """
    Remove duplicate blobs i.e blobs that point to an already detected and tracked object.
    """
    for blob_a in list(blobs):
        for blob_b in list(blobs):
            if blob_a == blob_b:
                break

            if blob_a.get_overlap(blob_b.bounding_box) >= 0.6 and blob_a in blobs:
                blobs.remove(blob_a)
    return blobs


def update_blob_tracker(blob: Blob, frame):
    """
    Update a blob's tracker object.
    """
    success, box = blob.tracker.update(frame)
    if success:
        blob.num_consecutive_tracking_failures = 0
        blob.update(box)
        logger.debug(
            "Object tracker updated.",
            extra={
                "meta": {
                    "label": "TRACKER_UPDATE",
                    "object_id": blob.id,
                    "bounding_box": blob.bounding_box,
                    "centroid": blob.centroid,
                },
            },
        )
    else:
        blob.num_consecutive_tracking_failures += 1

    return blob
