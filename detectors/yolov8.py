"""
Perform object detection using models created with the YOLO (You Only Look Once) neural net.
https://pjreddie.com/darknet/yolo/
"""

import settings

from ultralytics import YOLO


with open(settings.YOLOV8_CLASSES_PATH, "r") as classes_file:
    CLASSES = dict(enumerate([line.strip() for line in classes_file.readlines()]))
with open(settings.YOLOV8_CLASSES_OF_INTEREST_PATH, "r") as coi_file:
    CLASSES_OF_INTEREST = tuple([line.strip() for line in coi_file.readlines()])
conf_threshold = settings.YOLOV8_CONFIDENCE_THRESHOLD
model = YOLO(settings.YOLOV8_MODEL_PATH)


def get_bounding_boxes(image):
    """
    Return a list of bounding boxes of objects detected,
    their classes and the confidences of the detections made.
    """

    result = model.predict(image)[0]

    bounding_boxes = []
    classes = []
    confidences = []

    for box in result.boxes:
        conf = box.conf[0].item()
        if conf >= conf_threshold:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            x = int(cords[0])
            y = int(cords[1])
            w = int(cords[2]) - x
            h = int(cords[3]) - y
            bounding_boxes.append([x, y, w, h])
            classes.append(class_id)
            confidences.append(conf)

    return bounding_boxes, classes, confidences
