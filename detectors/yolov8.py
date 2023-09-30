"""
Perform object detection using models created with the YOLO (You Only Look Once) neural net.
https://docs.ultralytics.com/
"""

from . import BoundingBox
from ultralytics import YOLO


class UltralyticsYOLODetector:
    def __init__(self, model_path, confidence_threshold, classes_of_interest1):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.classes_of_interest = tuple(classes_of_interest1)

    def get_bounding_boxes(self, image) -> list[BoundingBox]:
        """
        Return a list of bounding boxes of objects detected,
        their classes and the confidences of the detections made.
        """

        result = self.model.predict(image, verbose=False)[0]
        bounding_boxes = []
        for box in result.boxes:
            class_name = result.names[box.cls[0].item()]
            confidence = box.conf[0].item()
            if (
                confidence > self.confidence_threshold
                and class_name in self.classes_of_interest
            ):
                cords = box.xyxy[0].tolist()
                x = int(cords[0])
                y = int(cords[1])
                w = int(cords[2]) - x
                h = int(cords[3]) - y
                bounding_boxes.append(BoundingBox((x, y, w, h), class_name, confidence))

        return bounding_boxes
