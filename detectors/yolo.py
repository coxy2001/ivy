"""
Perform object detection using models created with the YOLO (You Only Look Once) neural net.
https://pjreddie.com/darknet/yolo/
"""

import cv2
import numpy as np


class DarknetYOLODetector:
    def __init__(
        self,
        weights_path,
        config_path,
        confidence_threshold,
        classes,
        classes_of_interest,
    ):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.confidence_threshold = confidence_threshold
        self.classes = tuple(classes)
        self.classes_of_interest = tuple(classes_of_interest)

    def get_bounding_boxes(self, image):
        """
        Return a list of bounding boxes of objects detected,
        their classes and the confidences of the detections made.
        """

        # create image blob
        scale = 0.00392
        image_blob = cv2.dnn.blobFromImage(
            image, scale, (416, 416), (0, 0, 0), True, crop=False
        )

        # detect objects
        self.net.setInput(image_blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)

        classes = []
        confidences = []
        boxes = []
        nms_threshold = 0.4

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (
                    confidence > self.confidence_threshold
                    and self.classes[class_id] in self.classes_of_interest
                ):
                    width = image.shape[1]
                    height = image.shape[0]
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    classes.append(self.classes[class_id])
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # remove overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, nms_threshold
        )

        _bounding_boxes = []
        _classes = []
        _confidences = []
        for i in indices:
            _bounding_boxes.append(boxes[i])
            _classes.append(classes[i])
            _confidences.append(confidences[i])

        return _bounding_boxes, _classes, _confidences
