import uuid


class Blob:
    """
    A blob represents a tracked object as it moves around in a video.
    """

    def __init__(self, _bounding_box, _type, _confidence, _tracker):
        self.bounding_box = _bounding_box
        self.type = _type
        self.type_confidence = _confidence
        self.tracker = _tracker
        self.id = uuid.uuid4().hex
        self.centroid = self.get_centroid()
        self.area = self.get_area()
        self.num_consecutive_tracking_failures = 0
        self.num_consecutive_detection_failures = 0
        self.lines_crossed = []  # list of counting lines crossed by an object
        self.position_first_detected = tuple(self.centroid)
        self.classifications = []

    def update(self, _bounding_box, _type=None, _confidence=None, _tracker=None):
        self.bounding_box = _bounding_box
        if _type is not None:
            self.type = _type
        if _confidence is not None:
            self.type_confidence = _confidence
        if _tracker:
            self.tracker = _tracker
        self.centroid = self.get_centroid()
        self.area = self.get_area()
        # self.classifications.append((self.type, self.type_confidence))

    def classification(self):
        counts = {}
        for type, confidence in self.classifications:
            if type not in counts:
                counts[type] = 0
            counts[type] += 1
        print(self.classifications)
        print(counts)
        return max(counts, key=counts.get)

    def box_contains_point(self, point):
        """
        Checks if a given point is within a bounding box.
        """
        x, y, w, h = self.bounding_box
        px, py = point
        return x < px < x + w and y < py < y + h

    def get_area(self):
        """
        Calculates the area of a bounding box.
        """
        _, _, w, h = self.bounding_box
        return w * h

    def get_centroid(self):
        """
        Calculates the center point of a bounding box.
        """
        x, y, w, h = self.bounding_box
        return round(x + (w / 2)), round(y + (h / 2))

    def get_overlap(self, bbox2):
        """
        Calculates the degree of overlap of two bounding boxes.
        This can be any value from 0 to 1 where 0 means no overlap and 1 means complete overlap.
        The degree of overlap is the ratio of the area of overlap of two boxes and the area of the smaller box.
        """

        bbox1_x1, bbox1_y1, bbox1_w, bbox1_h = self.bounding_box
        bbox1_x2 = bbox1_x1 + bbox1_w
        bbox1_y2 = bbox1_y1 + bbox1_h

        bbox2_x1, bbox2_y1, bbox2_w, bbox2_h = bbox2
        bbox2_x2 = bbox2_x1 + bbox2_w
        bbox2_y2 = bbox2_y1 + bbox2_h

        overlap_x1 = max(bbox1_x1, bbox2_x1)
        overlap_y1 = max(bbox1_y1, bbox2_y1)
        overlap_x2 = min(bbox1_x2, bbox2_x2)
        overlap_y2 = min(bbox1_y2, bbox2_y2)

        overlap_width = overlap_x2 - overlap_x1
        overlap_height = overlap_y2 - overlap_y1

        if overlap_width < 0 or overlap_height < 0:
            return 0.0

        overlap_area = overlap_width * overlap_height
        smaller_area = min(self.area, bbox2_w * bbox2_h)

        epsilon = 1e-5  # small value to prevent division by zero
        return overlap_area / (smaller_area + epsilon)

    def get_box_image(self, frame):
        """
        Fetches the image of the area covered by a bounding box with a 10 pixel padding
        """
        x, y, w, h = list(map(int, self.bounding_box))
        return frame[y - 10 : y + h + 10, x - 10 : x + w + 10]
