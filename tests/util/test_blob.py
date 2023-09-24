import cv2
from util.blob import Blob


def test_blob_creation():
    _bounding_box = [1, 1, 4, 4]
    _type = "car"
    _confidence = 0.99
    _tracker = cv2.TrackerKCF_create()
    blob = Blob(_bounding_box, _type, _confidence, _tracker)
    assert isinstance(blob, Blob), "blob is an instance of class Blob"
    assert blob.bounding_box == _bounding_box
    assert blob.type == _type
    assert blob.type_confidence == _confidence
    assert isinstance(
        blob.tracker, cv2.Tracker
    ), "blob tracker is an instance of OpenCV Tracker class"


def test_blob_update():
    _bounding_box = [1, 1, 4, 4]
    _type = "car"
    _confidence = 0.99
    _tracker = cv2.TrackerKCF_create()
    blob = Blob(_bounding_box, _type, _confidence, _tracker)

    _new_bounding_box = [2, 2, 5, 5]
    _new_type = "bus"
    _new_confidence = 0.35
    _new_tracker = cv2.TrackerCSRT_create()
    blob.update(_new_bounding_box, _new_type, _new_confidence, _new_tracker)

    assert blob.bounding_box == _new_bounding_box
    assert blob.type == _new_type
    assert blob.type_confidence == _new_confidence
    assert blob.tracker == _new_tracker


def test_get_centroid():
    bounding_box = [1, 1, 4, 4]
    centroid = Blob(bounding_box, None, None, None).get_centroid()
    assert type(centroid) is tuple, "centroid is a tuple"
    assert len(centroid) == 2, "centroid is a 2d coordinate (x, y)"
    assert (
        centroid[0] == 3 and centroid[1] == 3
    ), "the centroid (center point) of box [1, 1, 4, 4] is (3, 3)"


def test_box_contains_point():
    bounding_box = [1, 1, 4, 4]
    point1 = (2, 2)
    point2 = (0, 0)
    contains_point1 = Blob(bounding_box, None, None, None).box_contains_point(point1)
    contains_point2 = Blob(bounding_box, None, None, None).box_contains_point(point2)
    assert (
        type(contains_point1) is bool and type(contains_point2) is bool
    ), "return type is boolean"
    assert contains_point1 == True, "box [1, 1, 4, 4] contains point (2, 2)"
    assert contains_point2 == False, "box [1, 1, 4, 4] does not contain point (0, 0)"


def test_get_area():
    bounding_box = [1, 1, 4, 4]
    area = Blob(bounding_box, None, None, None).get_area()
    assert area == 16, "area of box [1, 1, 4, 4] is 16"
