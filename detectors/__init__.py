from dataclasses import dataclass
from typing import Protocol


@dataclass
class BoundingBox:
    box: tuple[int, int, int, int]
    type: str
    confidence: float


class Detector(Protocol):
    def get_bounding_boxes(self, image) -> list[BoundingBox]:
        ...
